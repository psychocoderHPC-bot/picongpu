/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Artifact loader for the neural bunch-field model, format v1.
 *
 * Every check of MODEL-FORMAT.md that can be decided from metadata plus the payload file is performed here,
 * before any tensor storage is allocated:
 *
 *  - required keys, format name and version,
 *  - dtype/endianness, tile geometry and flatten convention,
 *  - feature specification (``fourier_v1``, octaves [-3,6], 69 inputs),
 *  - encoder layer chain and latent rank,
 *  - decoder axis grids, bank ordering and matrix shape,
 *  - domain bounds and finite / in-range numeric metadata,
 *  - tensor table (alignment, monotonicity, overlap, bounds, shape/count),
 *  - artifact byte count and SHA-256.
 *
 * Tensor payloads are read with ``alpaka::nn::onHost::loadBinaryTensor`` so the same code path serves Host, CUDA,
 * HIP and oneAPI builds. ``Model::load`` itself performs no allocation of tensor storage; a corrupt artifact is
 * rejected before ``loadTensors()`` is ever called.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/core/binary.hpp>

#include "Features.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace picongpu::particles::neuralBunch
{
    inline constexpr std::size_t kOutDim = 1536; // 8 * 8 * 4 cells * 6 components
    inline constexpr std::size_t kTileNx = 8;
    inline constexpr std::size_t kTileNy = 8;
    inline constexpr std::size_t kTileNz = 4;

    /** Thrown for every malformed or unsupported artifact. */
    class ModelError : public std::runtime_error
    {
    public:
        explicit ModelError(std::string const& what) : std::runtime_error{"neuralBunch model: " + what}
        {
        }
    };

    /** One row of the artifact tensor table. */
    struct TensorBinding
    {
        std::string name;
        std::size_t offsetBytes{};
        std::size_t count{};
        std::vector<std::size_t> shape;
    };

    /** One encoder layer as described by ``encoder.layers``. */
    struct EncoderLayer
    {
        std::string weight;
        std::string bias;
        std::size_t in{};
        std::size_t out{};
    };

    /** Physical normalisation of the decoder output.
     *
     * MODEL-FORMAT v1 defines ``E = e/(4*pi*eps0*L^2) * E_tilde`` and
     * ``B = e/(4*pi*eps0*c*L^2) * B_tilde`` with ``L`` the length unit (``dx``). The artifact stores the formula,
     * not the numerical value of ``dx``; the runtime supplies it here. ``lengthMeters == 1`` means "one length
     * unit", which is what the stand-alone parity harness uses.
     */
    struct FieldNormalization
    {
        double lengthMeters = 1.0;

        static constexpr double elementaryCharge = 1.602176634e-19;
        static constexpr double vacuumPermittivity = 8.8541878128e-12;
        static constexpr double speedOfLight = 299792458.0;

        [[nodiscard]] double scaleE() const
        {
            return elementaryCharge / (4.0 * std::numbers::pi * vacuumPermittivity * lengthMeters * lengthMeters);
        }

        [[nodiscard]] double scaleB() const
        {
            return scaleE() / speedOfLight;
        }
    };

    /** Everything the runtime needs besides the tensor payloads. */
    struct ModelMetadata
    {
        std::string format;
        int version{};
        std::string dtype;
        std::string endianness;
        std::string createdUtc;
        std::string codeSha;
        std::string license;

        std::size_t tileNx{};
        std::size_t tileNy{};
        std::size_t tileNz{};
        std::array<std::string, 6> components;
        std::string flatten;

        int octaveLow{};
        int octaveHigh{};
        std::vector<std::string> featureInputs;
        std::size_t featureDim{};
        FeatureKind featureKind = FeatureKind::fourierV1;
        std::vector<std::array<double, 3>> probeCellOffsets;
        std::array<std::string, 6> probeComponents;
        std::string probeGrid;
        std::string probeField;
        std::string featureOrdering;

        std::string encoderKind;
        std::string encoderActivation;
        std::string lastLayerActivation;
        std::vector<EncoderLayer> layers;
        std::size_t latent{};

        std::string decoderKind;
        std::string interpolation;
        std::vector<double> aYAnchors;
        std::vector<double> aZAnchors;
        std::array<std::size_t, 2> matrixShape{};
        std::array<std::size_t, 2> bankEntryShape{};
        bool decoderBias = false;
        std::vector<std::string> bank;

        double gammaMin{};
        double gammaMax{};
        double aYMin{};
        double aYMax{};
        double aZMin{};
        double aZMax{};
        double dMinCells{};
        double dMaxCells{};
        std::string distanceMetric;

        std::string normalizationLength;
        std::string normalizationCharge;
        std::string normalizationFieldE;
        std::string normalizationFieldB;
        std::string normalizationLatent;

        std::vector<TensorBinding> tensors;
        std::string artifactFile;
        std::size_t artifactBytes{};
        std::string artifactSha256;

        /** Index into ``tensors`` for a referenced name, or throws. */
        [[nodiscard]] std::size_t tensorIndex(std::string const& name) const
        {
            for(std::size_t i = 0; i < tensors.size(); ++i)
                if(tensors[i].name == name)
                    return i;
            throw ModelError{"tensor table has no entry named '" + name + "'."};
        }
    };

    namespace detail
    {
        using json = nlohmann::json;

        inline json const& required(json const& object, char const* key, std::string const& context)
        {
            if(!object.contains(key))
                throw ModelError{context + ": missing required key '" + key + "'."};
            return object.at(key);
        }

        inline std::string asString(json const& value, std::string const& context)
        {
            if(!value.is_string())
                throw ModelError{context + ": expected a string."};
            return value.get<std::string>();
        }

        inline std::size_t asSize(json const& value, std::string const& context)
        {
            if(!value.is_number_integer() && !value.is_number_unsigned())
                throw ModelError{context + ": expected an integer."};
            if(value.is_number_unsigned())
                return value.get<std::size_t>();
            auto const signedValue = value.get<long long>();
            if(signedValue < 0)
                throw ModelError{context + ": expected a non-negative integer."};
            return static_cast<std::size_t>(signedValue);
        }

        inline int asInt(json const& value, std::string const& context)
        {
            auto const number = asSize(value, context);
            if(number > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw ModelError{context + ": integer out of range."};
            return static_cast<int>(number);
        }

        inline int asSignedInt(json const& value, std::string const& context)
        {
            if(!value.is_number_integer())
                throw ModelError{context + ": expected an integer."};
            auto const number = value.get<long long>();
            if(number < std::numeric_limits<int>::min() || number > std::numeric_limits<int>::max())
                throw ModelError{context + ": integer out of range."};
            return static_cast<int>(number);
        }

        inline double asFiniteDouble(json const& value, std::string const& context)
        {
            if(!value.is_number())
                throw ModelError{context + ": expected a number."};
            auto const number = value.get<double>();
            if(!std::isfinite(number))
                throw ModelError{context + ": value is not finite."};
            return number;
        }

        inline std::vector<std::size_t> asShape(json const& value, std::string const& context)
        {
            if(!value.is_array())
                throw ModelError{context + ": expected an array."};
            std::vector<std::size_t> shape;
            shape.reserve(value.size());
            for(auto const& entry : value)
                shape.push_back(asSize(entry, context));
            return shape;
        }

        inline std::vector<double> asDoubleVector(json const& value, std::string const& context)
        {
            if(!value.is_array())
                throw ModelError{context + ": expected an array."};
            std::vector<double> result;
            result.reserve(value.size());
            for(auto const& entry : value)
                result.push_back(asFiniteDouble(entry, context));
            return result;
        }

        inline std::size_t product(std::vector<std::size_t> const& shape, std::string const& context)
        {
            std::size_t result = 1u;
            for(auto const extent : shape)
            {
                if(extent != 0u && result > std::numeric_limits<std::size_t>::max() / extent)
                    throw ModelError{context + ": shape product overflows std::size_t."};
                result *= extent;
            }
            return result;
        }

        inline void requireSize(std::size_t actual, std::size_t expected, std::string const& context)
        {
            if(actual != expected)
                throw ModelError{
                    context + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual) + "."};
        }
    } // namespace detail

    class Model
    {
    public:
        /** Parse, validate and hash an artifact. No tensor memory is allocated. */
        static Model load(std::filesystem::path prefix)
        {
            std::filesystem::path jsonPath = prefix;
            if(jsonPath.extension() != ".json")
                jsonPath += ".json";

            std::error_code error;
            if(!std::filesystem::is_regular_file(jsonPath, error))
                throw ModelError{"metadata file '" + jsonPath.string() + "' does not exist."};

            detail::json root;
            try
            {
                std::ifstream stream{jsonPath};
                root = detail::json::parse(stream);
            }
            catch(std::exception const& exception)
            {
                throw ModelError{"cannot parse '" + jsonPath.string() + "': " + exception.what()};
            }
            if(!root.is_object())
                throw ModelError{"'" + jsonPath.string() + "' is not a JSON object."};

            Model model;
            model.m_jsonPath = jsonPath;
            model.parseMetadata(root);
            model.verifyPayload();

            return model;
        }

        [[nodiscard]] ModelMetadata const& metadata() const
        {
            return m_metadata;
        }

        [[nodiscard]] std::filesystem::path const& jsonPath() const
        {
            return m_jsonPath;
        }

        [[nodiscard]] std::filesystem::path const& binPath() const
        {
            return m_binPath;
        }

        /** Load every tensor into a device buffer, in tensor-table order.
         *
         * The artifact was already hashed by ``load()``; the per-tensor call therefore passes an empty digest and
         * only re-checks bounds and shape/count. Callers must synchronize the queue before consuming the buffers.
         */
        template<typename TQueue>
        auto loadTensors(TQueue& queue) const
        {
            using Buffer = decltype(alpaka::nn::onHost::loadBinaryTensor<float>(
                std::declval<TQueue&>(),
                std::declval<std::string const&>(),
                std::declval<alpaka::nn::onHost::TensorSpec const&>(),
                std::declval<std::string const&>()));

            std::vector<Buffer> buffers;
            buffers.reserve(m_metadata.tensors.size());
            for(auto const& binding : m_metadata.tensors)
            {
                alpaka::nn::onHost::TensorSpec spec{
                    binding.name,
                    binding.offsetBytes,
                    binding.count,
                    binding.shape};
                buffers.push_back(
                    alpaka::nn::onHost::loadBinaryTensor<float>(queue, m_binPath.string(), spec, std::string{}));
            }
            alpaka::onHost::wait(queue);
            return buffers;
        }

        /** Host copy of every tensor, only used by the CPU reference implementation in tests. */
        [[nodiscard]] std::map<std::string, std::vector<float>> loadTensorsHost() const
        {
            std::ifstream file{m_binPath, std::ios::binary};
            if(!file)
                throw ModelError{"cannot open payload '" + m_binPath.string() + "'."};

            std::map<std::string, std::vector<float>> result;
            for(auto const& binding : m_metadata.tensors)
            {
                std::vector<float> data(binding.count);
                file.seekg(static_cast<std::streamoff>(binding.offsetBytes));
                file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * 4u));
                if(!file)
                    throw ModelError{"short read for tensor '" + binding.name + "'."};
                result.emplace(binding.name, std::move(data));
            }
            return result;
        }

    private:
        ModelMetadata m_metadata;
        std::filesystem::path m_jsonPath;
        std::filesystem::path m_binPath;

        using json = detail::json;

        void parseMetadata(json const& root)
        {
            using detail::asDoubleVector;
            using detail::asFiniteDouble;
            using detail::asInt;
            using detail::asShape;
            using detail::asSize;
            using detail::asString;
            using detail::required;

            m_metadata.format = asString(required(root, "format", "root"), "format");
            if(m_metadata.format != "picongpu-neural-bunch")
                throw ModelError{"unknown format '" + m_metadata.format + "'."};

            auto const versionJson = required(root, "version", "root");
            auto const version = asInt(versionJson, "version");
            if(version != 1)
                throw ModelError{"unsupported artifact version " + std::to_string(version) + " (only 1 is supported)."};
            m_metadata.version = version;

            m_metadata.dtype = asString(required(root, "dtype", "root"), "dtype");
            if(m_metadata.dtype != "float32")
                throw ModelError{"unsupported dtype '" + m_metadata.dtype + "'."};
            m_metadata.endianness = asString(required(root, "endianness", "root"), "endianness");
            if(m_metadata.endianness != "little")
                throw ModelError{"unsupported endianness '" + m_metadata.endianness + "'."};

            m_metadata.createdUtc = asString(required(root, "created_utc", "root"), "created_utc");
            m_metadata.codeSha = asString(required(root, "code_sha", "root"), "code_sha");
            if(m_metadata.codeSha.size() != 40u)
                throw ModelError{"code_sha must contain 40 hexadecimal characters."};
            m_metadata.license = root.contains("license") ? asString(root.at("license"), "license") : std::string{};

            parseTile(required(root, "tile", "root"));
            parseFeatures(required(root, "features", "root"));
            parseEncoder(required(root, "encoder", "root"));
            parseDecoder(required(root, "decoder", "root"));
            parseDomain(required(root, "domain", "root"));
            parseNormalization(required(root, "normalization", "root"));
            parseTensors(required(root, "tensors", "root"));
            parseArtifact(required(root, "artifact", "root"));
        }

        void parseTile(json const& tile)
        {
            using detail::asSize;
            using detail::asString;
            using detail::required;

            m_metadata.tileNx = asSize(required(tile, "nx", "tile"), "tile.nx");
            m_metadata.tileNy = asSize(required(tile, "ny", "tile"), "tile.ny");
            m_metadata.tileNz = asSize(required(tile, "nz", "tile"), "tile.nz");
            if(m_metadata.tileNx != kTileNx || m_metadata.tileNy != kTileNy || m_metadata.tileNz != kTileNz)
                throw ModelError{
                    "unsupported tile geometry " + std::to_string(m_metadata.tileNx) + "x"
                    + std::to_string(m_metadata.tileNy) + "x" + std::to_string(m_metadata.tileNz) + " (v1 requires 8x8x4)."};

            auto const components = required(tile, "components", "tile");
            if(!components.is_array() || components.size() != 6u)
                throw ModelError{"tile.components must list six components."};
            std::array<std::string, 6> const expected{"Ex", "Ey", "Ez", "Bx", "By", "Bz"};
            for(std::size_t i = 0; i < 6u; ++i)
            {
                m_metadata.components[i] = asString(components.at(i), "tile.components");
                if(m_metadata.components[i] != expected[i])
                    throw ModelError{"tile.components[" + std::to_string(i) + "] must be '" + expected[i] + "'."};
            }

            m_metadata.flatten = asString(required(tile, "flatten", "tile"), "tile.flatten");
            if(m_metadata.flatten != "c + 6*(i + nx*(j + ny*k))")
                throw ModelError{"unsupported tile.flatten convention."};
        }

        static std::vector<std::string> baseInputNames()
        {
            return {"r_x", "r_y", "r_z", "u_x", "u_y", "u_z", "gamma", "a_y", "a_z"};
        }

        static std::string expectedProbeGrid(FeatureKind kind)
        {
            if(kind == FeatureKind::fourierProbeV1)
                return "4x4x2 tile cell centres: x=2i+0.5 (i=0..3), y=(2j+0.5)*a_y (j=0..3), "
                       "z=(2k+0.5)*a_z (k=0..1)";
            return "one probe at every tile cell centre: x=i+0.5 (i=0..7), y=(j+0.5)*a_y (j=0..7), "
                   "z=(k+0.5)*a_z (k=0..3), i fastest then j then k";
        }

        static std::string expectedProbeField()
        {
            return "per unit q/e: E_n=4*pi*eps0*dx^2*E/q, B_n=4*pi*eps0*c*dx^2*B/q; "
                   "stable uniform-motion decomposition (1/gamma^2 direct, static branch)";
        }

        static std::string expectedOrdering(FeatureKind kind)
        {
            if(kind == FeatureKind::fourierProbeV1)
                return "9 base, 32 probes x [Ex,Ey,Ez,Bx,By,Bz], 3 axes x octaves -3..1 x (sin,cos)";
            return "9 base, 256 probes x [Ex,Ey,Ez,Bx,By,Bz] in i-fastest cell order; no Fourier terms";
        }

        void parseFeatures(json const& features)
        {
            using detail::asSize;
            using detail::asString;
            using detail::required;

            auto const kindName = asString(required(features, "kind", "features"), "features.kind");
            auto const kind = featureKindFromString(kindName);
            if(!kind)
                throw ModelError{"unsupported feature kind '" + kindName + "'."};
            m_metadata.featureKind = *kind;

            auto const inputs = required(features, "inputs", "features");
            if(!inputs.is_array())
                throw ModelError{"features.inputs must be an array."};
            auto const expectedInputs = baseInputNames();
            if(inputs.size() != expectedInputs.size())
                throw ModelError{"features.inputs must list the nine base inputs."};
            for(std::size_t i = 0; i < expectedInputs.size(); ++i)
            {
                m_metadata.featureInputs.push_back(asString(inputs.at(i), "features.inputs"));
                if(m_metadata.featureInputs.back() != expectedInputs[i])
                    throw ModelError{"features.inputs[" + std::to_string(i) + "] must be '" + expectedInputs[i] + "'."};
            }

            m_metadata.featureDim = asSize(required(features, "dim", "features"), "features.dim");
            auto const expectedDim = featureDimension(*kind);
            if(m_metadata.featureDim != expectedDim)
                throw ModelError{
                    "features.dim must be " + std::to_string(expectedDim) + " for " + kindName + ", got "
                    + std::to_string(m_metadata.featureDim) + "."};

            auto const octaves = required(features, "octaves", "features");
            if(!octaves.is_array())
                throw ModelError{"features.octaves must be an array."};
            if(*kind == FeatureKind::probeDenseV1)
            {
                if(!octaves.empty())
                    throw ModelError{"probe_dense_v1 requires features.octaves = []."};
            }
            else
            {
                if(octaves.size() != 2u)
                    throw ModelError{"features.octaves must contain two integers."};
                auto const low = detail::asSignedInt(octaves.at(0), "features.octaves[0]");
                auto const high = detail::asSignedInt(octaves.at(1), "features.octaves[1]");
                auto const expectedLow = *kind == FeatureKind::fourierV1 ? kOctaveLow : kProbeOctaveLow;
                auto const expectedHigh = *kind == FeatureKind::fourierV1 ? kOctaveHigh : kProbeOctaveHigh;
                if(low != expectedLow || high != expectedHigh)
                    throw ModelError{
                        std::string{"features.octaves for "} + kindName + " must be [" + std::to_string(expectedLow)
                        + ", " + std::to_string(expectedHigh) + "]."};
                m_metadata.octaveLow = low;
                m_metadata.octaveHigh = high;
            }

            if(*kind != FeatureKind::fourierV1)
                parseProbe(features, *kind);
        }

        void parseProbe(json const& features, FeatureKind kind)
        {
            using detail::asFiniteDouble;
            using detail::required;

            auto const probe = required(features, "probe", "features");
            if(!probe.is_object())
                throw ModelError{"features.probe must be an object for probe feature kinds."};

            auto const components = required(probe, "components", "features.probe");
            std::array<std::string, 6> const expectedComponents{"Ex", "Ey", "Ez", "Bx", "By", "Bz"};
            if(!components.is_array() || components.size() != 6u)
                throw ModelError{"features.probe.components must list six components."};
            for(std::size_t i = 0; i < 6u; ++i)
            {
                m_metadata.probeComponents[i]
                    = detail::asString(components.at(i), "features.probe.components");
                if(m_metadata.probeComponents[i] != expectedComponents[i])
                    throw ModelError{
                        "features.probe.components[" + std::to_string(i) + "] must be '" + expectedComponents[i]
                        + "'."};
            }

            auto const offsetsJson = required(probe, "cell_offsets", "features.probe");
            auto const expectedCount = featureProbeCount(kind);
            if(!offsetsJson.is_array() || offsetsJson.size() != expectedCount)
                throw ModelError{
                    "features.probe.cell_offsets must contain " + std::to_string(expectedCount) + " entries."};
            auto compiledOffset = [&](std::size_t p, std::size_t c) -> double
            {
                return kind == FeatureKind::fourierProbeV1 ? kProbeCellOffsets[p][c] : kDenseCellOffsets[p][c];
            };
            for(std::size_t p = 0; p < expectedCount; ++p)
            {
                auto const& entry = offsetsJson.at(p);
                if(!entry.is_array() || entry.size() != 3u)
                    throw ModelError{"features.probe.cell_offsets[" + std::to_string(p) + "] must be [x,y,z]."};
                std::array<double, 3> offset{};
                for(std::size_t c = 0; c < 3u; ++c)
                {
                    offset[c] = asFiniteDouble(
                        entry.at(c),
                        "features.probe.cell_offsets[" + std::to_string(p) + "]");
                    if(std::abs(offset[c] - compiledOffset(p, c)) > 1e-12)
                        throw ModelError{
                            "features.probe.cell_offsets[" + std::to_string(p)
                            + "] does not match the compiled probe grid."};
                }
                m_metadata.probeCellOffsets.push_back(offset);
            }

            m_metadata.probeGrid = detail::asString(required(probe, "grid", "features.probe"), "features.probe.grid");
            if(m_metadata.probeGrid != expectedProbeGrid(kind))
                throw ModelError{"features.probe.grid does not match the compiled probe grid."};
            m_metadata.probeField
                = detail::asString(required(probe, "field", "features.probe"), "features.probe.field");
            if(m_metadata.probeField != expectedProbeField())
                throw ModelError{"features.probe.field does not match the compiled point-field formula."};

            m_metadata.featureOrdering
                = detail::asString(required(features, "ordering", "features"), "features.ordering");
            if(m_metadata.featureOrdering != expectedOrdering(kind))
                throw ModelError{"features.ordering does not match the compiled ordering."};
        }

        void parseEncoder(json const& encoder)
        {
            using detail::asSize;
            using detail::asString;
            using detail::required;

            m_metadata.encoderKind = asString(required(encoder, "kind", "encoder"), "encoder.kind");
            if(m_metadata.encoderKind != "mlp")
                throw ModelError{"unsupported encoder kind '" + m_metadata.encoderKind + "'."};
            m_metadata.encoderActivation
                = asString(required(encoder, "activation", "encoder"), "encoder.activation");
            if(m_metadata.encoderActivation != "silu")
                throw ModelError{"encoder.activation must be 'silu'."};
            m_metadata.lastLayerActivation
                = asString(required(encoder, "last_layer_activation", "encoder"), "encoder.last_layer_activation");
            if(m_metadata.lastLayerActivation != "none")
                throw ModelError{"encoder.last_layer_activation must be 'none'."};

            auto const layers = required(encoder, "layers", "encoder");
            if(!layers.is_array() || layers.empty())
                throw ModelError{"encoder.layers must be a non-empty array."};

            std::size_t expectedIn = m_metadata.featureDim;
            for(std::size_t i = 0; i < layers.size(); ++i)
            {
                auto const context = "encoder.layers[" + std::to_string(i) + "]";
                auto const& layer = layers.at(i);
                EncoderLayer parsed;
                parsed.weight = asString(required(layer, "weight", context), context + ".weight");
                parsed.bias = asString(required(layer, "bias", context), context + ".bias");
                parsed.in = asSize(required(layer, "in", context), context + ".in");
                parsed.out = asSize(required(layer, "out", context), context + ".out");
                if(parsed.in != expectedIn)
                    throw ModelError{
                        context + ": in=" + std::to_string(parsed.in) + " does not match the previous out="
                        + std::to_string(expectedIn) + "."};
                if(parsed.out == 0u)
                    throw ModelError{context + ": out must be positive."};
                m_metadata.layers.push_back(std::move(parsed));
                expectedIn = m_metadata.layers.back().out;
            }

            m_metadata.latent = asSize(required(encoder, "output_dim", "encoder"), "encoder.output_dim");
            if(m_metadata.latent == 0u)
                throw ModelError{"encoder.output_dim must be positive."};
            if(expectedIn != m_metadata.latent)
                throw ModelError{"encoder.output_dim does not match the last layer out extent."};
        }

        void parseDecoder(json const& decoder)
        {
            using detail::asDoubleVector;
            using detail::asShape;
            using detail::asSize;
            using detail::asString;
            using detail::required;

            m_metadata.decoderKind = asString(required(decoder, "kind", "decoder"), "decoder.kind");
            if(m_metadata.decoderKind != "geometry_bank_bilinear")
                throw ModelError{"unsupported decoder kind '" + m_metadata.decoderKind + "'."};
            m_metadata.interpolation
                = asString(required(decoder, "interpolation", "decoder"), "decoder.interpolation");

            auto const axes = required(decoder, "axes", "decoder");
            m_metadata.aYAnchors = asDoubleVector(required(axes, "a_y", "decoder.axes"), "decoder.axes.a_y");
            m_metadata.aZAnchors = asDoubleVector(required(axes, "a_z", "decoder.axes"), "decoder.axes.a_z");
            validateAnchors(m_metadata.aYAnchors, "decoder.axes.a_y");
            validateAnchors(m_metadata.aZAnchors, "decoder.axes.a_z");

            auto const matrixShape = asShape(required(decoder, "matrix_shape", "decoder"), "decoder.matrix_shape");
            if(matrixShape.size() != 2u)
                throw ModelError{"decoder.matrix_shape must contain two extents."};
            m_metadata.matrixShape = {matrixShape[0], matrixShape[1]};
            if(m_metadata.matrixShape[0] != m_metadata.latent || m_metadata.matrixShape[1] != kOutDim)
                throw ModelError{"decoder.matrix_shape must be [latent, 1536]."};

            auto const bias = required(decoder, "bias", "decoder");
            if(!bias.is_boolean() || bias.get<bool>())
                throw ModelError{"format v1 requires decoder.bias = false."};
            m_metadata.decoderBias = false;

            auto const bank = required(decoder, "bank", "decoder");
            if(!bank.is_array())
                throw ModelError{"decoder.bank must be an array."};
            for(auto const& entry : bank)
                m_metadata.bank.push_back(asString(entry, "decoder.bank"));

            auto const entryShape = asShape(required(decoder, "bank_entry_shape", "decoder"), "decoder.bank_entry_shape");
            if(entryShape.size() != 2u)
                throw ModelError{"decoder.bank_entry_shape must contain two extents."};
            m_metadata.bankEntryShape = {entryShape[0], entryShape[1]};
            if(m_metadata.bankEntryShape[0] != m_metadata.latent || m_metadata.bankEntryShape[1] != kOutDim)
                throw ModelError{"decoder.bank_entry_shape must be [latent, 1536]."};

            auto const expectedBank = m_metadata.aYAnchors.size() * m_metadata.aZAnchors.size();
            if(m_metadata.bank.size() != expectedBank)
                throw ModelError{
                    "decoder.bank has " + std::to_string(m_metadata.bank.size()) + " entries but the axis grid needs "
                    + std::to_string(expectedBank) + "."};
        }

        static void validateAnchors(std::vector<double> const& anchors, std::string const& context)
        {
            if(anchors.size() < 2u)
                throw ModelError{context + " must contain at least two anchors."};
            for(auto const value : anchors)
            {
                if(!std::isfinite(value))
                    throw ModelError{context + " contains a non-finite anchor."};
                if(value < 0.1 - 1e-12 || value > 10.0 + 1e-12)
                    throw ModelError{context + " anchor is outside the trained range [0.1, 10]."};
            }
            for(std::size_t i = 1; i < anchors.size(); ++i)
            {
                if(!(std::log10(anchors[i]) > std::log10(anchors[i - 1])))
                    throw ModelError{context + " anchors must increase strictly in log10."};
            }
        }

        void parseDomain(json const& domain)
        {
            using detail::asDoubleVector;
            using detail::asFiniteDouble;
            using detail::required;

            auto const gamma = asDoubleVector(required(domain, "gamma", "domain"), "domain.gamma");
            if(gamma.size() != 2u)
                throw ModelError{"domain.gamma must contain [min, max]."};
            m_metadata.gammaMin = gamma[0];
            m_metadata.gammaMax = gamma[1];
            if(!(m_metadata.gammaMin >= 1.0 && m_metadata.gammaMin < m_metadata.gammaMax
                 && m_metadata.gammaMax <= 100.0))
                throw ModelError{"domain.gamma must satisfy 1 <= min < max <= 100."};

            auto parseAspect = [&](char const* key, double& low, double& high)
            {
                auto const values = asDoubleVector(required(domain, key, "domain"), std::string{"domain."} + key);
                if(values.size() != 2u)
                    throw ModelError{std::string{"domain."} + key + " must contain [min, max]."};
                low = values[0];
                high = values[1];
                if(!(low >= 0.1 && low < high && high <= 10.0))
                    throw ModelError{std::string{"domain."} + key + " must satisfy 0.1 <= min < max <= 10."};
            };
            parseAspect("a_y", m_metadata.aYMin, m_metadata.aYMax);
            parseAspect("a_z", m_metadata.aZMin, m_metadata.aZMax);

            auto const minAnchorY = m_metadata.aYAnchors.front();
            auto const maxAnchorY = m_metadata.aYAnchors.back();
            auto const minAnchorZ = m_metadata.aZAnchors.front();
            auto const maxAnchorZ = m_metadata.aZAnchors.back();
            if(std::abs(minAnchorY - m_metadata.aYMin) > 1e-12 || std::abs(maxAnchorY - m_metadata.aYMax) > 1e-12
               || std::abs(minAnchorZ - m_metadata.aZMin) > 1e-12
               || std::abs(maxAnchorZ - m_metadata.aZMax) > 1e-12)
                throw ModelError{"decoder axis grids must span the domain a_y/a_z bounds."};

            m_metadata.dMinCells = asFiniteDouble(required(domain, "d_min_cells", "domain"), "domain.d_min_cells");
            m_metadata.dMaxCells = asFiniteDouble(required(domain, "d_max_cells", "domain"), "domain.d_max_cells");
            if(!(m_metadata.dMinCells > 0.0 && m_metadata.dMinCells < m_metadata.dMaxCells))
                throw ModelError{"domain.d_min_cells/d_max_cells must be finite and satisfy 0 < min < max."};
            m_metadata.distanceMetric
                = detail::asString(required(domain, "distance_metric", "domain"), "domain.distance_metric");
        }

        void parseNormalization(json const& normalization)
        {
            using detail::asString;
            using detail::required;

            m_metadata.normalizationLength
                = asString(required(normalization, "length", "normalization"), "normalization.length");
            m_metadata.normalizationCharge
                = asString(required(normalization, "charge", "normalization"), "normalization.charge");
            m_metadata.normalizationFieldE
                = asString(required(normalization, "field_E", "normalization"), "normalization.field_E");
            m_metadata.normalizationFieldB
                = asString(required(normalization, "field_B", "normalization"), "normalization.field_B");
            m_metadata.normalizationLatent
                = asString(required(normalization, "latent", "normalization"), "normalization.latent");
            if(m_metadata.normalizationLength != "dx" || m_metadata.normalizationCharge != "q_i/e = -w_i")
                throw ModelError{"unsupported normalization length/charge convention."};
            if(m_metadata.normalizationLatent != "none")
                throw ModelError{"format v1 requires normalization.latent = none."};
        }

        void parseTensors(json const& tensors)
        {
            using detail::asShape;
            using detail::asSize;
            using detail::asString;
            using detail::required;

            if(!tensors.is_array() || tensors.empty())
                throw ModelError{"tensors must be a non-empty array."};

            std::set<std::string> names;
            std::size_t previousEnd = 0u;
            for(std::size_t i = 0; i < tensors.size(); ++i)
            {
                auto const context = "tensors[" + std::to_string(i) + "]";
                auto const& entry = tensors.at(i);
                TensorBinding binding;
                binding.name = asString(required(entry, "name", context), context + ".name");
                binding.offsetBytes = asSize(required(entry, "offset_bytes", context), context + ".offset_bytes");
                binding.count = asSize(required(entry, "count", context), context + ".count");
                binding.shape = asShape(required(entry, "shape", context), context + ".shape");
                auto const order = asString(required(entry, "order", context), context + ".order");
                if(order != "row_major_in_out")
                    throw ModelError{context + ".order must be 'row_major_in_out'."};

                if(binding.name.empty())
                    throw ModelError{context + ": empty tensor name."};
                if(!names.insert(binding.name).second)
                    throw ModelError{"duplicate tensor name '" + binding.name + "'."};
                if(binding.count == 0u)
                    throw ModelError{context + ": count must be positive."};
                if(detail::product(binding.shape, context) != binding.count)
                    throw ModelError{context + ": count does not match the shape product."};
                if(binding.offsetBytes % 64u != 0u)
                    throw ModelError{
                        context + ": offset " + std::to_string(binding.offsetBytes) + " is not 64-byte aligned."};
                if(binding.offsetBytes < previousEnd)
                    throw ModelError{context + ": offset overlaps or is not monotonically increasing."};
                if(binding.count > std::numeric_limits<std::size_t>::max() / 4u)
                    throw ModelError{context + ": byte size overflows std::size_t."};
                auto const end = binding.offsetBytes + binding.count * 4u;
                if(end < binding.offsetBytes)
                    throw ModelError{context + ": byte range overflows std::size_t."};
                previousEnd = end;
                m_metadata.tensors.push_back(std::move(binding));
            }

            // Every referenced tensor must exist with exactly the documented shape, and nothing may be unreferenced.
            std::set<std::string> referenced;
            for(auto const& layer : m_metadata.layers)
            {
                checkTensor(layer.weight, {layer.in, layer.out}, referenced, "encoder weight");
                checkTensor(layer.bias, {layer.out}, referenced, "encoder bias");
            }
            for(auto const& name : m_metadata.bank)
                checkTensor(name, {m_metadata.latent, kOutDim}, referenced, "decoder bank entry");

            if(referenced.size() != m_metadata.tensors.size())
            {
                for(auto const& binding : m_metadata.tensors)
                    if(!referenced.contains(binding.name))
                        throw ModelError{"tensor '" + binding.name + "' is not referenced by the metadata."};
            }
        }

        void checkTensor(
            std::string const& name,
            std::vector<std::size_t> const& expectedShape,
            std::set<std::string>& referenced,
            std::string const& role) const
        {
            auto const index = m_metadata.tensorIndex(name);
            auto const& binding = m_metadata.tensors[index];
            if(binding.shape != expectedShape)
            {
                std::string actual;
                for(auto const extent : binding.shape)
                    actual += (actual.empty() ? "" : ",") + std::to_string(extent);
                std::string expected;
                for(auto const extent : expectedShape)
                    expected += (expected.empty() ? "" : ",") + std::to_string(extent);
                throw ModelError{
                    role + " tensor '" + name + "' has shape [" + actual + "] but the metadata requires [" + expected
                    + "]."};
            }
            referenced.insert(name);
        }

        void parseArtifact(json const& artifact)
        {
            using detail::asSize;
            using detail::asString;
            using detail::required;

            m_metadata.artifactFile = asString(required(artifact, "file", "artifact"), "artifact.file");
            if(m_metadata.artifactFile.empty())
                throw ModelError{"artifact.file is empty."};
            m_metadata.artifactBytes = asSize(required(artifact, "bytes", "artifact"), "artifact.bytes");
            m_metadata.artifactSha256 = asString(required(artifact, "sha256", "artifact"), "artifact.sha256");
            if(m_metadata.artifactSha256.size() != 64u)
                throw ModelError{"artifact.sha256 must contain 64 hexadecimal characters."};

            m_binPath = m_jsonPath.parent_path() / m_metadata.artifactFile;
        }

        void verifyPayload()
        {
            std::error_code error;
            auto const actualBytes = std::filesystem::file_size(m_binPath, error);
            if(error)
                throw ModelError{"cannot stat payload '" + m_binPath.string() + "': " + error.message()};
            if(actualBytes != m_metadata.artifactBytes)
                throw ModelError{
                    "payload byte count mismatch: metadata says " + std::to_string(m_metadata.artifactBytes)
                    + ", file has " + std::to_string(actualBytes) + "."};

            for(auto const& binding : m_metadata.tensors)
            {
                auto const end = binding.offsetBytes + binding.count * 4u;
                if(end > actualBytes)
                    throw ModelError{"tensor '" + binding.name + "' exceeds the payload length."};
            }

            std::string actualSha;
            try
            {
                actualSha = alpaka::nn::onHost::sha256File(m_binPath.string());
            }
            catch(std::exception const& exception)
            {
                throw ModelError{"cannot hash payload: " + std::string{exception.what()}};
            }
            if(actualSha != m_metadata.artifactSha256)
                throw ModelError{
                    "payload SHA-256 mismatch: metadata says " + m_metadata.artifactSha256 + ", file has " + actualSha
                    + "."};
        }
    };
} // namespace picongpu::particles::neuralBunch
