/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Fast host test for the three feature kinds and their metadata validation.
 * Writes minimal synthetic artifacts (tiny encoder/decoder, zero weights) and
 * compares featuresHost against an independent point-field implementation.
 */

#include "picongpu/particles/neuralBunch/Features.hpp"
#include "picongpu/particles/neuralBunch/Model.hpp"

#include <alpaka/nn/onHost/core/binary.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace nb = picongpu::particles::neuralBunch;
using nlohmann::json;

namespace
{
    int failures = 0;

    void check(bool const ok, std::string const& what)
    {
        if(!ok)
        {
            std::cerr << "FAIL: " << what << "\n";
            ++failures;
        }
    }

    std::string kindName(nb::FeatureKind kind)
    {
        return std::string{nb::featureKindName(kind)};
    }

    std::size_t align64(std::size_t value)
    {
        return (value + 63u) / 64u * 64u;
    }

    json makeArtifact(nb::FeatureKind kind, fs::path const& directory)
    {
        auto const dim = nb::featureDimension(kind);
        json metadata;
        metadata["format"] = "picongpu-neural-bunch";
        metadata["version"] = 1;
        metadata["dtype"] = "float32";
        metadata["endianness"] = "little";
        metadata["created_utc"] = "2026-01-01T00:00:00Z";
        metadata["code_sha"] = std::string(40, 'a');
        metadata["tile"] = {
            {"nx", 8}, {"ny", 8}, {"nz", 4},
            {"components", {"Ex", "Ey", "Ez", "Bx", "By", "Bz"}},
            {"flatten", "c + 6*(i + nx*(j + ny*k))"}};
        json features = {
            {"kind", kindName(kind)},
            {"inputs", {"r_x", "r_y", "r_z", "u_x", "u_y", "u_z", "gamma", "a_y", "a_z"}},
            {"dim", dim}};
        if(kind == nb::FeatureKind::fourierV1)
            features["octaves"] = {-3, 6};
        else if(kind == nb::FeatureKind::fourierProbeV1)
            features["octaves"] = {-3, 1};
        else
            features["octaves"] = json::array();
        if(kind != nb::FeatureKind::fourierV1)
        {
            json offsets = json::array();
            auto const probeKind = kind;
            for(std::size_t p = 0; p < nb::featureProbeCount(probeKind); ++p)
            {
                std::array<double, 3> offset{};
                if(probeKind == nb::FeatureKind::fourierProbeV1)
                {
                    auto const i = p % 4u;
                    auto const j = (p / 4u) % 4u;
                    auto const k = p / 16u;
                    offset = {2.0 * i + 0.5, 2.0 * j + 0.5, 2.0 * k + 0.5};
                }
                else
                {
                    auto const i = p % 8u;
                    auto const j = (p / 8u) % 8u;
                    auto const k = p / 64u;
                    offset = {1.0 * i + 0.5, 1.0 * j + 0.5, 1.0 * k + 0.5};
                }
                offsets.push_back({offset[0], offset[1], offset[2]});
            }
            features["probe"] = {
                {"components", {"Ex", "Ey", "Ez", "Bx", "By", "Bz"}},
                {"cell_offsets", offsets},
                {"grid",
                 kind == nb::FeatureKind::fourierProbeV1
                     ? "4x4x2 tile cell centres: x=2i+0.5 (i=0..3), y=(2j+0.5)*a_y (j=0..3), z=(2k+0."
                       "5)*a_z (k=0..1)"
                     : "one probe at every tile cell centre: x=i+0.5 (i=0..7), y=(j+0.5)*a_y (j=0..7), z=("
                       "k+0.5)*a_z (k=0..3), i fastest then j then k"},
                {"field",
                 "per unit q/e: E_n=4*pi*eps0*dx^2*E/q, B_n=4*pi*eps0*c*dx^2*B/q; stable uniform-motion "
                 "decomposition (1/gamma^2 direct, static branch)"}};
            features["ordering"]
                = kind == nb::FeatureKind::fourierProbeV1
                      ? "9 base, 32 probes x [Ex,Ey,Ez,Bx,By,Bz], 3 axes x octaves -3..1 x (sin,cos)"
                      : "9 base, 256 probes x [Ex,Ey,Ez,Bx,By,Bz] in i-fastest cell order; no Fourier terms";
        }
        metadata["features"] = features;
        metadata["encoder"] = {
            {"kind", "mlp"}, {"activation", "silu"}, {"last_layer_activation", "none"},
            {"layers", {{{"weight", "enc.0.w"}, {"bias", "enc.0.b"}, {"in", dim}, {"out", 1}}}},
            {"output_dim", 1}};
        metadata["decoder"] = {
            {"kind", "geometry_bank_bilinear"},
            {"interpolation", "bilinear in log10(a), clamped at the outer anchors"},
            {"axes", {{"a_y", {0.1, 10.0}}, {"a_z", {0.1, 10.0}}}},
            {"matrix_shape", {1, 1536}}, {"bias", false},
            {"bank", {"dec.0.0.w", "dec.0.1.w", "dec.0.2.w", "dec.0.3.w"}},
            {"bank_entry_shape", {1, 1536}}};
        metadata["domain"] = {
            {"gamma", {1.0, 100.0}}, {"a_y", {0.1, 10.0}}, {"a_z", {0.1, 10.0}},
            {"d_min_cells", 4.0}, {"d_max_cells", 512.0},
            {"distance_metric", "minimum over the 1536 staggered output points of |x_out - x_source| / dx"}};
        metadata["normalization"] = {
            {"length", "dx"}, {"charge", "q_i/e = -w_i"},
            {"field_E", "E = e/(4*pi*eps0*L^2) * E_tilde"},
            {"field_B", "B = e/(4*pi*eps0*c*L^2) * B_tilde"},
            {"latent", "none"}};

        std::size_t offset = 0;
        json tensors = json::array();
        auto addTensor = [&](std::string name, std::vector<std::size_t> shape)
        {
            std::size_t count = 1;
            for(auto extent : shape)
                count *= extent;
            offset = align64(offset);
            tensors.push_back({
                {"name", name}, {"offset_bytes", offset}, {"count", count},
                {"shape", shape}, {"order", "row_major_in_out"}});
            offset += count * 4u;
        };
        addTensor("enc.0.w", {dim, 1});
        addTensor("enc.0.b", {1});
        for(int i = 0; i < 4; ++i)
            addTensor("dec.0." + std::to_string(i) + ".w", {1, 1536});
        metadata["tensors"] = tensors;
        metadata["artifact"] = {
            {"file", "model.bin"}, {"bytes", offset}, {"sha256", std::string(64, '0')}};
        metadata["license"] = "MPL-2.0";

        fs::create_directories(directory);
        std::vector<char> payload(offset, 0);
        {
            std::ofstream out(directory / "model.bin", std::ios::binary);
            out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        }
        auto const sha = alpaka::nn::onHost::sha256File((directory / "model.bin").string());
        metadata["artifact"]["sha256"] = sha;
        std::ofstream jsonOut(directory / "model.json");
        jsonOut << metadata.dump(1);
        return metadata;
    }

    void rewrite(json metadata, fs::path const& directory)
    {
        auto const sha = alpaka::nn::onHost::sha256File((directory / "model.bin").string());
        metadata["artifact"]["sha256"] = sha;
        std::ofstream out(directory / "model.json");
        out << metadata.dump(1);
    }

    bool rejects(json const& metadata, fs::path const& directory, std::string const& what)
    {
        rewrite(metadata, directory);
        try
        {
            (void) nb::Model::load(directory / "model.json");
            check(false, what + ": expected rejection");
            return false;
        }
        catch(nb::ModelError const&)
        {
            return true;
        }
    }

    struct V
    {
        double x, y, z;
    };

    void pointField(V const& r, V const& u, V& e, V& b)
    {
        double const uu2 = u.x * u.x + u.y * u.y + u.z * u.z;
        double const invG2 = 1.0 / (1.0 + uu2);
        if(uu2 == 0.0)
        {
            double const r2 = r.x * r.x + r.y * r.y + r.z * r.z;
            double const d = r2 * std::sqrt(r2);
            e = {r.x / d, r.y / d, r.z / d};
            b = {0.0, 0.0, 0.0};
            return;
        }
        double const speed = std::sqrt(uu2);
        V const bhat{u.x / speed, u.y / speed, u.z / speed};
        double const rpar = r.x * bhat.x + r.y * bhat.y + r.z * bhat.z;
        V const rp{r.x - rpar * bhat.x, r.y - rpar * bhat.y, r.z - rpar * bhat.z};
        double const d = rpar * rpar + (rp.x * rp.x + rp.y * rp.y + rp.z * rp.z) * invG2;
        double const den = d * std::sqrt(d);
        e = {invG2 * r.x / den, invG2 * r.y / den, invG2 * r.z / den};
        double const gamma = std::sqrt(1.0 + uu2);
        V const beta{u.x / gamma, u.y / gamma, u.z / gamma};
        b = {beta.y * e.z - beta.z * e.y, beta.z * e.x - beta.x * e.z, beta.x * e.y - beta.y * e.x};
    }

    std::vector<float> expectedFeatures(
        nb::FeatureKind kind,
        double rX, double rY, double rZ,
        double uX, double uY, double uZ,
        double aY, double aZ)
    {
        std::vector<float> out(nb::featureDimension(kind));
        double const gamma = std::sqrt(1.0 + uX * uX + uY * uY + uZ * uZ);
        double const base[9]{rX, rY, rZ, uX, uY, uZ, gamma, aY, aZ};
        for(int i = 0; i < 9; ++i)
            out[i] = static_cast<float>(base[i]);
        std::size_t w = 9;
        if(kind == nb::FeatureKind::fourierV1)
        {
            double const r[3]{rX, rY, rZ};
            for(int axis = 0; axis < 3; ++axis)
                for(int oct = -3; oct <= 6; ++oct)
                {
                    double const phase = nb::kPi * std::ldexp(1.0, oct) * r[axis];
                    out[w++] = static_cast<float>(std::sin(phase));
                    out[w++] = static_cast<float>(std::cos(phase));
                }
            return out;
        }
        auto const probes = nb::featureProbeCount(kind);
        for(std::size_t p = 0; p < probes; ++p)
        {
            double ox, oy, oz;
            if(kind == nb::FeatureKind::fourierProbeV1)
            {
                auto const i = p % 4u;
                auto const j = (p / 4u) % 4u;
                auto const k = p / 16u;
                ox = 2.0 * i + 0.5;
                oy = 2.0 * j + 0.5;
                oz = 2.0 * k + 0.5;
            }
            else
            {
                auto const i = p % 8u;
                auto const j = (p / 8u) % 8u;
                auto const k = p / 64u;
                ox = i + 0.5;
                oy = j + 0.5;
                oz = k + 0.5;
            }
            V e, b;
            pointField({ox - rX, oy * aY - rY, oz * aZ - rZ}, {uX, uY, uZ}, e, b);
            out[w++] = static_cast<float>(e.x);
            out[w++] = static_cast<float>(e.y);
            out[w++] = static_cast<float>(e.z);
            out[w++] = static_cast<float>(b.x);
            out[w++] = static_cast<float>(b.y);
            out[w++] = static_cast<float>(b.z);
        }
        if(kind == nb::FeatureKind::fourierProbeV1)
        {
            double const r[3]{rX, rY, rZ};
            for(int axis = 0; axis < 3; ++axis)
                for(int oct = -3; oct <= 1; ++oct)
                {
                    double const phase = nb::kPi * std::ldexp(1.0, oct) * r[axis];
                    out[w++] = static_cast<float>(std::sin(phase));
                    out[w++] = static_cast<float>(std::cos(phase));
                }
        }
        return out;
    }

    void featureParity(nb::FeatureKind kind)
    {
        double const values[8]{0.25, -0.4, 1.75, 0.3, -0.2, 0.1, 2.0, 0.5};
        auto const expected = expectedFeatures(
            kind, values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
        std::vector<float> actual(nb::featureDimension(kind));
        nb::featuresHost(
            kind, actual.data(), values[0], values[1], values[2], values[3], values[4], values[5], values[6],
            values[7]);
        check(actual.size() == expected.size(), kindName(kind) + ": dimension");
        double maxDiff = 0.0;
        for(std::size_t i = 0; i < actual.size(); ++i)
            maxDiff = std::max(maxDiff, std::abs(double(actual[i]) - double(expected[i])));
        check(maxDiff <= 1e-5, kindName(kind) + ": feature parity");
    }
} // namespace

int main()
{
    auto const root = fs::temp_directory_path() / "pic_neural_probe_test";
    fs::remove_all(root);
    for(auto kind : {nb::FeatureKind::fourierV1, nb::FeatureKind::fourierProbeV1, nb::FeatureKind::probeDenseV1})
    {
        auto const directory = root / kindName(kind);
        auto metadata = makeArtifact(kind, directory);
        auto const model = nb::Model::load(directory / "model.json");
        auto const& meta = model.metadata();
        check(meta.featureKind == kind, kindName(kind) + ": kind accepted");
        check(meta.featureDim == nb::featureDimension(kind), kindName(kind) + ": dim accepted");
        if(kind != nb::FeatureKind::fourierV1)
        {
            check(meta.probeCellOffsets.size() == nb::featureProbeCount(kind), kindName(kind) + ": probes");
            check(!meta.featureOrdering.empty(), kindName(kind) + ": ordering");
        }
        featureParity(kind);

        auto wrongDim = metadata;
        wrongDim["features"]["dim"] = nb::featureDimension(kind) + 1;
        rejects(wrongDim, directory, kindName(kind) + " wrong dim");
        auto badKind = metadata;
        badKind["features"]["kind"] = "unknown_kind";
        rejects(badKind, directory, kindName(kind) + " unknown kind");
        if(kind != nb::FeatureKind::fourierV1)
        {
            auto badOffset = metadata;
            badOffset["features"]["probe"]["cell_offsets"][0][0] = 0.75;
            rejects(badOffset, directory, kindName(kind) + " bad probe offset");
            auto badOrder = metadata;
            badOrder["features"]["ordering"] = "reversed";
            rejects(badOrder, directory, kindName(kind) + " bad ordering");
            auto badOctaves = metadata;
            badOctaves["features"]["octaves"] = (kind == nb::FeatureKind::probeDenseV1)
                                                    ? json::array({-3, 1})
                                                    : json::array({-3, 6});
            rejects(badOctaves, directory, kindName(kind) + " bad octaves");
        }
    }
    fs::remove_all(root);
    if(failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "probeFeatureTest: OK\n";
    return 0;
}
