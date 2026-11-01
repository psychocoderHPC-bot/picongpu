"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from picongpu import picmi, templates
from picongpu.pypicongpu.neural_bunch import NeuralBunchInit
from picongpu.pypicongpu.rendering.renderer import Renderer

TEMPLATE = Path(templates.path()) / "include" / "picongpu" / "param" / "neuralBunch.param.mustache"

NUMBER_OF_CELLS = [8, 8, 4]
CELL_SIZE = 2.0e-7


def make_grid():
    return picmi.Cartesian3DGrid(
        number_of_cells=NUMBER_OF_CELLS,
        lower_bound=[0.0, 0.0, 0.0],
        upper_bound=[n * CELL_SIZE for n in NUMBER_OF_CELLS],
        lower_boundary_conditions=["open", "open", "open"],
        upper_boundary_conditions=["open", "open", "open"],
    )


def make_initializer(tmp_path, **overrides):
    series = tmp_path / "bunch.h5"
    series.write_bytes(b"")
    model = tmp_path / "model"
    (tmp_path / "model.json").write_text("{}")
    (tmp_path / "model.bin").write_bytes(b"")
    electrons = picmi.Species(particle_type="electron", name="electrons")
    kwargs = {
        "filename": str(series),
        "iteration": 3,
        "source_species": "e",
        "model_path": str(model),
        "chunk_size": 4096,
        "tile_batch": 2,
        "field_initializer": "referenceOnly",
        "enable_reference_fallback": True,
        "time_policy": 0,
        "species": electrons,
    }
    kwargs.update(overrides)
    return picmi.NeuralBunchInitializer(**kwargs), electrons


def make_simulation(tmp_path, **overrides):
    initializer, electrons = make_initializer(tmp_path, **overrides)
    sim = picmi.Simulation(
        max_steps=2,
        solver=picmi.ElectromagneticSolver(grid=make_grid(), method="Yee", cfl=1.0),
        species=[electrons],
        layouts=[None],
        picongpu_neural_bunch=initializer,
    )
    return sim, initializer


def render_neural_bunch(sim) -> str:
    pypic = sim.get_as_pypicongpu()
    context = pypic.get_rendering_context()
    Renderer.check_rendering_context(context)
    return Renderer.get_rendered_template(Renderer.get_context_preprocessed(context), TEMPLATE.read_text())


class TestNeuralBunchInitializer:
    def test_translation_derives_grid_geometry_and_species_alias(self, tmp_path):
        sim, _ = make_simulation(tmp_path)
        pypic = sim.get_as_pypicongpu()
        initializer = pypic.neural_bunch
        assert isinstance(initializer, NeuralBunchInit)
        assert initializer.type_neuralbunch is True
        assert initializer.filename == str(tmp_path / "bunch.h5")
        assert initializer.iteration == 3
        assert initializer.source_species == "e"
        assert initializer.chunkSize == 4096
        assert initializer.tileBatch == 2
        assert initializer.fieldInitializer == "referenceOnly"
        assert initializer.destination_species == "species_electrons"
        assert initializer.destination_species_name == "electrons"
        assert initializer.cell_size_si == (CELL_SIZE, CELL_SIZE, CELL_SIZE)

    def test_template_matches_requested_values(self, tmp_path):
        sim, _ = make_simulation(tmp_path)
        rendered = render_neural_bunch(sim)
        assert "using PIC_electrons = species_electrons;" in rendered
        assert "using Species = PIC_electrons;" in rendered
        assert f'std::string filename = "{tmp_path / "bunch.h5"}";' in rendered
        assert "std::uint64_t iteration = 3;" in rendered
        assert 'std::string species = "e";' in rendered
        assert f'std::string modelPath = "{tmp_path / "model"}";' in rendered
        assert "std::uint64_t chunkSize = 4096;" in rendered
        assert "std::uint32_t tileBatch = 2;" in rendered
        assert "bool enableReferenceFallback = true;" in rendered
        assert "FieldInitializer fieldInitializer = FieldInitializer::referenceOnly;" in rendered
        assert "int timePolicy = 0;" in rendered

    def test_bool_false_renders_cpp_false(self, tmp_path):
        sim, _ = make_simulation(tmp_path, enable_reference_fallback=False)
        rendered = render_neural_bunch(sim)
        assert "bool enableReferenceFallback = false;" in rendered

    def test_relative_paths_resolve_against_setup_directory(self, tmp_path):
        setup = tmp_path / "setup"
        (setup / "data").mkdir(parents=True)
        (setup / "data" / "bunch.h5").write_bytes(b"")
        (setup / "artifacts").mkdir()
        (setup / "artifacts" / "model.json").write_text("{}")
        (setup / "artifacts" / "model.bin").write_bytes(b"")
        sim, _ = make_simulation(
            tmp_path,
            filename="data/bunch.h5",
            model_path="artifacts/model",
            field_initializer="neural",
        )
        pypic = sim.get_as_pypicongpu()
        pypic.spread_directory_information(setup)
        initializer = pypic.neural_bunch
        assert initializer.filename == str((setup / "data" / "bunch.h5").resolve())
        assert initializer.modelPath == str((setup / "artifacts" / "model").resolve())
        assert initializer.provenance["filename"] == "data/bunch.h5"
        assert "model_json_sha256" in initializer.provenance
        assert "model_binary_sha256" in initializer.provenance

    def test_adios2_directory_series_is_accepted(self, tmp_path):
        setup = tmp_path / "setup"
        (setup / "bunch.bp5").mkdir(parents=True)
        sim, _ = make_simulation(tmp_path, filename="bunch.bp5")
        pypic = sim.get_as_pypicongpu()
        pypic.spread_directory_information(setup)
        assert pypic.neural_bunch.filename == str((setup / "bunch.bp5").resolve())

    def test_missing_series_raises_at_render_time(self, tmp_path):
        sim, _ = make_simulation(tmp_path)
        (tmp_path / "bunch.h5").unlink()
        pypic = sim.get_as_pypicongpu()
        with pytest.raises(FileNotFoundError, match="openPMD series not found"):
            pypic.spread_directory_information(tmp_path)

    def test_missing_model_artifact_raises_at_render_time(self, tmp_path):
        sim, _ = make_simulation(tmp_path, field_initializer="neural")
        (tmp_path / "model.bin").unlink()
        pypic = sim.get_as_pypicongpu()
        with pytest.raises(FileNotFoundError, match="model artifact file not found"):
            pypic.spread_directory_information(tmp_path)

    def test_unconfigured_simulation_renders_placeholder(self, tmp_path):
        electrons = picmi.Species(particle_type="electron", name="electrons")
        sim = picmi.Simulation(
            max_steps=2,
            solver=picmi.ElectromagneticSolver(grid=make_grid(), method="Yee", cfl=1.0),
            species=[electrons],
            layouts=[None],
        )
        pypic = sim.get_as_pypicongpu()
        context = pypic.get_rendering_context()
        rendered = Renderer.get_rendered_template(Renderer.get_context_preprocessed(context), TEMPLATE.read_text())
        assert "using Species = species_electrons;" in rendered
        assert "FieldInitializer::referenceOnly" in rendered


class TestNeuralBunchValidation:
    def test_non_electron_destination_is_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="electron destination species"):
            make_initializer(
                tmp_path,
                species=picmi.Species(particle_type="proton", name="protons"),
            )

    def test_model_path_required_for_neural_policies(self, tmp_path):
        with pytest.raises(ValidationError, match="model_path is required"):
            make_initializer(tmp_path, model_path=None, field_initializer="neural")

    def test_time_policy_must_be_zero(self, tmp_path):
        with pytest.raises(ValidationError):
            make_initializer(tmp_path, time_policy=1)

    def test_chunk_size_and_tile_batch_must_be_positive(self, tmp_path):
        with pytest.raises(ValidationError):
            make_initializer(tmp_path, chunk_size=0)
        with pytest.raises(ValidationError):
            make_initializer(tmp_path, tile_batch=0)

    def test_non_3d_grid_is_rejected(self, tmp_path):
        initializer, electrons = make_initializer(tmp_path)
        grid_2d = picmi.Cartesian2DGrid(
            number_of_cells=[16, 16],
            lower_bound=[0.0, 0.0],
            upper_bound=[16 * CELL_SIZE, 16 * CELL_SIZE],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )
        sim = picmi.Simulation(
            max_steps=2,
            solver=picmi.ElectromagneticSolver(grid=grid_2d, method="Yee", cfl=1.0),
            species=[electrons],
            layouts=[None],
            picongpu_neural_bunch=initializer,
        )
        with pytest.raises(ValueError, match="requires a 3D simulation"):
            sim.get_as_pypicongpu()

    def test_non_tsc_shape_is_rejected(self, tmp_path):
        electrons = picmi.Species(particle_type="electron", name="electrons", particle_shape="linear")
        initializer, _ = make_initializer(tmp_path, species=electrons)
        sim = picmi.Simulation(
            max_steps=2,
            solver=picmi.ElectromagneticSolver(grid=make_grid(), method="Yee", cfl=1.0),
            species=[electrons],
            layouts=[None],
            picongpu_neural_bunch=initializer,
        )
        with pytest.raises(ValueError, match="requires the TSC particle shape"):
            sim.get_as_pypicongpu()

    def test_solver_without_yee_layout_is_rejected(self, tmp_path):
        initializer, electrons = make_initializer(tmp_path)
        sim = picmi.Simulation(
            max_steps=2,
            solver=picmi.ElectromagneticSolver(grid=make_grid(), method="other:None"),
            species=[electrons],
            layouts=[None],
            picongpu_neural_bunch=initializer,
        )
        with pytest.raises(ValueError, match="does not support field solver"):
            sim.get_as_pypicongpu()

    def test_conflicting_density_initializer_is_rejected(self, tmp_path):
        electrons = picmi.Species(
            particle_type="electron",
            name="electrons",
            initial_distribution=picmi.UniformDistribution(density=1.0e24),
        )
        initializer, _ = make_initializer(tmp_path, species=electrons)
        sim = picmi.Simulation(
            max_steps=2,
            solver=picmi.ElectromagneticSolver(grid=make_grid(), method="Yee", cfl=1.0),
            species=[electrons],
            layouts=[picmi.PseudoRandomLayout(n_macroparticles_per_cell=1)],
            picongpu_neural_bunch=initializer,
        )
        with pytest.raises(ValueError, match="already has an initial_distribution"):
            sim.get_as_pypicongpu()

    def test_destination_species_must_be_in_simulation(self, tmp_path):
        initializer, _ = make_initializer(tmp_path)
        other = picmi.Species(particle_type="electron", name="other_electrons")
        sim = picmi.Simulation(
            max_steps=2,
            solver=picmi.ElectromagneticSolver(grid=make_grid(), method="Yee", cfl=1.0),
            species=[other],
            layouts=[None],
            picongpu_neural_bunch=initializer,
        )
        with pytest.raises(ValueError, match="is not part of the simulation species list"):
            sim.get_as_pypicongpu()
