"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..pypicongpu.neural_bunch import FieldInitializerName
from ..pypicongpu.neural_bunch import NeuralBunchInit as PyPIConGPUNeuralBunchInit
from .species import Species

# Field solvers whose E/B field layout is the staggered Yee layout the neural
# bunch stage stores into. ``other:None`` disables the vacuum field update and
# therefore cannot carry the bunch fields at all.
SUPPORTED_FIELD_SOLVERS = ("Yee", "Lehe", "CKC", "other:ArbitraryOrderFDTD")

# The reference kernel and the model both assume the PIConGPU TSC shape (support
# three cells per axis). PICMI names TSC ``quadratic``.
SUPPORTED_PARTICLE_SHAPE = "quadratic"


class NeuralBunchInitializer(BaseModel):
    """PICMI control of the external electron-bunch initializer.

    The initializer is attached to a :class:`~picongpu.picmi.Simulation` via
    ``picongpu_neural_bunch``. It reads an openPMD electron bunch written by an
    external generator, adds the bunch fields at the start of the simulation and
    inserts the particles into the destination species.

    Cell geometry (``dx``/``dy``/``dz``), the field layout and the destination
    species alias are derived from the PICMI grid and solver. They are not user
    fields.

    ``field_initializer`` selects the field policy:

    * unset (default): resolves to ``"banded"``, the analytic point field for
      mid/far and the exact reference for near.
    * ``"neural"``: the trained model handles pairs inside the artifact
      domain; kept for compatibility.
    * ``"validatedNeural"``: the model is used only inside a measured
      ``validation.validated_band``.
    * ``"referenceOnly"``: the exact FP64 reference is used and no model
      artifact is read.
    * ``"banded"``: distance-banded near/mid/far composition. The near band
      uses the exact reference by default (or the model when explicitly opted
      in), the mid band defaults to the analytic point field only, and the far
      band uses the analytic point field plus an optional ``far_multipole``
      Taylor head. The learned ``mid_correction`` is opt-in and measured worse
      than the point field alone.
    """

    filename: str = Field(min_length=1)
    """openPMD series path, e.g. ``bunch.h5`` or ``bunch.bp5``"""

    iteration: int = Field(0, ge=0)
    """iteration inside the series read for both passes"""

    source_species: str = Field("e", min_length=1)
    """particle species name inside the openPMD series"""

    model_path: str | None = None
    """artifact path prefix (``<model_path>.json`` / ``.bin``); not needed for referenceOnly"""

    chunk_size: int = Field(1 << 20, ge=1)
    """global source particles streamed per collective chunk"""

    tile_batch: int = Field(1, ge=1)
    """tiles per inference batch"""

    field_initializer: FieldInitializerName = "banded"
    """field-initializer policy, see the class docstring"""

    enable_reference_fallback: bool = True
    """use the exact reference outside the model domain instead of failing"""

    time_policy: Literal[0] = 0
    """must stay 0: source time equals injection time"""

    species: Species
    """destination electron species that receives the particles"""

    @model_validator(mode="after")
    def _validate(self):
        if self.field_initializer in ("neural", "validatedNeural") and not self.model_path:
            raise ValueError(
                f"model_path is required for field_initializer={self.field_initializer!r}; "
                "use field_initializer='referenceOnly' or 'banded' to run without an artifact."
            )
        if not self._is_electron(self.species):
            raise ValueError(
                "The neural bunch initializer only supports an electron destination species, "
                f"got {self.species.name!r} with particle_type={self.species.particle_type!r}."
            )
        return self

    @staticmethod
    def _is_electron(species: Species) -> bool:
        if species.particle_type == "electron":
            return True
        # A species with explicit mass/charge but no particle_type can still be
        # an electron if the values match. Keep the tolerance tight; this is a
        # convenience, not a unit conversion.
        mass = getattr(species, "mass", None)
        charge = getattr(species, "charge", None)
        if mass is None or charge is None:
            return False
        return abs(mass - 9.1093837015e-31) <= 1e-40 and abs(charge + 1.602176634e-19) <= 1e-30

    def get_as_pypicongpu(
        self,
        grid,
        solver_method: str,
        default_particle_shape: str | None = None,
    ) -> PyPIConGPUNeuralBunchInit:
        """Translate to the PyPIConGPU rendering model.

        Derives the cell size, field layout and destination species alias from
        the PICMI grid/solver. Relative paths are resolved later against the
        generated setup directory.
        """
        if getattr(grid, "number_of_dimensions", None) != 3:
            raise ValueError(
                "The neural bunch initializer requires a 3D simulation; "
                f"got a {getattr(grid, 'number_of_dimensions', 'unknown')}D grid."
            )
        if solver_method not in SUPPORTED_FIELD_SOLVERS:
            raise ValueError(
                f"The neural bunch initializer does not support field solver {solver_method!r}. "
                f"Supported solvers with the Yee field layout: {', '.join(SUPPORTED_FIELD_SOLVERS)}."
            )
        shape = self.species._resolved_particle_shape(default_particle_shape)
        if shape != SUPPORTED_PARTICLE_SHAPE:
            raise ValueError(
                "The neural bunch initializer requires the TSC particle shape "
                f"(PICMI name {SUPPORTED_PARTICLE_SHAPE!r}), got {shape!r} for species "
                f"{self.species.name!r}."
            )

        cell_size = getattr(grid, "picongpu_cell_size", None)
        if cell_size is None:
            raise ValueError(
                "Could not derive the cell size from the PICMI grid; expected a Cartesian3DGrid "
                f"with picongpu_cell_size, got {type(grid).__name__}."
            )

        return PyPIConGPUNeuralBunchInit(
            enabled=True,
            filename=self.filename,
            iteration=self.iteration,
            source_species=self.source_species,
            modelPath=self.model_path or "",
            chunkSize=self.chunk_size,
            tileBatch=self.tile_batch,
            fieldInitializer=self.field_initializer,
            enableReferenceFallback=self.enable_reference_fallback,
            timePolicy=self.time_policy,
            destination_species="species_" + self.species.name,
            destination_species_name=self.species.name,
            cell_size_si=tuple(float(component) for component in cell_size),
            provenance={
                "filename": self.filename,
                "model_path": self.model_path or "",
                "destination_species": self.species.name,
                "field_initializer": self.field_initializer or "auto",
            },
        )


AnyNeuralBunchInitializer = NeuralBunchInitializer
"""union of PICMI external-bunch initializers"""


__all__ = ["AnyNeuralBunchInitializer", "NeuralBunchInitializer"]
