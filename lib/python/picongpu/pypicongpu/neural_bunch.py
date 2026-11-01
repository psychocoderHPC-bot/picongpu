"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PlainSerializer, field_validator, model_validator
from typing import Annotated

from .grid import serialise_vec3

FieldInitializerName = Literal["neural", "validatedNeural", "referenceOnly"]
"""Field-initializer policy names, identical to the C++ enum values."""

Vec3_float = Annotated[tuple[float, float, float], PlainSerializer(serialise_vec3)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NeuralBunchInit(BaseModel):
    """External bunch initializer parameters rendered into ``neuralBunch.param``.

    The C++ stage reads the openPMD series, evaluates the field model (or the
    exact reference kernel) and inserts the particles into the destination
    species. This model mirrors the C++ ``picongpu::neuralBunch::NeuralBunchParam``
    field names so the rendering context and the generated param file use the
    same vocabulary.

    ``fieldInitializer`` is the policy discriminator: ``neural`` (default),
    ``validatedNeural`` (only inside a measured validation band) or
    ``referenceOnly`` (exact FP64 reference, no model artifact required).
    """

    type_neuralbunch: Literal[True] = True
    """discriminator of the external-bunch initializer union"""

    filename: str = ""
    """openPMD series path (resolved at render time)"""

    iteration: int = 0
    """iteration inside the series read for both passes"""

    source_species: str = "e"
    """particle species name inside the openPMD series (C++ ``species``)"""

    modelPath: str = ""
    """artifact path prefix; ``<modelPath>.json`` and ``<modelPath>.bin``"""

    chunkSize: int = 1 << 20
    """global source particles streamed per collective chunk"""

    tileBatch: int = 1
    """tiles per inference batch"""

    fieldInitializer: FieldInitializerName = "neural"
    """field-initializer policy"""

    enableReferenceFallback: bool = True
    """use the exact reference outside the model domain instead of failing"""

    timePolicy: int = 0
    """must stay 0: source time equals injection time"""

    destination_species: str = ""
    """C++ typename of the destination species, e.g. ``species_electrons``"""

    destination_species_name: str = ""
    """PICMI species name used to build the ``PIC_<name>`` alias"""

    cell_size_si: Vec3_float = Field((1.0, 1.0, 1.0), json_schema_extra={"derived_from": "PICMI grid"})
    """cell size (dx, dy, dz) in metres, derived from the PICMI grid"""

    provenance: dict[str, str] = Field(default_factory=dict)
    """original and resolved source/model paths plus artifact digests"""

    @field_validator("cell_size_si")
    @classmethod
    def _positive_cell_size(cls, value):
        if any(not (component > 0.0) for component in value):
            raise ValueError(f"cell sizes must be positive, got {value}.")
        return value

    @model_validator(mode="after")
    def _check(self):
        if self.chunkSize < 1:
            raise ValueError(f"chunkSize must be >= 1, got {self.chunkSize}.")
        if self.tileBatch < 1:
            raise ValueError(f"tileBatch must be >= 1, got {self.tileBatch}.")
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}.")
        if self.timePolicy != 0:
            raise ValueError(
                "timePolicy must be 0: the neural bunch source time equals the injection time "
                f"(no linear drift is applied), got {self.timePolicy}."
            )
        if not self.source_species:
            raise ValueError("source_species must not be empty.")
        if not self.filename:
            raise ValueError("filename must not be empty.")
        if self.fieldInitializer != "referenceOnly" and not self.modelPath:
            raise ValueError(
                f"modelPath is required for fieldInitializer={self.fieldInitializer!r}; "
                "use field_initializer='referenceOnly' to run without an artifact."
            )
        return self

    @property
    def needs_model(self) -> bool:
        return self.fieldInitializer != "referenceOnly"

    @property
    def model_json(self) -> str:
        return self.modelPath + ".json"

    @property
    def model_binary(self) -> str:
        return self.modelPath + ".bin"

    @staticmethod
    def _resolve(raw: str, setup_dir: Path) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(setup_dir) / path
        return path.resolve()

    def resolve_paths(self, setup_dir: Path | str) -> None:
        """Resolve relative paths against the generated simulation directory.

        Absolute paths are kept. ``setup_dir`` is the directory the generated
        simulation lives in, so a path written relative to the example script
        cannot silently depend on the process working directory when the
        executable later runs in its own run directory.

        Existence of the series and (for neural policies) of the two artifact
        files is checked here, i.e. at render time. Errors name the resolved
        path and the accepted alternatives.
        """
        if not self.filename:
            raise ValueError("neuralBunch: filename must not be empty.")
        setup_dir = Path(setup_dir)
        original_filename = self.filename
        original_model_path = self.modelPath

        resolved_filename = self._resolve(original_filename, setup_dir)
        if not resolved_filename.exists():
            raise FileNotFoundError(
                "neuralBunch: openPMD series not found: "
                f"{resolved_filename} (given as {original_filename!r}, setup dir {setup_dir})."
            )
        # HDF5 series are single files; ADIOS2/BP5 series are directories.
        # Both are valid openPMD series paths.
        self.filename = str(resolved_filename)

        resolved_model_path = ""
        if self.modelPath:
            resolved_model = self._resolve(original_model_path, setup_dir)
            resolved_model_path = str(resolved_model)
            if self.needs_model:
                for path in (Path(resolved_model_path + ".json"), Path(resolved_model_path + ".bin")):
                    if not path.exists():
                        raise FileNotFoundError(
                            "neuralBunch: model artifact file not found: "
                            f"{path} (modelPath given as {original_model_path!r}, setup dir {setup_dir})."
                        )
            self.modelPath = resolved_model_path

        self.provenance = {
            "filename": original_filename,
            "resolved_filename": str(resolved_filename),
            "model_path": original_model_path,
            "resolved_model_path": resolved_model_path,
            "source_species": self.source_species,
            "destination_species": self.destination_species,
            "field_initializer": self.fieldInitializer,
        }
        if self.needs_model:
            self.provenance["model_json_sha256"] = _sha256(Path(self.model_json))
            self.provenance["model_binary_sha256"] = _sha256(Path(self.model_binary))


AnyNeuralBunchInitializer = NeuralBunchInit
"""union of external-bunch initializers understood by PyPIConGPU"""


__all__ = [
    "AnyNeuralBunchInitializer",
    "FieldInitializerName",
    "NeuralBunchInit",
]
