"""
This file is part of the PIConGPU.
Copyright 2025-2026 PIConGPU contributors
Authors: Edgar Marquardt
License: GPLv3+
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, computed_field

from .rendering import RenderedObject


class PoissonSolver(RenderedObject, BaseModel):
    """
    Bi-CGSTAB Poisson solver for the electric field in the starting condition.

    Every option defaults to ``None``, meaning "use the PIConGPU default"
    (``include/picongpu/simulation/stage/Poisson.hpp``): 2000 solver steps, a
    tolerance of 1e-8, the Chebyshev preconditioner enabled, and 20
    preconditioner steps. Options left at ``None`` are not emitted into the
    generated parameter file, so the C++ defaults apply.
    """

    max_steps: Annotated[int, Field(gt=0)] | None = None
    """maximum number of iterations for the Poisson solver, ``None`` -> PIConGPU default (2000)"""

    tolerance: Annotated[float, Field(gt=0.0)] | None = None
    """maximum tolerance for the Poisson solver, ``None`` -> PIConGPU default (1e-8)"""

    preconditioner: Literal["chebyshev", "none"] | None = None
    """preconditioner for the Poisson solver, ``None`` -> PIConGPU default (chebyshev, enabled)"""

    preconditioner_max_steps: Annotated[int, Field(gt=0)] | None = None
    """maximum number of iterations for the preconditioner, ``None`` -> PIConGPU default (20)"""

    @computed_field
    def preconditioner_disabled(self) -> bool | None:
        if self.preconditioner is None:
            return None
        return self.preconditioner == "none"
