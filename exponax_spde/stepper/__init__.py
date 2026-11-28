"""Stochastic PDE steppers."""

from ._allen_cahn import StochasticAllenCahn, phi1

__all__ = ["StochasticAllenCahn", "phi1"]


def __getattr__(name: str) -> None:
    # Migration messages for v0.3.0. Remove in v0.4.0.
    if name in {"Burgers", "generic", "reaction", "HeatEquation"}:
        replacement = "Diffusion" if name == "HeatEquation" else name
        raise AttributeError(
            f"{__name__}.{name} has moved. Use exponax.stepper.{replacement}."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
