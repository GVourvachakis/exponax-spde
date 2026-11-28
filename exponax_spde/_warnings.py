"""Warnings emitted by stochastic steppers."""

import sys
import warnings


class NoiseRegularityWarning(UserWarning):
    """The finite-grid noise is too rough for the classical continuum model."""


class TamingOrderWarning(UserWarning):
    """Taming limits convergence to first order against the untamed model."""


def _warn_at_caller(message: str, category: type[Warning]) -> None:
    # Equinox wraps construction in a version-dependent number of frames.
    frame = sys._getframe()
    stacklevel = 1
    while frame is not None:
        package = frame.f_globals.get("__name__", "").partition(".")[0]
        if package not in {"equinox", "exponax_spde"}:
            break
        stacklevel += 1
        frame = frame.f_back
    warnings.warn(message, category, stacklevel=stacklevel)
