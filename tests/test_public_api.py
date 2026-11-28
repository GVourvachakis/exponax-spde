from types import ModuleType

import exponax
import pytest

import exponax_spde as spde

ROOT_API = [
    "StochasticAllenCahn",
    "TamedPolynomialNonlinearFun",
    "NoiseRegularityWarning",
    "TamingOrderWarning",
    "stochastic_rollout",
    "stochastic_ensemble_rollout",
    "structure_factor",
    "stochastic_structure_factor",
    "richardson_weak_extrapolation",
    "strang_split_step",
    "stepper",
    "nonlin_fun",
    "__version__",
]


@pytest.mark.parametrize(
    ("module", "exports"),
    [
        (spde, ROOT_API),
        (spde.stepper, ["StochasticAllenCahn", "phi1"]),
        (spde.nonlin_fun, ["TamedPolynomialNonlinearFun"]),
    ],
)
def test_public_namespace(module, exports):
    assert module.__all__ == exports
    public = {name for name in dir(module) if not name.startswith("_")}
    assert public == set(exports) - {"__version__"}

    for name in exports:
        value = getattr(module, name)
        if name == "__version__":
            assert isinstance(value, str)
        else:
            origin = (
                value.__name__ if isinstance(value, ModuleType) else value.__module__
            )
            assert origin.startswith("exponax_spde."), name


def test_stochastic_classes_use_upstream_bases():
    assert issubclass(spde.StochasticAllenCahn, exponax.BaseStepper)
    assert issubclass(
        spde.TamedPolynomialNonlinearFun, exponax.nonlin_fun.BaseNonlinearFun
    )


@pytest.mark.parametrize(
    ("module", "name", "replacement"),
    [
        (spde, "fft", "exponax.fft"),
        (spde, "ic", "exponax.ic"),
        (spde.stepper, "Burgers", "exponax.stepper.Burgers"),
        (spde.stepper, "HeatEquation", "exponax.stepper.Diffusion"),
    ],
)
def test_migration_errors(module, name, replacement):
    with pytest.raises(AttributeError) as exc:
        getattr(module, name)

    assert str(exc.value) == f"{module.__name__}.{name} has moved. Use {replacement}."
