import importlib

import numpy as np
import pytest
import tensorflow as tf
from scipy import stats

from neuraldist.generative_model.activation_functions import softplus
from neuraldist.generative_model.base import GenerativeModel
from neuraldist.generative_model.distributions import StandardNormal
from neuraldist.generative_model.mlp import make_mlp
from neuraldist.generative_model.registry import build_model, list_models, register_model


def test_registry_contains_all_models():
    importlib.import_module("neuraldist.generative_model.realnvp")
    importlib.import_module("neuraldist.generative_model.vae")
    assert {"realnvp", "crealnvp", "crealnvp_db", "vae", "cvae"} <= set(list_models())
    model = build_model("realnvp", x_dim=2, n_coupling=2)
    assert type(model).__name__ == "RealNVP" and model.x_dim == 2


def test_registry_unknown_and_custom():
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("does-not-exist")

    @register_model("_test_dummy")
    class Dummy:
        def __init__(self, a):
            self.a = a

    assert build_model("_test_dummy", a=3).a == 3


def test_package_exports():
    """Everything listed in __all__ must be importable."""
    for name in ("neuraldist.generative_model.realnvp", "neuraldist.generative_model.vae",
                 "neuraldist.losses", "neuraldist.training"):
        mod = importlib.import_module(name)
        for attr in mod.__all__:
            assert hasattr(mod, attr), f"{name}.{attr}"


@pytest.mark.parametrize("dim", [1, 3])
def test_standard_normal(dim):
    dist = StandardNormal(dim)
    x = np.random.default_rng(0).normal(size=(20, dim)).astype("float32")
    ref = stats.multivariate_normal(np.zeros(dim), np.eye(dim)).logpdf(x)
    np.testing.assert_allclose(dist.log_prob(x), ref, rtol=1e-5)
    assert dist.sample(7).shape == (7, dim)


def test_softplus():
    f = softplus(2.0)
    x = tf.constant([-3.0, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(f(x), np.log1p(np.exp(2 * x.numpy())) / 2, rtol=1e-4)
    assert f.__name__ == "softplus_2.0"
    # no overflow for large inputs, asymptotically linear / zero
    big = f(tf.constant([-100.0, 50.0, 1e4]))
    assert np.all(np.isfinite(big))
    np.testing.assert_allclose(big, [0.0, 50.0, 1e4], atol=1e-6)


def test_make_mlp():
    mlp = make_mlp([{"neurons": 5, "activation": "relu"}, {"neurons": 4, "activation": "tanh"}],
                   [{"neurons": 2, "activation": "linear"}])
    assert mlp(tf.zeros((3, 7))).shape == (3, 2)
    assert [layer.units for layer in mlp.layers] == [5, 4, 2]


def test_base_class_interface():
    unc, cond = GenerativeModel(2), GenerativeModel(2, cond_dim=1)
    assert not unc.conditional and cond.conditional
    for call in (lambda: unc.compute_losses(tf.zeros((1, 2))), lambda: unc.sample(1),
                 lambda: unc.log_prob(tf.zeros((1, 2)))):
        with pytest.raises(NotImplementedError):
            call()

    x, c = unc._unpack((np.zeros((2, 2)),))
    assert x.dtype == tf.float32 and c is None
    x, c = cond._unpack((np.zeros((2, 2)), np.ones((2, 1))))
    assert c.dtype == tf.float32
    with pytest.raises(ValueError):
        cond._unpack(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        unc._unpack((np.zeros((2, 2)), np.zeros((2, 1))))


def test_expand_cond_repeats_rows():
    model = GenerativeModel(2, cond_dim=1)
    c = tf.constant([[1.0], [2.0]])
    rep, lead, rows = model._expand_cond(3, c)
    np.testing.assert_array_equal(rep[:, 0], [1, 1, 1, 2, 2, 2])
    assert lead.numpy().tolist() == [2, 3] and int(rows) == 6
    # rows of the flat batch end up in the right [B, n] slot
    out = model._to_lead_shape(tf.concat([rep, rep], -1), lead)
    np.testing.assert_array_equal(out[1, :, 0], [2, 2, 2])
