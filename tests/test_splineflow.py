import numpy as np
import pytest
import tensorflow as tf

from neuraldist.generative_model.splineflow.splineflow import (SplineCouplingLayer, SplineFlow,
                                                               SymmetricSplineNetwork, cSplineFlow,
                                                               rational_quadratic_spline)
from neuraldist.training import fit_model

SMALL = [{"neurons": 8, "activation": "tanh"}]


@pytest.fixture(autouse=True)
def seed():
    tf.keras.utils.set_random_seed(0)


def rand(*shape, seed=0, scale=1.0):
    return tf.constant(scale * np.random.default_rng(seed).normal(size=shape), tf.float32)


def spline_params(n, d, k, seed=1):
    return rand(n, d, k, seed=seed), rand(n, d, k, seed=seed + 1), rand(n, d, k - 1, seed=seed + 2)


# --- rational-quadratic spline ---------------------------------------------------------------

def test_spline_inverse_log_det_and_tails():
    x = rand(500, 2, scale=3.0)                        # some points outside [-5, 5]
    w, h, d = spline_params(500, 2, 6)
    y, ld = rational_quadratic_spline(x, w, h, d, tail_bound=5.0)
    x2, ld_inv = rational_quadratic_spline(y, w, h, d, inverse=True, tail_bound=5.0)
    np.testing.assert_allclose(x2, x, atol=1e-3)
    np.testing.assert_allclose(ld, -ld_inv, atol=1e-3)

    outside = np.abs(x.numpy()) > 5.0
    assert outside.any()
    np.testing.assert_array_equal(y.numpy()[outside], x.numpy()[outside])
    np.testing.assert_array_equal(ld.numpy()[outside], 0.0)
    # [-B, B] is mapped onto itself
    assert np.all(np.abs(y.numpy()[~outside]) <= 5.0 + 1e-5)


def test_spline_is_monotonic_and_log_det_matches_autodiff():
    x = tf.constant(np.linspace(-6, 6, 1001)[:, None], tf.float32)
    w, h, d = [tf.repeat(p, 1001, axis=0) for p in spline_params(1, 1, 8)]
    with tf.GradientTape() as tape:
        tape.watch(x)
        y, ld = rational_quadratic_spline(x, w, h, d, tail_bound=4.0)
    assert np.all(np.diff(y.numpy()[:, 0]) > 0)
    np.testing.assert_allclose(ld, np.log(tape.gradient(y, x).numpy()), atol=1e-4)


def test_spline_zero_params_is_identity():
    x = rand(50, 3)
    y, ld = rational_quadratic_spline(x, tf.zeros((50, 3, 5)), tf.zeros((50, 3, 5)),
                                      tf.zeros((50, 3, 4)))
    np.testing.assert_allclose(y, x, atol=1e-5)
    np.testing.assert_allclose(ld, 0.0, atol=1e-5)


def test_spline_gradients_are_finite_for_tail_inputs():
    x = tf.constant([[-20.0], [0.3], [20.0]])
    w, h, d = [tf.Variable(p) for p in spline_params(3, 1, 4)]
    with tf.GradientTape() as tape:
        y, ld = rational_quadratic_spline(x, w, h, d, tail_bound=3.0)
        loss = tf.reduce_sum(y) + tf.reduce_sum(ld)
    for g in tape.gradient(loss, [w, h, d]):
        assert np.all(np.isfinite(g.numpy()))


# --- coupling layer --------------------------------------------------------------------------

@pytest.mark.parametrize("cond_dim", [0, 2])
def test_coupling_layer_inverse_and_log_det(cond_dim):
    layer = SplineCouplingLayer(3, cond_dim, [1, 0, 1], n_bins=6, hidden_layers=SMALL)
    y, cond = rand(50, 3), (rand(50, cond_dim, seed=1) if cond_dim else None)
    z, ld_fwd = layer(y, cond)
    y2, ld_inv = layer(z, cond, inverse=True)
    np.testing.assert_allclose(y2, y, atol=1e-4)
    np.testing.assert_allclose(ld_fwd, -ld_inv, atol=1e-4)
    np.testing.assert_array_equal(tf.gather(z, [0, 2], axis=1), tf.gather(y, [0, 2], axis=1))


@pytest.mark.parametrize("kwargs", [dict(mask=[1, 1, 1]), dict(mask=[0, 0, 0]), dict(mask=[1, 0]),
                                    dict(mask=[1, 0, 1], n_bins=1),
                                    dict(mask=[1, 0, 1], tail_bound=0.0)])
def test_coupling_layer_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        SplineCouplingLayer(3, 0, **kwargs)


def test_symmetric_spline_network_is_sign_invariant():
    net = SymmetricSplineNetwork(1, 2, 2, n_bins=4, hidden_layers=SMALL)
    x, c = rand(20, 1), rand(20, 2, seed=1)
    out1 = net(x, c)
    out2 = net(x, -c * tf.constant([[1.0, -1.0]]))
    for a, b in zip(out1, out2):
        np.testing.assert_allclose(a, b, atol=1e-6)
    assert [p.shape for p in out1] == [(20, 2, 4), (20, 2, 4), (20, 2, 3)]
    with pytest.raises(ValueError):
        net(x, None)


# --- full flow -------------------------------------------------------------------------------

@pytest.mark.parametrize("x_dim, cond_dim", [(2, 0), (3, 0), (3, 2), (4, 1)])
def test_forward_inverse_roundtrip(x_dim, cond_dim):
    model = SplineFlow(x_dim, cond_dim, hidden_layers=SMALL, n_coupling=4)
    x, c = rand(64, x_dim, scale=2.0), (rand(64, cond_dim, seed=1) if cond_dim else None)
    z, _ = model.forward(x, c)
    np.testing.assert_allclose(model.inverse(z, c), x, atol=1e-3)


@pytest.mark.parametrize("cond_dim", [0, 2])
def test_log_det_matches_autodiff_jacobian(cond_dim):
    model = SplineFlow(3, cond_dim, hidden_layers=SMALL, n_coupling=4)
    x, c = rand(10, 3), (rand(10, cond_dim, seed=1) if cond_dim else None)
    with tf.GradientTape() as tape:
        tape.watch(x)
        z, log_det = model.forward(x, c)
    jac = tape.batch_jacobian(z, x)
    _, ref = np.linalg.slogdet(jac.numpy().astype("float64"))
    np.testing.assert_allclose(log_det, ref, atol=1e-3)


def test_log_prob_is_normalized_2d():
    """Untrained flow on a grid: the density must integrate to one."""
    model = SplineFlow(2, hidden_layers=SMALL, n_coupling=3, tail_bound=3.0)
    g = np.linspace(-10, 10, 801)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], -1).astype("float32")
    p = np.exp(model.log_prob(pts).numpy().astype("float64"))
    assert np.sum(p) * (g[1] - g[0]) ** 2 == pytest.approx(1.0, abs=1e-2)


def test_sample_shapes_and_explicit_latent():
    model = SplineFlow(3, hidden_layers=SMALL, n_coupling=2)
    assert model.sample(7).shape == (7, 3)
    z = rand(5, 3)
    np.testing.assert_allclose(model.sample(5, z=z), model.inverse(z), atol=1e-6)

    cmodel = cSplineFlow(3, 2, hidden_layers=SMALL, n_coupling=2)
    assert cmodel.sample(4, cond=rand(6, 2)).shape == (6, 4, 3)
    with pytest.raises(tf.errors.InvalidArgumentError):
        cmodel.sample(4, cond=rand(6, 2), z=rand(5, 3))


def test_constructor_and_condition_checks():
    with pytest.raises(ValueError):
        SplineFlow(1)
    with pytest.raises(ValueError):
        SplineFlow(3, symmetric=True)
    with pytest.raises(ValueError):
        cSplineFlow(3, 0)
    with pytest.raises(ValueError):
        SplineFlow(3, hidden_layers=SMALL).log_prob(rand(4, 3), rand(4, 1))
    with pytest.raises(ValueError):
        cSplineFlow(3, 1, hidden_layers=SMALL).log_prob(rand(4, 3))
    with pytest.raises(ValueError):
        cSplineFlow(3, 1, hidden_layers=SMALL).sample(4)


def test_save_and_load_weights(tmp_path):
    kw = dict(x_dim=3, cond_dim=2, hidden_layers=SMALL, n_coupling=3)
    a, b = cSplineFlow(**kw), cSplineFlow(**kw)
    a.warmup(), b.warmup()
    x, c = rand(8, 3), rand(8, 2, seed=1)
    assert not np.allclose(a.log_prob(x, c), b.log_prob(x, c))
    path = str(tmp_path / "flow.weights.h5")
    a.save_weights(path)
    b.load_weights(path)
    np.testing.assert_allclose(a.log_prob(x, c), b.log_prob(x, c), atol=1e-6)


def test_fits_bimodal_conditional_distribution():
    """cond = +-1 selects the sign of the mean of a bimodal target; training must lower the NLL
    and the samples must follow the condition."""
    rng = np.random.default_rng(0)
    n = 4000
    c = rng.choice([-1.0, 1.0], size=(n, 1)).astype("float32")
    x = (c * 1.5 + 0.3 * rng.normal(size=(n, 2))).astype("float32")
    model = cSplineFlow(2, 1, n_coupling=2, n_bins=8, tail_bound=4.0)
    hist = fit_model(model, x, c, epochs=15, batch_size=256, learning_rate=3e-3, seed=0, verbose=0)
    assert hist.history["loss"][-1] < hist.history["loss"][0] - 1.0
    s = model.sample(500, cond=np.array([[-1.0], [1.0]], dtype="float32")).numpy()
    np.testing.assert_allclose(s[0].mean(0), [-1.5, -1.5], atol=0.3)
    np.testing.assert_allclose(s[1].mean(0), [1.5, 1.5], atol=0.3)
