import numpy as np
import pytest
import tensorflow as tf

from neuraldist.generative_model.realnvp.realnvp import (CouplingLayer, RealNVP, SymmetricSTNetwork,
                                                         cRealNVP, default_masks)

SMALL = [{"neurons": 8, "activation": "tanh"}]


@pytest.fixture(autouse=True)
def seed():
    tf.keras.utils.set_random_seed(0)


def rand(*shape, seed=0):
    return tf.constant(np.random.default_rng(seed).normal(size=shape), tf.float32)


# --- coupling layer --------------------------------------------------------------------------

@pytest.mark.parametrize("cond_dim", [0, 2])
def test_coupling_layer_inverse_and_log_det(cond_dim):
    layer = CouplingLayer(3, cond_dim, [1, 0, 1], SMALL, scale_clamp=2.0)
    y, cond = rand(50, 3), (rand(50, cond_dim, seed=1) if cond_dim else None)
    z, ld_fwd = layer(y, cond)
    y2, ld_inv = layer(z, cond, inverse=True)
    np.testing.assert_allclose(y2, y, atol=1e-5)
    np.testing.assert_allclose(ld_fwd, -ld_inv, atol=1e-5)
    # kept dimensions pass through unchanged
    np.testing.assert_array_equal(tf.gather(z, [0, 2], axis=1), tf.gather(y, [0, 2], axis=1))
    # log-scale is bounded by the clamp: |log det| <= clamp * n_transformed
    assert np.all(np.abs(ld_fwd) <= 2.0 + 1e-6)


@pytest.mark.parametrize("mask", [[1, 1, 1], [0, 0, 0], [1, 0]])
def test_coupling_layer_rejects_bad_masks(mask):
    with pytest.raises(ValueError):
        CouplingLayer(3, 0, mask)


def test_symmetric_st_network_is_sign_invariant():
    net = SymmetricSTNetwork(1, 2, 2, SMALL)
    x, c = rand(20, 1), rand(20, 2, seed=1)
    s1, t1 = net(x, c)
    s2, t2 = net(x, -c * tf.constant([[1.0, -1.0]]))
    np.testing.assert_allclose(s1, s2, atol=1e-6)
    np.testing.assert_allclose(t1, t2, atol=1e-6)
    with pytest.raises(ValueError):
        net(x, None)


def test_default_masks():
    m3 = default_masks(3, 8)
    assert len(m3) == 8 and m3[6] == m3[0]
    for d in (2, 4, 5):
        masks = default_masks(d, 4)
        assert all(len(m) == d and 0 < sum(m) < d for m in masks)
        assert masks[0] != masks[1]   # alternating checkerboard


# --- full flow -------------------------------------------------------------------------------

@pytest.mark.parametrize("x_dim, cond_dim", [(2, 0), (3, 0), (3, 2), (4, 1)])
def test_forward_inverse_roundtrip(x_dim, cond_dim):
    model = RealNVP(x_dim, cond_dim, hidden_layers=SMALL, n_coupling=4)
    x, c = rand(64, x_dim), (rand(64, cond_dim, seed=1) if cond_dim else None)
    z, _ = model.forward(x, c)
    np.testing.assert_allclose(model.inverse(z, c), x, atol=1e-4)


@pytest.mark.parametrize("cond_dim", [0, 2])
def test_log_det_matches_autodiff_jacobian(cond_dim):
    model = RealNVP(3, cond_dim, hidden_layers=SMALL, n_coupling=4, scale_clamp=2.0)
    x, c = rand(10, 3), (rand(10, cond_dim, seed=1) if cond_dim else None)
    with tf.GradientTape() as tape:
        tape.watch(x)
        z, log_det = model.forward(x, c)
    jac = tape.batch_jacobian(z, x)                                 # [batch, 3, 3]
    _, ref = np.linalg.slogdet(jac.numpy().astype("float64"))
    np.testing.assert_allclose(log_det, ref, atol=1e-4)


def test_log_prob_is_normalized_2d():
    """Untrained flow on a grid: the density must integrate to one."""
    model = RealNVP(2, hidden_layers=SMALL, n_coupling=4, scale_clamp=0.5)
    g = np.linspace(-12, 12, 401)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], -1).astype("float32")
    p = np.exp(model.log_prob(pts).numpy().astype("float64"))
    assert np.sum(p) * (g[1] - g[0]) ** 2 == pytest.approx(1.0, abs=1e-2)


def test_sample_shapes_and_explicit_latent():
    model = RealNVP(3, hidden_layers=SMALL, n_coupling=2)
    assert model.sample(7).shape == (7, 3)
    z = rand(5, 3)
    np.testing.assert_allclose(model.sample(5, z=z), model.inverse(z), atol=1e-6)

    cmodel = cRealNVP(3, 2, hidden_layers=SMALL, n_coupling=2)
    assert cmodel.sample(4, cond=rand(6, 2)).shape == (6, 4, 3)
    with pytest.raises(tf.errors.InvalidArgumentError):
        cmodel.sample(4, cond=rand(6, 2), z=rand(5, 3))


def test_sample_with_latent_is_consistent_with_log_prob():
    """x = inverse(z) must be mapped back to z, so log_prob(x) = log N(z) + log|det dz/dx|."""
    model = cRealNVP(2, 1, hidden_layers=SMALL, n_coupling=3)
    c, z = rand(1, 1), rand(20, 2, seed=3)
    x = model.sample(20, cond=c, z=z)[0]
    z_back, log_det = model.forward(x, tf.repeat(c, 20, axis=0))
    np.testing.assert_allclose(z_back, z, atol=1e-4)
    np.testing.assert_allclose(model.log_prob(x, tf.repeat(c, 20, axis=0)),
                               model.prior.log_prob(z) + log_det, atol=1e-4)


def test_constructor_validation():
    with pytest.raises(ValueError):
        RealNVP(1)
    with pytest.raises(ValueError):
        RealNVP(3, symmetric=True)
    with pytest.raises(ValueError):
        RealNVP(3, n_coupling=3, scale_clamp=[1.0, 1.0])
    with pytest.raises(ValueError):
        cRealNVP(3, 0)
    model = RealNVP(3, n_coupling=3, scale_clamp=[0.5, 1, 2], hidden_layers=SMALL)
    assert [layer.stnet.scale_clamp for layer in model.coupling] == [0.5, 1.0, 2.0]


def test_condition_checks():
    with pytest.raises(ValueError):
        RealNVP(3, hidden_layers=SMALL).log_prob(rand(4, 3), rand(4, 1))
    with pytest.raises(ValueError):
        cRealNVP(3, 1, hidden_layers=SMALL).log_prob(rand(4, 3))
    with pytest.raises(ValueError):
        cRealNVP(3, 1, hidden_layers=SMALL).sample(4)


def test_save_and_load_weights(tmp_path):
    kw = dict(x_dim=3, cond_dim=2, hidden_layers=SMALL, n_coupling=3)
    a, b = cRealNVP(**kw), cRealNVP(**kw)
    a.warmup(), b.warmup()
    x, c = rand(8, 3), rand(8, 2, seed=1)
    assert not np.allclose(a.log_prob(x, c), b.log_prob(x, c))
    path = str(tmp_path / "flow.weights.h5")
    a.save_weights(path)
    b.load_weights(path)
    np.testing.assert_allclose(a.log_prob(x, c), b.log_prob(x, c), atol=1e-6)
