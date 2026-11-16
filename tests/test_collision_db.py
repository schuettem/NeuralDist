import numpy as np
import pytest
import tensorflow as tf

from neuraldist.losses import CollisionDBLoss, from_relative_frame, reduced_mass, to_relative_frame
from neuraldist.training import fit_model

MU, KT = 0.5, 1.0
VAR = KT / MU


def ou_log_q(g_post, g_pre, rho=0.6):
    """Gaussian (Ornstein-Uhlenbeck) kernel: reversible w.r.t. M_mu, i.e. DB for Maxwell molecules."""
    var = (1 - rho**2) * VAR
    r = g_post - rho * g_pre
    return -0.5 * np.sum(r**2, -1) / var - 0.5 * g_pre.shape[-1] * np.log(2 * np.pi * var)


def residual(loss, log_q, g_pre, g_post):
    d = loss.make_db_data(g_pre, g_post)
    lq_f = tf.constant(log_q(d["v_r"].astype("float64"), d["v_i"].astype("float64")), tf.float32)
    lq_b = tf.constant(log_q(d["v_r_bwd"].astype("float64"), d["v_i_bwd"].astype("float64")), tf.float32)
    return loss(lq_f, lq_b, d)


@pytest.fixture
def pairs():
    rng = np.random.default_rng(0)
    return rng.normal(0, 2, (2000, 3)), rng.normal(0, 2, (2000, 3))


@pytest.mark.parametrize("time_reversal", [True, False])
def test_exact_ou_kernel_maxwell_molecules(pairs, time_reversal):
    loss = CollisionDBLoss(MU, KT, rate_exponent=0.0, time_reversal=time_reversal)
    value, acc = residual(loss, ou_log_q, *pairs)
    assert float(value) < 1e-6 and float(acc) > 0.999


def test_exact_thermalizing_kernel_hard_spheres(pairs):
    """q(g'|g) ~ B(g') M(g') (memoryless, collision-rate weighted) is in DB for any rate law."""
    loss = CollisionDBLoss(MU, KT, rate_exponent=1.0)
    value, _ = residual(loss, lambda gp, g: loss.log_weight(gp), *pairs)
    assert float(value) < 1e-6


def test_wrong_rate_law_is_detected(pairs):
    loss = CollisionDBLoss(MU, KT, rate_exponent=1.0)       # OU kernel is not in DB with B ~ |g|
    value, acc = residual(loss, ou_log_q, *pairs)
    assert float(value) > 0.1 and float(acc) < 0.9


def test_db_data_format(pairs):
    loss = CollisionDBLoss(MU, KT)
    d = loss.make_db_data(*pairs)
    np.testing.assert_allclose(d["v_i_bwd"], -d["v_r"])
    np.testing.assert_allclose(d["v_r_bwd"], -d["v_i"])
    assert all(v.dtype == np.float32 and len(v) == 2000 for v in d.values())
    assert loss.get_config()["loss:db:type"] == "collision"


def test_frame_roundtrip_and_conservation():
    rng = np.random.default_rng(1)
    m1, m2 = 1.0, 3.0
    v1, v2 = rng.normal(size=(100, 3)), rng.normal(size=(100, 3))
    G, g = to_relative_frame(v1, v2, m1, m2)
    w1, w2 = from_relative_frame(G, g, m1, m2)
    np.testing.assert_allclose(w1, v1, atol=1e-12)
    np.testing.assert_allclose(w2, v2, atol=1e-12)
    # kinetic energy splits into centre-of-mass and relative part
    e_lab = 0.5 * m1 * np.sum(v1**2, -1) + 0.5 * m2 * np.sum(v2**2, -1)
    e_split = 0.5 * (m1 + m2) * np.sum(G**2, -1) + 0.5 * reduced_mass(m1, m2) * np.sum(g**2, -1)
    np.testing.assert_allclose(e_lab, e_split)


@pytest.mark.parametrize("name", ["crealnvp_db", "csplineflow_db"])
def test_training_smoke(name):
    import neuraldist.generative_model.realnvp.crealnvp_db  # noqa: F401  registers "crealnvp_db"
    import neuraldist.generative_model.splineflow.csplineflow_db  # noqa: F401  registers "csplineflow_db"
    from neuraldist.generative_model.registry import build_model

    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    g = rng.normal(0, np.sqrt(VAR), (200, 3))
    g_post = 0.6 * g + np.sqrt(1 - 0.36) * rng.normal(0, np.sqrt(VAR), g.shape)
    loss = CollisionDBLoss(MU, KT, rate_exponent=0.0)
    model = build_model(name, x_dim=3, cond_dim=3, n_coupling=4)
    hist = fit_model(model, g_post, g, db_data=loss.sample_db_points(512, rng=1),
                     compile_kwargs={"db_loss": loss}, epochs=3, batch_size=128)
    assert np.isfinite(hist.history["db_loss"]).all()
