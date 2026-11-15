import numpy as np
import pytest
import tensorflow as tf
from scipy import stats

from neuraldist.losses import CollisionDBLoss, DBLoss, SurfaceDBLoss

ALPHA_T, ALPHA_N = 0.5, 0.6
RHO, VAR_T = 1 - ALPHA_T, ALPHA_T * (2 - ALPHA_T)


def cl_log_kernel(t_r, u_r, t_i, u_i):
    """Exact Cercignani-Lampis log K(v_r | v_i) (kT = m = 1), t of shape (N, d_t)."""
    s, nu = np.sqrt(ALPHA_N), np.sqrt(1 - ALPHA_N) * u_i
    return (stats.norm.logpdf(t_r, RHO * t_i, np.sqrt(VAR_T)).sum(1)
            + stats.rice.logpdf(u_r, nu / s, scale=s))


def notebook_pair_arrays(t_i, u_i, t_r, u_r):
    """Reference: the former pair_arrays of the gas-surface notebooks (kT = m = 1)."""
    t_i, t_r = np.atleast_2d(t_i.T).T, np.atleast_2d(t_r.T).T
    log_M = lambda t, u: np.log(u) - 0.5 * ((t**2).sum(1) + u**2) - 0.5 * (t.shape[1] + 1) * np.log(2 * np.pi)
    f = lambda a: np.asarray(a, dtype="float32")
    return dict(
        v_i=f(np.column_stack([t_i, np.log(u_i)])), v_r=f(np.column_stack([t_r, np.log(u_r)])),
        v_i_bwd=f(np.column_stack([-t_r, np.log(u_r)])), v_r_bwd=f(np.column_stack([-t_i, np.log(u_i)])),
        log_det_jacobian_fwd=f(-np.log(u_r)), log_det_jacobian_bwd=f(-np.log(u_i)),
        log_M_i=f(log_M(t_i, u_i)), log_M_r=f(log_M(t_r, u_r)),
    )


def random_pairs(n, t_dim, rng):
    shape = (n,) if t_dim is None else (n, t_dim)
    return rng.normal(0, 1.5, shape), rng.uniform(0.1, 3.5, n), rng.normal(0, 1.5, shape), rng.uniform(0.1, 3.5, n)


@pytest.mark.parametrize("t_dim", [None, 1, 2])
def test_matches_notebook_format(t_dim):
    pairs = random_pairs(500, t_dim, np.random.default_rng(0))
    got, ref = SurfaceDBLoss().make_db_data(*pairs), notebook_pair_arrays(*pairs)
    assert got.keys() == ref.keys()
    for k in ref:
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-6, atol=1e-6, err_msg=k)


def test_exact_cl_kernel_has_zero_residual():
    loss = SurfaceDBLoss()
    t_i, u_i, t_r, u_r = random_pairs(2000, 2, np.random.default_rng(1))
    d = loss.make_db_data(t_i, u_i, t_r, u_r)
    # log q in abstract variables = log K + log u of the target
    lq_f = cl_log_kernel(t_r, u_r, t_i, u_i) + np.log(u_r)
    lq_b = cl_log_kernel(-t_i, u_i, -t_r, u_r) + np.log(u_i)
    value, acc = loss(tf.constant(lq_f, tf.float32), tf.constant(lq_b, tf.float32), d)
    assert float(value) < 1e-6 and float(acc) > 0.999


def test_abstract_roundtrip_and_validation():
    rng = np.random.default_rng(2)
    t, u = rng.normal(size=(10, 2)), rng.uniform(0.1, 2, 10)
    t2, u2 = SurfaceDBLoss.from_abstract(SurfaceDBLoss.to_abstract(t, u))
    np.testing.assert_allclose(t2, t)
    np.testing.assert_allclose(u2, u)
    with pytest.raises(ValueError):
        SurfaceDBLoss().make_db_data(t, -u, t, u)


def test_kinds_in_config():
    assert DBLoss().get_config()["loss:db:type"] == "generic"
    assert SurfaceDBLoss().get_config()["loss:db:type"] == "surface"
    assert CollisionDBLoss().get_config()["loss:db:type"] == "collision"
