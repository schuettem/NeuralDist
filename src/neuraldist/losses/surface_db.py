"""
    Detailed-balance (DB) loss for gas-surface scattering kernels.

    Physics
    -------
    A particle hits a wall with velocity v_i = (t_i, n_i) (tangential components t, normal component n) and
    leaves with v_r = (t_r, n_r). The scattering kernel K(v_i -> v_r) of a wall in equilibrium (Maxwellian
    M at the wall temperature kT) satisfies
        |n_i| M(v_i) K(v_i -> v_r) = |n_r| M(v_r) K(-v_r -> -v_i)
    The flux factor |n| makes |n| M the distribution of particles *hitting* the wall.

    Variables
    ---------
    Physical: tangential components t (N, d_t) and normal speed u = |n| > 0 (N,).
    Abstract (what the flow works in): y = (t, log u) in R^(d_t + 1), so the flow needs no constraint u > 0.
    The density transforms with log|det d y / d v| = -log u of the target velocity.
    Time reversal: the reversed pair has the condition (-t_r, u_r) and the target (-t_i, u_i).

    This is the gas-surface counterpart of `CollisionDBLoss`; both produce the same DB data format
    (keys DB_KEYS of training.make_db_dataset) and work with `cRealNVPDB` and `fit_model`.
"""
from __future__ import annotations

import numpy as np

from .db import DBLoss


def _as_2d(t) -> np.ndarray:
    """Tangential components as (N, d_t); a 1D array is one tangential component."""
    t = np.asarray(t, dtype="float64")
    return t[:, None] if t.ndim == 1 else t


class SurfaceDBLoss(DBLoss):
    """
    DB loss for a gas-surface scattering kernel K(v_i -> v_r) in the abstract variables (t, log u).

    mass, kT        : particle mass and wall temperature of the Maxwellian M (variance kT/m per component)
    epoch_weighting : float or weighting object (ConstantWeight, LinearRampWeight, ...), see DBLoss
    enabled         : switch the loss on/off without removing it from compile()

    The residual is the one of DBLoss:
        [log q(y_r|y_i) + log J(u_r) + log(|n_i| M(v_i))] - [log q(y_i'|y_r') + log J(u_i) + log(|n_r| M(v_r))]
    `make_db_data` / `sample_db_points` build the DB batch dict.
    """

    kind = "surface"

    def __init__(self, mass: float = 1.0, kT: float = 1.0, epoch_weighting=1.0, enabled: bool = True):
        super().__init__(epoch_weighting=epoch_weighting, enabled=enabled)
        if mass <= 0 or kT <= 0:
            raise ValueError("mass and kT must be positive.")
        self.mass = float(mass)
        self.kT = float(kT)

    # --- variables --------------------------------------------------------------------------------
    @property
    def thermal_variance(self) -> float:
        """Variance kT / m of each velocity component of the wall Maxwellian."""
        return self.kT / self.mass

    @staticmethod
    def to_abstract(t, u) -> np.ndarray:
        """Physical (t, u = |n|) -> abstract y = (t, log u), shape (N, d_t + 1)."""
        return np.column_stack([_as_2d(t), np.log(np.asarray(u, dtype="float64"))])

    @staticmethod
    def from_abstract(y):
        """Abstract y = (t, log u) -> physical (t (N, d_t), u (N,)). Use it for sampled reflected velocities."""
        y = np.asarray(y, dtype="float64")
        return y[..., :-1], np.exp(y[..., -1])

    @staticmethod
    def log_det_jacobian(u) -> np.ndarray:
        """log|det d y / d v| of the target velocity: log p_phys = log p_abstract - log u."""
        return -np.log(np.asarray(u, dtype="float64"))

    # --- equilibrium weight -----------------------------------------------------------------------
    def log_maxwellian(self, t, u) -> np.ndarray:
        """log M(v): normalized wall Maxwellian in d_t + 1 dimensions."""
        t, u = _as_2d(t), np.asarray(u, dtype="float64")
        var, d = self.thermal_variance, t.shape[1] + 1
        return -0.5 * (np.sum(t**2, axis=1) + u**2) / var - 0.5 * d * np.log(2 * np.pi * var)

    def log_weight(self, t, u) -> np.ndarray:
        """log(|n| M(v)): log density (flux-weighted) of the particles hitting the wall."""
        return np.log(np.asarray(u, dtype="float64")) + self.log_maxwellian(t, u)

    # --- DB data ----------------------------------------------------------------------------------
    def make_db_data(self, t_i, u_i, t_r, u_r) -> dict:
        """Pairs (v_i -> v_r) in physical variables -> dict with the DB keys (abstract variables).

        t_i, t_r : tangential components, (N, d_t) or (N,) for one component
        u_i, u_r : normal speeds |n| > 0, (N,)
        """
        t_i, t_r = _as_2d(t_i), _as_2d(t_r)
        u_i, u_r = np.asarray(u_i, dtype="float64"), np.asarray(u_r, dtype="float64")
        if t_i.shape != t_r.shape or u_i.shape != u_r.shape or len(u_i) != len(t_i) or u_i.ndim != 1:
            raise ValueError(f"Inconsistent shapes: t_i {t_i.shape}, t_r {t_r.shape}, u_i {u_i.shape}, "
                             f"u_r {u_r.shape}.")
        if np.any(u_i <= 0) or np.any(u_r <= 0):
            raise ValueError("Normal speeds u_i, u_r must be positive.")
        f32 = lambda a: np.asarray(a, dtype="float32")
        return dict(
            v_i=f32(self.to_abstract(t_i, u_i)), v_r=f32(self.to_abstract(t_r, u_r)),
            v_i_bwd=f32(self.to_abstract(-t_r, u_r)), v_r_bwd=f32(self.to_abstract(-t_i, u_i)),
            log_det_jacobian_fwd=f32(self.log_det_jacobian(u_r)),
            log_det_jacobian_bwd=f32(self.log_det_jacobian(u_i)),
            log_M_i=f32(self.log_weight(t_i, u_i)), log_M_r=f32(self.log_weight(t_r, u_r)),
        )

    def sample_db_points(self, n: int, t_dim: int = 2, t_spread: float = 1.5,
                         u_range: tuple[float, float] = (0.1, 3.5), rng=None) -> dict:
        """Arbitrary DB points: t ~ N(0, (t_spread sigma)^2), u ~ U(u_range * sigma), sigma = sqrt(kT/m).

        DB holds for every pair, so the points do not need to come from the true kernel.
        """
        rng = np.random.default_rng(rng)
        sig = np.sqrt(self.thermal_variance)
        t = lambda: rng.normal(0.0, t_spread * sig, (n, t_dim))
        u = lambda: rng.uniform(u_range[0] * sig, u_range[1] * sig, n)
        t_i, u_i, t_r, u_r = t(), u(), t(), u()
        return self.make_db_data(t_i, u_i, t_r, u_r)

    # --- evaluation -------------------------------------------------------------------------------
    def db_sides(self, model, t_i, u_i, t_r, u_r):
        """Left and right side of the DB equation (log form, physical space) of a model with log_prob."""
        d = self.make_db_data(t_i, u_i, t_r, u_r)
        lhs = model.log_prob(d["v_r"], d["v_i"]).numpy() + d["log_det_jacobian_fwd"] + d["log_M_i"]
        rhs = model.log_prob(d["v_r_bwd"], d["v_i_bwd"]).numpy() + d["log_det_jacobian_bwd"] + d["log_M_r"]
        return lhs, rhs

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"loss:db:mass": self.mass, "loss:db:kT": self.kT})
        return config
