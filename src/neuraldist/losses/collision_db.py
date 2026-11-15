"""
    Detailed-balance (DB) loss for binary particle-particle collisions.

    Physics
    -------
    Two particles (masses m1, m2) collide with pre-collision velocities (v1, v2) and leave with (v1', v2').
    Momentum conservation fixes the centre-of-mass velocity G, so the collision is fully described in the
    relative frame by the relative velocity g = v1 - v2 -> g' = v1' - v2'. The model learns the collision
    kernel q(g' | g) (condition: g, target: g').

    In equilibrium (Maxwellians at temperature kT) the rate of collisions with relative velocity g is
    proportional to
        W(g) = B(|g|) * M_mu(g),    M_mu(g) ~ exp(-mu |g|^2 / (2 kT)),    mu = m1 m2 / (m1 + m2)
    (the G-part of f1(v1) f2(v2) cancels because G is conserved). B(|g|) = |g| sigma_T(|g|) is the
    collision-rate factor; for the VHS model B ~ |g|^(2 - 2 omega) (hard spheres: 1, Maxwell molecules: 0).

    Detailed balance (microscopic reversibility) then reads
        W(g) q(g' | g) = W(g') q(T g | T g')
    with the time-reversal T g = -g. For isotropic interactions T can be dropped (parity), see
    `time_reversal`. Elastic collisions conserve |g| exactly, so a full 3D density q(g' | g) only exists for
    inelastic collisions, e.g. when energy is exchanged with internal degrees of freedom that are themselves
    in equilibrium (they then act as a heat bath, like the wall in the gas-surface case).

    This is the same structure as the gas-surface DB loss (`DBLoss`), with
        v_i -> g,   v_r -> g',   log M_i -> log W(g),   log M_r -> log W(g'),
    so `CollisionDBLoss` produces the same DB data format and works with `cRealNVPDB` and `fit_model`.
"""
from __future__ import annotations

import numpy as np

from .db import DBLoss

_EPS = 1e-12


# ---------------------------------------------------------------------------------------------------
# lab frame <-> relative frame
# ---------------------------------------------------------------------------------------------------
def reduced_mass(m1: float, m2: float) -> float:
    return m1 * m2 / (m1 + m2)


def to_relative_frame(v1, v2, m1: float = 1.0, m2: float = 1.0):
    """Lab-frame velocities (N, 3) -> (G, g): centre-of-mass velocity and relative velocity g = v1 - v2."""
    v1, v2 = np.asarray(v1, dtype="float64"), np.asarray(v2, dtype="float64")
    G = (m1 * v1 + m2 * v2) / (m1 + m2)
    return G, v1 - v2


def from_relative_frame(G, g, m1: float = 1.0, m2: float = 1.0):
    """(G, g) -> lab-frame velocities (v1, v2). Use it to turn sampled post-collision g' into v1', v2'."""
    G, g = np.asarray(G, dtype="float64"), np.asarray(g, dtype="float64")
    M = m1 + m2
    return G + (m2 / M) * g, G - (m1 / M) * g


# ---------------------------------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------------------------------
class CollisionDBLoss(DBLoss):
    """
    DB loss for a binary-collision kernel q(g' | g) in the relative-velocity frame.

    reduced_mass    : mu = m1 m2 / (m1 + m2) (identical particles with m = 1: 0.5)
    kT              : temperature of the equilibrium the kernel must be in detailed balance with
    rate_exponent   : a in B(|g|) ~ |g|^a (collision-rate factor |g| sigma_T); VHS: a = 2 - 2 omega,
                      hard spheres a = 1, Maxwell molecules a = 0
    time_reversal   : True: reversed pair is (-g' -> -g); False: (g' -> g) (isotropic interactions)
    epoch_weighting : float or weighting object (ConstantWeight, LinearRampWeight, ...), see DBLoss
    enabled         : switch the loss on/off without removing it from compile()

    The residual itself is the one of DBLoss:
        [log q(g'|g) + log W(g)] - [log q(T g | T g') + log W(g')]
    `make_db_data` / `sample_db_points` build the DB batch dict (keys DB_KEYS of training.make_db_dataset).
    """

    kind = "collision"

    def __init__(self, reduced_mass: float = 0.5, kT: float = 1.0, rate_exponent: float = 1.0,
                 time_reversal: bool = True, epoch_weighting=1.0, enabled: bool = True):
        super().__init__(epoch_weighting=epoch_weighting, enabled=enabled)
        if reduced_mass <= 0 or kT <= 0:
            raise ValueError("reduced_mass and kT must be positive.")
        self.reduced_mass = float(reduced_mass)
        self.kT = float(kT)
        self.rate_exponent = float(rate_exponent)
        self.time_reversal = bool(time_reversal)

    # --- equilibrium collision-rate density -------------------------------------------------------
    @property
    def thermal_variance(self) -> float:
        """Variance kT / mu of each component of g in equilibrium."""
        return self.kT / self.reduced_mass

    def log_maxwellian(self, g) -> np.ndarray:
        """log M_mu(g): normalized Maxwellian of the relative velocity, g of shape (N, d)."""
        g = np.asarray(g, dtype="float64")
        var, d = self.thermal_variance, g.shape[-1]
        return -0.5 * np.sum(g**2, axis=-1) / var - 0.5 * d * np.log(2 * np.pi * var)

    def log_rate_factor(self, g) -> np.ndarray:
        """log B(|g|) = a log|g| (unnormalized; constants cancel in the DB residual)."""
        if self.rate_exponent == 0.0:
            return np.zeros(np.shape(g)[:-1])
        speed = np.linalg.norm(np.asarray(g, dtype="float64"), axis=-1)
        return self.rate_exponent * np.log(np.maximum(speed, _EPS))

    def log_weight(self, g) -> np.ndarray:
        """log W(g) = log B(|g|) + log M_mu(g): log density of colliding pairs in equilibrium."""
        return self.log_rate_factor(g) + self.log_maxwellian(g)

    def reverse(self, g) -> np.ndarray:
        """Time-reversal operator T applied to g (identity if time_reversal=False)."""
        g = np.asarray(g, dtype="float64")
        return -g if self.time_reversal else g

    # --- DB data ----------------------------------------------------------------------------------
    def make_db_data(self, g_pre, g_post) -> dict:
        """Pairs (g -> g'), arrays (N, d) -> dict with the DB keys expected by DBLoss / fit_model(db_data=...).

        The flow works directly on g (no abstract variables), so the log-Jacobians are 0.
        """
        g_pre, g_post = np.asarray(g_pre, dtype="float64"), np.asarray(g_post, dtype="float64")
        if g_pre.ndim != 2 or g_pre.shape != g_post.shape:
            raise ValueError(f"g_pre and g_post must have the same shape (N, d), got {g_pre.shape} "
                             f"and {g_post.shape}.")
        f32 = lambda a: np.asarray(a, dtype="float32")
        zeros = np.zeros(len(g_pre), dtype="float32")
        return dict(
            v_i=f32(g_pre), v_r=f32(g_post),
            v_i_bwd=f32(self.reverse(g_post)), v_r_bwd=f32(self.reverse(g_pre)),
            log_det_jacobian_fwd=zeros, log_det_jacobian_bwd=zeros.copy(),
            log_M_i=f32(self.log_weight(g_pre)), log_M_r=f32(self.log_weight(g_post)),
        )

    def make_db_data_lab(self, v1, v2, v1_post, v2_post, m1: float = 1.0, m2: float = 1.0) -> dict:
        """Same as make_db_data, but from lab-frame pre/post-collision velocities of both particles."""
        _, g_pre = to_relative_frame(v1, v2, m1, m2)
        _, g_post = to_relative_frame(v1_post, v2_post, m1, m2)
        return self.make_db_data(g_pre, g_post)

    def sample_db_points(self, n: int, dim: int = 3, spread: float = 1.5, rng=None) -> dict:
        """Arbitrary DB points: g and g' drawn independently from N(0, spread^2 * kT/mu * I).

        DB holds for every pair, so the points do not need to come from the true kernel; `spread` > 1
        also covers the tails of the equilibrium distribution.
        """
        rng = np.random.default_rng(rng)
        std = spread * np.sqrt(self.thermal_variance)
        return self.make_db_data(rng.normal(0.0, std, (n, dim)), rng.normal(0.0, std, (n, dim)))

    # --- evaluation -------------------------------------------------------------------------------
    def db_sides(self, model, g_pre, g_post):
        """Left and right side of the DB equation (log form) of a model with log_prob(x, cond)."""
        d = self.make_db_data(g_pre, g_post)
        lhs = model.log_prob(d["v_r"], d["v_i"]).numpy() + d["log_M_i"]
        rhs = model.log_prob(d["v_r_bwd"], d["v_i_bwd"]).numpy() + d["log_M_r"]
        return lhs, rhs

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"loss:db:reduced_mass": self.reduced_mass,
                       "loss:db:kT": self.kT,
                       "loss:db:rate_exponent": self.rate_exponent,
                       "loss:db:time_reversal": self.time_reversal})
        return config
