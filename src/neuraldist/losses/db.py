"""
    Generic detailed-balance (DB) loss for conditional flows q(z' | z) and the DB-weight schedules.

    A transition kernel that is in detailed balance with respect to an equilibrium weight W satisfies
        W(z) q(z' | z) = W(z') q(T z' | T z)    (physical space, T: time reversal)
    The loss penalizes the squared log-ratio of both sides. It is physics-agnostic: the physics (W, T,
    abstract variables) enters only through the DB batch. Physics-specific subclasses build that batch:
        SurfaceDBLoss   (surface_db.py)   : gas-surface scattering kernels K(v_i -> v_r)
        CollisionDBLoss (collision_db.py) : particle-particle collision kernels q(g' | g)
"""
import keras
import numpy as np
import tensorflow as tf


class ConstantWeight:
    """
    DB-loss weight that is the same for every epoch.
    """

    def __init__(self, value: float = 1.0):
        self.value = float(value)

    def get_weights(self, n_epochs: int) -> np.ndarray:
        """Weights for epochs 1..n_epochs (array of length n_epochs)."""
        return np.full(n_epochs, self.value, dtype="float32")

    def get_config(self) -> dict:
        return {"loss:db:weighting:name": "Constant", "loss:db:weighting:value": self.value}


class LinearRampWeight:
    """
    Linear ramp from `start` to `end` over `n_ramp_epochs` epochs, constant afterwards.
    """

    def __init__(self, start: float, end: float, n_ramp_epochs: int):
        self.start, self.end, self.n_ramp_epochs = float(start), float(end), int(n_ramp_epochs)

    def get_weights(self, n_epochs: int) -> np.ndarray:
        frac = np.minimum(np.arange(n_epochs) / max(self.n_ramp_epochs, 1), 1.0)
        return (self.start + (self.end - self.start) * frac).astype("float32")

    def get_config(self) -> dict:
        return {"loss:db:weighting:name": "LinearRamp", "loss:db:weighting:start": self.start,
                "loss:db:weighting:end": self.end, "loss:db:weighting:n_ramp_epochs": self.n_ramp_epochs}


class ExpStepScheduler:
    """
    Exponentially increasing weighting factor
    """

    def __init__(self, base=0.001, interval=20):
        super().__init__()
        self.base = base
        self.interval = interval

    def get_weights(self, epochs):
        increments = (np.arange(1, epochs + 1) - 1) // self.interval
        return self.base * 10**(increments)

    def get_config(self):
        return {"loss:db:weighting:name": "ExpStepScheduler",
                "loss:db:weighting:base": self.base,
                "loss:db:weighting:interval": self.interval}


class DBLoss:
    """
    Generic DB loss; use SurfaceDBLoss or CollisionDBLoss for the physics-specific DB data.

    epoch_weighting : float or object with `get_weights(n_epochs) -> array` (e.g. LinearRampWeight)
    enabled         : switch the loss on/off without removing it from compile()

    Expected keys of the DB batch `db` (all [batch] unless noted). The names stem from the gas-surface
    case but are generic: v_i = condition z, v_r = target z' of the kernel.
        v_i, v_r                 : forward pair (z -> z'), condition / target in abstract variables
        v_i_bwd, v_r_bwd         : reversed pair (T z' -> T z)
        log_det_jacobian_fwd/bwd : log|det J| of the map abstract -> physical space (target variables)
        log_M_i, log_M_r         : log equilibrium weight log W(z) / log W(z')
                                   (surface: flux-weighted wall Maxwellian, collisions: B(|g|) M_mu(g))
    """

    kind = "generic"   # overwritten by the physics-specific subclasses, stored in get_config()

    def __init__(self, epoch_weighting=1.0, enabled: bool = True):
        if not hasattr(epoch_weighting, "get_weights"):
            epoch_weighting = ConstantWeight(epoch_weighting)
        self.epoch_weighting = epoch_weighting
        self.enabled = bool(enabled)

    def initial_weight(self) -> float:
        return float(self.epoch_weighting.get_weights(1)[-1])

    def weight_for_epoch(self, epoch: int) -> float:
        """epoch is 0-based."""
        return float(self.epoch_weighting.get_weights(epoch + 1)[-1])

    def __call__(self, log_q_fwd, log_q_bwd, db):
        """log_q_fwd = log q(v_r | v_i), log_q_bwd = log q(v_r_bwd | v_i_bwd), both in abstract space.
        Returns (loss, mean acceptance probability)."""
        log_q_fwd_phys = log_q_fwd + db["log_det_jacobian_fwd"]
        log_q_bwd_phys = log_q_bwd + db["log_det_jacobian_bwd"]
        diff = (log_q_fwd_phys + db["log_M_i"]) - (log_q_bwd_phys + db["log_M_r"])
        loss = tf.reduce_mean(tf.square(diff))
        mean_acc = tf.reduce_mean(tf.exp(-tf.abs(diff)))  # mean MH acceptance probability
        return loss, mean_acc

    def get_config(self) -> dict:
        config = {"loss:db": self.enabled, "loss:db:type": self.kind}
        config.update(self.epoch_weighting.get_config())
        return config


class DBWeightScheduler(keras.callbacks.Callback):
    """Sets the DB-loss weight at the start of every epoch from DBLoss.epoch_weighting."""

    def on_epoch_begin(self, epoch, logs=None):
        if getattr(self.model, "use_db_loss", False):
            self.model.db_weight.assign(self.model.db_loss.weight_for_epoch(epoch))