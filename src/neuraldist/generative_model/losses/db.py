"""
    Detailed-balance (DB) loss for conditional flows q(v_r | v_i).

    For a reflection kernel that is in detailed balance with respect to the incident flux density M,
        M(v_i) q(v_r | v_i) = M(-v_r) q(-v_i | -v_r)    (all in physical space)
    holds. The loss penalizes the squared log-ratio of both sides.
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
    epoch_weighting : float or object with `get_weights(n_epochs) -> array` (e.g. LinearRampWeight)
    enabled         : switch the loss on/off without removing it from compile()

    Expected keys of the DB batch `db` (all [batch] unless noted):
        v_i, v_r, v_i_bwd, v_r_bwd   : forward pair (v_i -> v_r) and reversed pair (v_r -> v_i)
        log_det_jacobian_fwd/bwd     : log|det J| of the map abstract -> physical space
        log_M_i, log_M_r             : log density of the incident / reflected velocity
    """

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
        """log_q_fwd = log q(v_r | v_i), log_q_bwd = log q(v_i | v_r), both in abstract space.
        Returns (loss, mean acceptance probability)."""
        log_q_fwd_phys = log_q_fwd + db["log_det_jacobian_fwd"]
        log_q_bwd_phys = log_q_bwd + db["log_det_jacobian_bwd"]
        diff = (log_q_fwd_phys + db["log_M_i_flux"]) - (log_q_bwd_phys + db["log_M_r_flux"])
        loss = tf.reduce_mean(tf.square(diff))
        mean_acc = tf.reduce_mean(tf.exp(-tf.abs(diff)))  # mean MH acceptance probability
        return loss, mean_acc

    def get_config(self) -> dict:
        config = {"loss:db": self.enabled}
        config.update(self.epoch_weighting.get_config())
        return config


class DBWeightScheduler(keras.callbacks.Callback):
    """Sets the DB-loss weight at the start of every epoch from DBLoss.epoch_weighting."""

    def on_epoch_begin(self, epoch, logs=None):
        if getattr(self.model, "use_db_loss", False):
            self.model.db_weight.assign(self.model.db_loss.weight_for_epoch(epoch))