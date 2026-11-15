"""
    cRealNVP with additional backward NLL and detailed-balance (DB) loss.
    Physics-agnostic: works with every DBLoss subclass (SurfaceDBLoss, CollisionDBLoss).

    Data format (key names are generic: v_i = condition, v_r = target of the kernel):
        data["main"] : v_i, v_r, v_i_bwd, v_r_bwd  -> NLL terms
        data["db"]   : see DBLoss                  -> DB term (only if db_loss is used)
"""
import tensorflow as tf

from ...losses.db import DBLoss
from ..registry import register_model
from .realnvp import cRealNVP


@register_model("crealnvp_db")
class cRealNVPDB(cRealNVP):
    loss_names = ("loss", "nll_fwd", "nll_bwd", "db_loss", "db_acc")

    def __init__(self, x_dim: int, cond_dim: int, **kwargs):
        super().__init__(x_dim, cond_dim, **kwargs)
        self.nll_fwd, self.nll_bwd = True, False
        self.db_loss, self.use_db_loss = None, False
        self.db_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="db_weighting")

    def compile(self, optimizer, nll_fwd: bool = True, nll_bwd: bool = False, db_loss=None, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self.nll_fwd, self.nll_bwd = nll_fwd, nll_bwd
        if isinstance(db_loss, DBLoss) and db_loss.enabled:
            self.db_loss, self.use_db_loss = db_loss, True
            self.db_weight.assign(db_loss.initial_weight())
        else:
            self.db_loss, self.use_db_loss = None, False
        if not (nll_fwd or nll_bwd or self.use_db_loss):
            raise ValueError("At least one of nll_fwd, nll_bwd or db_loss must be active.")

    def _compute(self, data, training: bool):
        main, zero = data["main"], tf.constant(0.0)
        loss, nll_fwd, nll_bwd, db_l, acc = zero, zero, zero, zero, zero

        if self.nll_fwd:  # NLL on all data (MD + possible EQ)
            nll_fwd = -tf.reduce_mean(self.log_prob(main["v_r"], main["v_i"]))
            loss += nll_fwd
        if self.nll_bwd:  # NLL of the reversed pairs
            nll_bwd = -tf.reduce_mean(self.log_prob(main["v_r_bwd"], main["v_i_bwd"]))
            loss += nll_bwd
        if self.use_db_loss:
            # detailed balance loss on physical space
            if "db" in data:
                db = data["db"]
                log_q_fwd = self.log_prob(db["v_r"], db["v_i"])
                log_q_bwd = self.log_prob(db["v_r_bwd"], db["v_i_bwd"])
                db_l, acc = self.db_loss(log_q_fwd, log_q_bwd, db)
                loss += self.db_weight * db_l
            elif training:
                raise KeyError('db_loss is active, but the training data has no "db" entry.')
            # validation data without "db": DB terms are reported as 0
        return {"loss": loss, "nll_fwd": nll_fwd, "nll_bwd": nll_bwd, "db_loss": db_l, "db_acc": acc}

    def train_step(self, data):
        with tf.GradientTape() as tape:
            losses = self._compute(data, training=True)
        grads = tape.gradient(losses["loss"], self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        logs = self._update_metrics(losses)
        logs["db_weight"] = tf.identity(self.db_weight)
        return logs

    def test_step(self, data):
        return self._update_metrics(self._compute(data, training=False))