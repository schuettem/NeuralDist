"""
    RealNVP and cRealNVP:
    Normalizing Flow models for generating samples from complex conditional and unconditional distributions.
"""
import keras
import numpy as np
import tensorflow as tf

from ..distributions import StandardNormal
from ..base import GenerativeModel
from ..mlp import make_mlp
from ..registry import register_model

def _default_hidden_layers():
    return [{"neurons": n, "activation": "relu"} for n in [32, 32]]


class STNetwork(keras.layers.Layer):
    """Scale and translate network of a RealNVP coupling layer.

    in_dim   : number of kept (masked) features
    cond_dim : conditioning dimension (0 for unconditional models)
    out_dim  : number of transformed features
    """

    def __init__(self, in_dim, cond_dim, out_dim,
                 hidden_layers: list[dict] = None,
                 scale_clamp=1.0, **kwargs):
        super().__init__(**kwargs)
        self.in_dim, self.cond_dim, self.out_dim = in_dim, cond_dim, out_dim
        self.scale_clamp = float(scale_clamp)

        if hidden_layers is None:
            hidden_layers = _default_hidden_layers()

        # make_mlp stacks its output layers sequentially, so a single linear output layer with
        # 2 * out_dim units is used and split into [s, t] in call().
        output_layers = [{"neurons": 2 * out_dim, "activation": "linear"}]

        self.stnet = make_mlp(
            hidden_layers=hidden_layers,
            output_layers=output_layers
        )

    def _condition(self, cond):
        return cond

    def call(self, x, cond=None):
        h = GenerativeModel._concat(x, self._condition(cond))  # concatenate x and conditional input
        s, t = tf.split(self.stnet(h), 2, axis=-1)
        return tf.tanh(s) * self.scale_clamp, t  # clamp log-scale


class SymmetricSTNetwork(STNetwork):
    """ST-network that only sees |cond|, i.e. it is invariant to the sign of the conditioning
    components. This enforces that forward and backward transformations share the same logic."""

    def _condition(self, cond):
        if cond is None:
            raise ValueError("SymmetricSTNetwork requires a conditional input.")
        return tf.abs(cond)


class CouplingLayer(keras.layers.Layer):
    """A single affine coupling layer: y_off = y_off * exp(s) + t, with [s, t] = ST(y_on, cond)."""

    def __init__(self, x_dim: int, cond_dim: int, mask,
                 hidden_layers: list[dict] = None,
                 scale_clamp: float = 1.0, symmetric: bool = False, **kwargs):
        super().__init__(**kwargs)
        mask = np.asarray(mask, dtype=np.int32)
        if mask.shape != (x_dim,):
            raise ValueError(f"mask must have length x_dim={x_dim}, got shape {mask.shape}.")
        # indices of kept vs. to-be-transformed dims
        idx_on = np.where(mask == 1)[0].astype(np.int32)   # dimensions that are kept
        idx_off = np.where(mask == 0)[0].astype(np.int32)  # dimensions that are transformed
        if len(idx_on) == 0 or len(idx_off) == 0:
            raise ValueError(f"mask {mask.tolist()} must keep and transform at least one dimension.")
        self.x_dim = x_dim
        self.idx_on, self.idx_off = idx_on, idx_off

        # precompute one-hot projection matrices for merging
        eye = np.eye(x_dim, dtype=np.float32)
        self.P_on = eye[idx_on, :]    # [in_dim,  x_dim]
        self.P_off = eye[idx_off, :]  # [out_dim, x_dim]

        # the scale/translate net: SymmetricSTNetwork or STNetwork
        st_cls = SymmetricSTNetwork if symmetric else STNetwork
        self.stnet = st_cls(len(idx_on), cond_dim, len(idx_off),
                            hidden_layers, scale_clamp)

    def call(self, y, cond=None, inverse=False):
        """
        y:    [batch, x_dim]     current state
        cond: [batch, cond_dim]  conditioning input (None for unconditional)
        """
        # 1) split
        y_on = tf.gather(y, self.idx_on, axis=1)    # [batch, in_dim]  -> kept
        y_off = tf.gather(y, self.idx_off, axis=1)  # [batch, out_dim] -> transformed

        # 2) compute scale/translate
        s, t = self.stnet(y_on, cond)  # both [batch, out_dim]

        # 3) forward vs inverse
        if not inverse:
            y_off_out = y_off * tf.exp(s) + t
            log_det = tf.reduce_sum(s, axis=1)   # [batch]
        else:
            y_off_out = (y_off - t) * tf.exp(-s)
            log_det = -tf.reduce_sum(s, axis=1)

        # 4) merge back into full dim
        y_on_full = tf.matmul(y_on, tf.constant(self.P_on))          # [batch, x_dim]
        y_off_full = tf.matmul(y_off_out, tf.constant(self.P_off))   # [batch, x_dim]
        return y_on_full + y_off_full, log_det


def default_masks(x_dim: int, n_coupling: int):
    """Mask patterns (1 = kept, 0 = transformed). For x_dim == 3 the patterns of the original
    velocity model are used; otherwise alternating checkerboard masks."""
    if x_dim == 3:
        patterns = [[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    else:
        patterns = [((np.arange(x_dim) + i) % 2 == 0).astype(int).tolist() for i in range(2)]
    return [patterns[i % len(patterns)] for i in range(n_coupling)]


@register_model("realnvp")
class RealNVP(GenerativeModel):
    """RealNVP normalizing flow. With cond_dim > 0 every coupling network additionally receives
    the conditioning input (-> conditional flow).

    hidden_dim_list / hidden_activation_list : hidden layers of each ST-network
    n_coupling       : number of coupling layers
    scale_clamp      : multiplier on the tanh log-scale; float or list of length n_coupling
    mask_patterns    : list of 0/1 lists of length x_dim (cycled over the coupling layers)
    symmetric        : use SymmetricSTNetwork (ST-nets only see |cond|); needs cond_dim > 0
    """

    loss_names = ("loss",)
    supports_weights = True

    def __init__(self, x_dim: int, cond_dim: int = 0, hidden_layers: list[dict] = None, n_coupling: int = 6, scale_clamp=1.0,
                 mask_patterns=None, symmetric: bool = False, **kwargs):
        super().__init__(x_dim, cond_dim, **kwargs)
        if x_dim < 2:
            raise ValueError("RealNVP coupling layers need x_dim >= 2.")
        if symmetric and cond_dim < 1:
            raise ValueError("symmetric=True requires cond_dim > 0.")

        # clamping of the scale output of the ST-networks
        if isinstance(scale_clamp, (float, int)):
            scale_clamp = [float(scale_clamp)] * n_coupling
        elif len(scale_clamp) != n_coupling:
            raise ValueError(f"Length of scale_clamp ({len(scale_clamp)}) must match "
                             f"n_coupling ({n_coupling}).")

        if mask_patterns is None:
            mask_patterns = default_masks(x_dim, n_coupling)

        self.coupling = [
            CouplingLayer(x_dim, cond_dim, mask_patterns[i % len(mask_patterns)],
                          hidden_layers, scale_clamp[i], symmetric)
            for i in range(n_coupling)
        ]
        self.prior = StandardNormal(x_dim)

    def forward(self, x, cond=None):
        """data -> latent: returns z and log|det J|"""
        z, log_det_total = x, tf.zeros(tf.shape(x)[:1])
        for layer in self.coupling:
            z, log_det = layer(z, cond, inverse=False)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, z, cond=None):
        """latent -> data"""
        x = z
        for layer in reversed(self.coupling):
            x, _ = layer(x, cond, inverse=True)
        return x

    def log_prob(self, x, cond=None):
        x, cond = self._check_cond(x, cond)
        z, log_det = self.forward(x, cond)
        return self.prior.log_prob(z) + log_det  # [batch]

    def compute_losses(self, x, cond=None, training=True, weight=None):
        """Negative log-likelihood. Without weight: mean over samples x ~ p.
        With weight w_i ∝ p(x_i) / r(x_i) (points x_i ~ r, e.g. a grid): the weighted mean
        mean(w_i * -log q(x_i)), an estimate of the cross entropy H(p, q) = KL(p || q) + H(p)
        when the weights have mean 1 over the data set (see training.pdf_weights)."""
        nll = -self.log_prob(x, cond)
        if weight is None:
            return {"loss": tf.reduce_mean(nll)}
        return {"loss": tf.reduce_mean(tf.cast(weight, tf.float32) * nll)}

    def sample(self, n_samples, cond=None, z=None):
        """cond: [B, cond_dim] -> [B, n_samples, x_dim]; without cond -> [n_samples, x_dim].
        Optional z: latent samples of shape [B * n_samples, x_dim]."""
        c, lead, rows = self._expand_cond(n_samples, cond)
        if z is None:
            z = self.prior.sample(rows)
        else:
            z = tf.cast(z, tf.float32)
            tf.debugging.assert_equal(tf.shape(z)[0], rows, message="z must have B*n_samples rows.")
        return self._to_lead_shape(self.inverse(z, c), lead)


@register_model("crealnvp")
class cRealNVP(RealNVP):
    """
    Conditional RealNVP normalizing flow. With cond_dim > 0 every coupling network additionally
    receives the conditioning input (-> conditional flow).
    """
    def __init__(self, x_dim: int, cond_dim: int, **kwargs):
        if cond_dim <= 0:
            raise ValueError("Conditional dimension must be greater than 0 for cRealNVP.")
        super().__init__(x_dim, cond_dim, **kwargs)
