"""
    SplineFlow and cSplineFlow:
    Normalizing Flow models for generating samples from complex conditional and unconditional distributions.

    Neural spline flow (Durkan et al., 2019, "Neural Spline Flows") with rational-quadratic spline coupling
    layers. Each transformed dimension is mapped by a monotonic rational-quadratic spline with n_bins bins on
    the interval [-tail_bound, tail_bound]; outside of it the transformation is the identity (linear tails).
    The spline parameters (bin widths, bin heights and knot derivatives) are predicted by a network from the
    kept dimensions and the (optional) condition.

    Compared to RealNVP, a single coupling layer is far more expressive (it can create multimodal marginals),
    so fewer coupling layers are usually needed. The data should lie (mostly) inside [-tail_bound, tail_bound],
    e.g. by standardizing it, because the flow is the identity outside.
"""

import keras
import numpy as np
import tensorflow as tf

from ..base import GenerativeModel
from ..distributions import StandardNormal
from ..mlp import make_mlp
from ..realnvp.realnvp import default_masks
from ..registry import register_model

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def _default_hidden_layers():
    return [{"neurons": n, "activation": "relu"} for n in [32, 32]]


# --- rational-quadratic spline ---------------------------------------------------------------

def _knots(unnormalized, tail_bound, min_size):
    """softmax -> bin sizes with a minimum size -> knot positions on [-B, B].
    Returns (knots [..., K+1], sizes [..., K])."""
    n_bins = unnormalized.shape[-1]
    sizes = min_size + (1.0 - min_size * n_bins) * tf.nn.softmax(unnormalized, axis=-1)
    inner = 2.0 * tail_bound * tf.cumsum(sizes, axis=-1)[..., :-1] - tail_bound
    edge = tf.ones_like(sizes[..., :1]) * tail_bound
    knots = tf.concat([-edge, inner, edge], axis=-1)  # exact end points
    return knots, knots[..., 1:] - knots[..., :-1]


def _gather_bin(params, idx):
    """params [N, D, K'], idx [N, D] -> params[n, d, idx[n, d]]"""
    return tf.gather(params, idx[..., None], batch_dims=2)[..., 0]


def rational_quadratic_spline(x, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
                              inverse: bool = False, tail_bound: float = 5.0,
                              min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
                              min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative: float = DEFAULT_MIN_DERIVATIVE):
    """Elementwise monotonic rational-quadratic spline with identity tails.

    x                        : [N, D]
    unnormalized_widths      : [N, D, K]    (softmax -> bin widths)
    unnormalized_heights     : [N, D, K]    (softmax -> bin heights)
    unnormalized_derivatives : [N, D, K-1]  (softplus -> derivatives at the interior knots; the derivative at
                                             +-tail_bound is 1, so the spline joins the identity tails smoothly.
                                             An input of 0 gives a derivative of 1.)
    Returns (y, log|dy/dx|) of shapes [N, D]; for inverse=True the inverse map and log|dx/dy|.
    """
    n_bins = unnormalized_widths.shape[-1]
    if min_bin_width * n_bins > 1.0 or min_bin_height * n_bins > 1.0:
        raise ValueError("min_bin_width / min_bin_height too large for the number of bins.")
    tail_bound = float(tail_bound)

    x = tf.convert_to_tensor(x, tf.float32)
    inside = tf.logical_and(x >= -tail_bound, x <= tail_bound)
    # evaluate the spline on clamped inputs, so the branch of tf.where that is not used stays finite
    xc = tf.clip_by_value(x, -tail_bound, tail_bound)

    cumwidths, widths = _knots(unnormalized_widths, tail_bound, min_bin_width)
    cumheights, heights = _knots(unnormalized_heights, tail_bound, min_bin_height)

    shift = float(np.log(np.expm1(1.0 - min_derivative)))  # softplus(0 + shift) + min_derivative = 1
    inner_d = min_derivative + tf.nn.softplus(unnormalized_derivatives + shift)
    one = tf.ones_like(inner_d[..., :1])
    derivatives = tf.concat([one, inner_d, one], axis=-1)  # [N, D, K+1]

    # bin index of every input
    locations = cumheights if inverse else cumwidths
    idx = tf.reduce_sum(tf.cast(xc[..., None] >= locations, tf.int32), axis=-1) - 1
    idx = tf.clip_by_value(idx, 0, n_bins - 1)

    x_k, w_k = _gather_bin(cumwidths, idx), _gather_bin(widths, idx)
    y_k, h_k = _gather_bin(cumheights, idx), _gather_bin(heights, idx)
    d_k, d_k1 = _gather_bin(derivatives, idx), _gather_bin(derivatives, idx + 1)
    s_k = h_k / w_k                                    # slope of the bin
    d_sum = d_k + d_k1 - 2.0 * s_k

    if not inverse:
        theta = (xc - x_k) / w_k
        theta_1mt = theta * (1.0 - theta)
        numerator = h_k * (s_k * tf.square(theta) + d_k * theta_1mt)
        denominator = s_k + d_sum * theta_1mt
        out = y_k + numerator / denominator
    else:
        dy = xc - y_k
        a = h_k * (s_k - d_k) + dy * d_sum
        b = h_k * d_k - dy * d_sum
        c = -s_k * dy
        disc = tf.maximum(tf.square(b) - 4.0 * a * c, 0.0)
        theta = (2.0 * c) / (-b - tf.sqrt(disc))      # numerically stable root in [0, 1]
        theta_1mt = theta * (1.0 - theta)
        denominator = s_k + d_sum * theta_1mt
        out = x_k + theta * w_k

    deriv_numerator = tf.square(s_k) * (d_k1 * tf.square(theta) + 2.0 * s_k * theta_1mt
                                        + d_k * tf.square(1.0 - theta))
    log_det = tf.math.log(deriv_numerator) - 2.0 * tf.math.log(denominator)  # log dy/dx
    if inverse:
        log_det = -log_det

    return tf.where(inside, out, x), tf.where(inside, log_det, tf.zeros_like(log_det))


# --- networks and coupling layer ------------------------------------------------------------

class SplineNetwork(keras.layers.Layer):
    """Parameter network of a spline coupling layer.

    in_dim   : number of kept (masked) features
    cond_dim : conditioning dimension (0 for unconditional models)
    out_dim  : number of transformed features
    n_bins   : number of spline bins K -> 3K - 1 parameters per transformed feature
    """

    def __init__(self, in_dim, cond_dim, out_dim, n_bins: int = 8,
                 hidden_layers: list[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.in_dim, self.cond_dim, self.out_dim = in_dim, cond_dim, out_dim
        self.n_bins = int(n_bins)
        if hidden_layers is None:
            hidden_layers = _default_hidden_layers()
        n_params = 3 * self.n_bins - 1
        self.net = make_mlp(hidden_layers=hidden_layers,
                            output_layers=[{"neurons": out_dim * n_params, "activation": "linear"}])

    def _condition(self, cond):
        return cond

    def call(self, x, cond=None):
        """Returns unnormalized (widths [N, D, K], heights [N, D, K], derivatives [N, D, K-1])."""
        h = GenerativeModel._concat(x, self._condition(cond))
        params = tf.reshape(self.net(h), [-1, self.out_dim, 3 * self.n_bins - 1])
        return tf.split(params, [self.n_bins, self.n_bins, self.n_bins - 1], axis=-1)


class SymmetricSplineNetwork(SplineNetwork):
    """Spline network that only sees |cond|, i.e. it is invariant to the sign of the conditioning
    components."""

    def _condition(self, cond):
        if cond is None:
            raise ValueError("SymmetricSplineNetwork requires a conditional input.")
        return tf.abs(cond)


class SplineCouplingLayer(keras.layers.Layer):
    """A single rational-quadratic spline coupling layer: y_off = RQS(y_off; theta(y_on, cond))."""

    def __init__(self, x_dim: int, cond_dim: int, mask, n_bins: int = 8,
                 hidden_layers: list[dict] = None, tail_bound: float = 5.0, symmetric: bool = False,
                 min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative: float = DEFAULT_MIN_DERIVATIVE, **kwargs):
        super().__init__(**kwargs)
        mask = np.asarray(mask, dtype=np.int32)
        if mask.shape != (x_dim,):
            raise ValueError(f"mask must have length x_dim={x_dim}, got shape {mask.shape}.")
        idx_on = np.where(mask == 1)[0].astype(np.int32)   # dimensions that are kept
        idx_off = np.where(mask == 0)[0].astype(np.int32)  # dimensions that are transformed
        if len(idx_on) == 0 or len(idx_off) == 0:
            raise ValueError(f"mask {mask.tolist()} must keep and transform at least one dimension.")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if tail_bound <= 0:
            raise ValueError("tail_bound must be positive.")
        self.x_dim = x_dim
        self.idx_on, self.idx_off = idx_on, idx_off
        self.tail_bound = float(tail_bound)
        self.min_bin_width, self.min_bin_height = float(min_bin_width), float(min_bin_height)
        self.min_derivative = float(min_derivative)

        eye = np.eye(x_dim, dtype=np.float32)
        self.P_on = eye[idx_on, :]    # [in_dim,  x_dim]
        self.P_off = eye[idx_off, :]  # [out_dim, x_dim]

        net_cls = SymmetricSplineNetwork if symmetric else SplineNetwork
        self.spline_net = net_cls(len(idx_on), cond_dim, len(idx_off), n_bins, hidden_layers)

    def call(self, y, cond=None, inverse=False):
        """
        y:    [batch, x_dim]     current state
        cond: [batch, cond_dim]  conditioning input (None for unconditional)
        """
        y_on = tf.gather(y, self.idx_on, axis=1)    # [batch, in_dim]  -> kept
        y_off = tf.gather(y, self.idx_off, axis=1)  # [batch, out_dim] -> transformed

        widths, heights, derivatives = self.spline_net(y_on, cond)
        y_off_out, log_det = rational_quadratic_spline(
            y_off, widths, heights, derivatives, inverse=inverse, tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width, min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative)

        y_on_full = tf.matmul(y_on, tf.constant(self.P_on))
        y_off_full = tf.matmul(y_off_out, tf.constant(self.P_off))
        return y_on_full + y_off_full, tf.reduce_sum(log_det, axis=1)


# --- models ---------------------------------------------------------------------------------

@register_model("splineflow")
class SplineFlow(GenerativeModel):
    """Rational-quadratic neural spline flow. With cond_dim > 0 every spline network additionally
    receives the conditioning input (-> conditional flow).

    hidden_layers  : hidden layers of each spline network (list of {"neurons", "activation"} dicts)
    n_coupling     : number of coupling layers
    n_bins         : number of spline bins per transformed dimension
    tail_bound     : the splines act on [-tail_bound, tail_bound], identity outside
    mask_patterns  : list of 0/1 lists of length x_dim (cycled over the coupling layers)
    symmetric      : spline networks only see |cond|; needs cond_dim > 0
    min_bin_width, min_bin_height, min_derivative : numerical lower bounds of the spline parameters
    """

    loss_names = ("loss",)

    def __init__(self, x_dim: int, cond_dim: int = 0, hidden_layers: list[dict] = None,
                 n_coupling: int = 4, n_bins: int = 8, tail_bound: float = 5.0, mask_patterns=None,
                 symmetric: bool = False, min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative: float = DEFAULT_MIN_DERIVATIVE, **kwargs):
        super().__init__(x_dim, cond_dim, **kwargs)
        if x_dim < 2:
            raise ValueError("SplineFlow coupling layers need x_dim >= 2.")
        if symmetric and cond_dim < 1:
            raise ValueError("symmetric=True requires cond_dim > 0.")

        if mask_patterns is None:
            mask_patterns = default_masks(x_dim, n_coupling)

        self.n_bins, self.tail_bound = int(n_bins), float(tail_bound)
        self.coupling = [
            SplineCouplingLayer(x_dim, cond_dim, mask_patterns[i % len(mask_patterns)], n_bins,
                                hidden_layers, tail_bound, symmetric,
                                min_bin_width, min_bin_height, min_derivative)
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

    def compute_losses(self, x, cond=None, training=True):
        return {"loss": -tf.reduce_mean(self.log_prob(x, cond))}

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


@register_model("csplineflow")
class cSplineFlow(SplineFlow):
    """Conditional rational-quadratic neural spline flow: every spline network additionally
    receives the conditioning input."""

    def __init__(self, x_dim: int, cond_dim: int, **kwargs):
        if cond_dim <= 0:
            raise ValueError("Conditional dimension must be greater than 0 for cSplineFlow.")
        super().__init__(x_dim, cond_dim, **kwargs)
