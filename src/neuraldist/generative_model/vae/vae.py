"""
VAE and cVAE:
    encoder: MLP that outputs mean and log variance of latent distribution:
        - encoder_hidden_layers: list of dicts specifying hidden layer configurations dict{neurons: int, activation: str}
        - encoder_output_layers (fixed): neurons: 2 * latent_dim, activation: 'linear' (for mean and log variance)
    decoder: MLP that outputs reconstructed data:
        - decoder_hidden_layers: list of dicts specifying hidden layer configurations dict{neurons: int, activation: str}
        - decoder_output_layers: list of dicts specifying output layer configurations dict{neurons: int, activation: str}
    latent_dim: int specifying the dimension of the latent space
    kl_weight: float specifying the weight of the KL divergence term in the loss
"""

import tensorflow as tf

from ..base import GenerativeModel
from ..mlp import make_mlp
from ..registry import register_model

def _default_hidden_layers():
    return [{"neurons": n, "activation": "relu"} for n in [64, 32, 16]]

@register_model("vae")
class VAE(GenerativeModel):
    loss_names = ["loss", "recon", "kl"]

    def __init__(self, x_dim: int, cond_dim: int = 0, latent_dim: int = 3, kl_weight: float = 1.0,
                 encoder_hidden_layers: list[dict] = None, decoder_hidden_layers: list[dict] = None,
                 encoder_output_layers: list[dict] = None, decoder_output_layers: list[dict] = None, **kwargs):
        super().__init__(x_dim, cond_dim, **kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Default hidden layers if not provided
        if encoder_hidden_layers is None:
            encoder_hidden_layers = _default_hidden_layers()
        if decoder_hidden_layers is None:
            decoder_hidden_layers = [dict(layer) for layer in reversed(_default_hidden_layers())]

        # Default output layers if not provided
        if decoder_output_layers is None:
            decoder_output_layers = [{"neurons": x_dim, "activation": "linear"}]
        elif decoder_output_layers[-1]["neurons"] != x_dim:
            raise ValueError(f"Decoder output layer must have {x_dim} neurons, got {decoder_output_layers[-1]['neurons']}.")

        if encoder_output_layers is None:
            encoder_output_layers = [{"neurons": 2 * latent_dim, "activation": "linear"}]
        elif encoder_output_layers[-1]["neurons"] != 2 * latent_dim:
            raise ValueError(f"Encoder output layer must have {2 * latent_dim} neurons, got {encoder_output_layers[-1]['neurons']}.")

        # Create encoder and decoder MLPs
        self.encoder = make_mlp(
            hidden_layers=encoder_hidden_layers,
            output_layers=encoder_output_layers
        )
        self.decoder = make_mlp(
            hidden_layers=decoder_hidden_layers,
            output_layers=decoder_output_layers
        )

    def encode(self, x, cond=None):
        mu, logvar = tf.split(self.encoder(self._concat(x, cond)), num_or_size_splits=2, axis=-1)
        return mu, logvar

    def decode(self, z, cond=None):
        return self.decoder(self._concat(z, cond))

    def compute_losses(self, x, cond=None, training=True):
        mu, logvar = self.encode(x, cond)
        z = mu + tf.random.normal(tf.shape(mu)) * tf.exp(0.5 * logvar)
        recon = tf.reduce_mean(tf.reduce_sum(tf.square(self.decode(z, cond) - x), axis=-1))
        kl = tf.reduce_mean(
            -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1)
        )
        return {"loss": recon + self.kl_weight * kl, "recon": recon, "kl": kl}

    def sample(self, n_samples, cond=None):
        c, lead, rows = self._expand_cond(n_samples, cond)
        z = tf.random.normal(tf.stack([rows, tf.constant(self.latent_dim, tf.int32)]))
        return self._to_lead_shape(self.decode(z, c), lead)

@register_model("cvae")
class cVAE(VAE):
    def __init__(self, x_dim: int, cond_dim: int = 0, **kwargs):
        if cond_dim <= 0:
            raise ValueError("Conditional dimension must be greater than 0 for cVAE.")
        super().__init__(x_dim, cond_dim, **kwargs)