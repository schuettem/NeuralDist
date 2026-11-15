import numpy as np
import pytest
import tensorflow as tf

from neuraldist.generative_model.vae import VAE, cVAE

ENC = [{"neurons": 8, "activation": "relu"}]
DEC = [{"neurons": 8, "activation": "relu"}]


@pytest.fixture(autouse=True)
def seed():
    tf.keras.utils.set_random_seed(0)


def test_shapes_and_loss_terms():
    model = VAE(3, latent_dim=2, kl_weight=0.5, encoder_hidden_layers=ENC, decoder_hidden_layers=DEC)
    x = tf.random.normal((16, 3))
    mu, logvar = model.encode(x)
    assert mu.shape == logvar.shape == (16, 2)
    assert model.decode(mu).shape == (16, 3)
    losses = model.compute_losses(x)
    assert set(losses) == {"loss", "recon", "kl"}
    np.testing.assert_allclose(losses["loss"], losses["recon"] + 0.5 * losses["kl"], rtol=1e-6)
    assert float(losses["recon"]) >= 0 and float(losses["kl"]) >= 0


def test_kl_matches_closed_form():
    model = VAE(2, latent_dim=2, encoder_hidden_layers=ENC, decoder_hidden_layers=DEC)
    x = tf.random.normal((32, 2))
    mu, logvar = (t.numpy() for t in model.encode(x))
    ref = np.mean(0.5 * np.sum(mu**2 + np.exp(logvar) - 1 - logvar, -1))
    np.testing.assert_allclose(model.compute_losses(x)["kl"], ref, rtol=1e-5)


def test_sample_shapes():
    assert VAE(3, encoder_hidden_layers=ENC, decoder_hidden_layers=DEC).sample(5).shape == (5, 3)
    cvae = cVAE(3, 2, encoder_hidden_layers=ENC, decoder_hidden_layers=DEC)
    assert cvae.sample(5, cond=tf.zeros((4, 2))).shape == (4, 5, 3)


def test_validation():
    with pytest.raises(ValueError):
        cVAE(3, 0)
    with pytest.raises(ValueError):
        VAE(3, decoder_output_layers=[{"neurons": 2, "activation": "linear"}])
    with pytest.raises(ValueError):
        VAE(3, latent_dim=2, encoder_output_layers=[{"neurons": 2, "activation": "linear"}])
    with pytest.raises(NotImplementedError):
        VAE(3).log_prob(tf.zeros((1, 3)))
    with pytest.raises(ValueError):
        cVAE(3, 1).sample(2)
    with pytest.raises(ValueError):
        VAE(3).sample(2, cond=tf.zeros((1, 1)))


def test_default_decoder_mirrors_encoder():
    model = VAE(3)
    model.warmup()
    enc = [layer.units for layer in model.encoder.layers]
    dec = [layer.units for layer in model.decoder.layers]
    assert enc == [64, 32, 16, 6] and dec == [16, 32, 64, 3]
