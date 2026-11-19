import numpy as np
import pytest
import tensorflow as tf

from neuraldist.generative_model.realnvp import RealNVP, cRealNVP, cRealNVPDB
from neuraldist.generative_model.vae import VAE, cVAE
from neuraldist.losses import (CollisionDBLoss, ConstantWeight, DBLoss, DBWeightScheduler,
                               ExpStepScheduler, LinearRampWeight)
from neuraldist.training import fit_model, pdf_weights
from neuraldist.training.training import DB_KEYS, make_dataset, make_db_dataset

SMALL = [{"neurons": 8, "activation": "tanh"}]
RNG = np.random.default_rng(0)


@pytest.fixture(autouse=True)
def seed():
    tf.keras.utils.set_random_seed(0)


def data(n, dim):
    return RNG.normal(size=(n, dim)).astype("float32")


def db_data(n=300):
    loss = CollisionDBLoss(rate_exponent=0.0)
    return loss, loss.make_db_data(data(n, 3), data(n, 3))


# --- dataset helpers -------------------------------------------------------------------------

def test_make_dataset_batches_all_rows():
    x, c = data(10, 2), data(10, 1)
    batches = list(make_dataset(x, c, batch_size=4, shuffle=False))
    assert [len(b[0]) for b in batches] == [4, 4, 2]
    np.testing.assert_array_equal(np.concatenate([b[1] for b in batches]), c)


def test_make_db_dataset_structure_and_steps():
    _, db = db_data(300)
    ds, steps = make_db_dataset(data(50, 3), data(50, 3), db, batch_size=16, db_batch_size=32)
    assert steps == max(4, 10)
    batch = next(iter(ds))
    assert set(batch) == {"main", "db"}
    assert set(batch["main"]) == {"v_r", "v_i"} and set(batch["db"]) == set(DB_KEYS)
    assert batch["main"]["v_r"].shape == (16, 3) and batch["db"]["v_i"].shape == (32, 3)

    ds, steps = make_db_dataset(data(50, 3), data(50, 3), x_bwd=data(50, 3), cond_bwd=data(50, 3),
                                batch_size=16)
    batch = next(iter(ds))
    assert steps == 4 and set(batch) == {"main"} and "v_r_bwd" in batch["main"]


def test_make_db_dataset_validation():
    x, c = data(10, 3), data(10, 3)
    with pytest.raises(ValueError, match="together"):
        make_db_dataset(x, c, x_bwd=x)
    with pytest.raises(ValueError, match="same number"):
        make_db_dataset(x, data(9, 3))
    _, db = db_data(20)
    with pytest.raises(ValueError, match="missing"):
        make_db_dataset(x, c, {k: v for k, v in db.items() if k != "log_M_i"})
    db["log_M_r"] = db["log_M_r"][:-1]
    with pytest.raises(ValueError, match="same number"):
        make_db_dataset(x, c, db)


# --- fit_model: plain models -----------------------------------------------------------------

@pytest.mark.parametrize("model_fn, cond_dim", [
    (lambda: RealNVP(2, hidden_layers=SMALL, n_coupling=2), 0),
    (lambda: cRealNVP(2, 1, hidden_layers=SMALL, n_coupling=2), 1),
    (lambda: VAE(2, encoder_hidden_layers=SMALL, decoder_hidden_layers=SMALL), 0),
    (lambda: cVAE(2, 1, encoder_hidden_layers=SMALL, decoder_hidden_layers=SMALL), 1),
])
def test_fit_smoke(model_fn, cond_dim):
    model = model_fn()
    x = data(64, 2)
    c = data(64, 1)[:, 0] if cond_dim else None          # 1-D cond is reshaped automatically
    hist = fit_model(model, x, c, epochs=2, batch_size=32, validation_split=0.25, seed=0)
    assert all(np.isfinite(v).all() for v in hist.history.values())
    assert "val_loss" in hist.history and len(hist.history["loss"]) == 2


def test_fit_reduces_nll():
    x = (RNG.normal(size=(512, 2)) * [0.3, 2.0] + [1.0, -1.0]).astype("float32")
    model = RealNVP(2, hidden_layers=[{"neurons": 32, "activation": "relu"}], n_coupling=4)
    before = -float(tf.reduce_mean(model.log_prob(x)))
    fit_model(model, x, epochs=40, batch_size=128, learning_rate=1e-2, seed=0)
    after = -float(tf.reduce_mean(model.log_prob(x)))
    assert after < before - 1.0


def test_fit_with_validation_data_and_dataset():
    model = cRealNVP(2, 1, hidden_layers=SMALL, n_coupling=2)
    x, c = data(40, 2), data(40, 1)
    hist = fit_model(model, x, c, epochs=1, validation_data=(x[:8], c[:8]))
    assert "val_loss" in hist.history
    ds = make_dataset(x, c, batch_size=16)
    hist = fit_model(model, ds, epochs=1, compile_model=False, validation_data=ds)
    assert "val_loss" in hist.history


@pytest.mark.parametrize("kwargs, match", [
    (dict(x=data(10, 3)), "shape"),
    (dict(cond=data(10, 1)), "unconditional"),
    (dict(validation_split=1.5), r"\(0, 1\)"),
    (dict(validation_split=0.2, validation_data=(data(4, 2),)), "either"),
    (dict(db_data={}), "no DB loss"),
])
def test_fit_validation_unconditional(kwargs, match):
    args = dict(x=data(10, 2), cond=None) | kwargs
    with pytest.raises(ValueError, match=match):
        fit_model(RealNVP(2, hidden_layers=SMALL, n_coupling=2), args.pop("x"), args.pop("cond"),
                  epochs=1, **args)


def test_fit_validation_conditional():
    model = cRealNVP(2, 1, hidden_layers=SMALL, n_coupling=2)
    with pytest.raises(ValueError, match="conditional"):
        fit_model(model, data(10, 2), epochs=1)
    with pytest.raises(ValueError, match="cond must have shape"):
        fit_model(model, data(10, 2), data(9, 1), epochs=1)
    ds = make_dataset(data(10, 2), data(10, 1))
    with pytest.raises(ValueError, match="cond must be None"):
        fit_model(model, ds, data(10, 1), epochs=1)
    with pytest.raises(ValueError, match="arrays only"):
        fit_model(model, ds, epochs=1, validation_split=0.2)


# --- fit_model: DB model ---------------------------------------------------------------------

def db_model():
    return cRealNVPDB(3, 3, hidden_layers=SMALL, n_coupling=2)


def test_db_model_compile_options():
    model = db_model()
    with pytest.raises(ValueError, match="At least one"):
        model.compile("adam", nll_fwd=False)
    model.compile("adam", db_loss=DBLoss(enabled=False))
    assert not model.use_db_loss
    model.compile("adam", db_loss=DBLoss(epoch_weighting=LinearRampWeight(0.2, 1.0, 4)))
    assert model.use_db_loss and float(model.db_weight) == pytest.approx(0.2)


def test_db_model_fit_with_scheduler():
    loss, db = db_data(256)
    loss.epoch_weighting = LinearRampWeight(0.0, 1.0, 2)
    model = db_model()
    x, c = data(64, 3), data(64, 3)
    hist = fit_model(model, x, c, db_data=db, compile_kwargs={"db_loss": loss, "nll_bwd": True},
                     x_bwd=-c, cond_bwd=-x, epochs=3, batch_size=32, callbacks=[DBWeightScheduler()])
    np.testing.assert_allclose(hist.history["db_weight"], [0.0, 0.5, 1.0])
    assert np.all(np.isfinite(hist.history["db_loss"]))
    assert np.all(np.array(hist.history["nll_bwd"]) > 0)


def test_db_model_validation_without_db_entry():
    loss, db = db_data(128)
    x, c = data(32, 3), data(32, 3)
    hist = fit_model(db_model(), x, c, db_data=db, compile_kwargs={"db_loss": loss}, epochs=2,
                     batch_size=16, validation_data=(x[:8], c[:8]))
    # validation data has no "db" entry: DB terms are reported as zero
    assert hist.history["val_db_loss"] == [0.0, 0.0]
    assert np.all(np.isfinite(hist.history["val_nll_fwd"]))


@pytest.mark.xfail(raises=KeyError, strict=True,
                   reason="_fit_db_model builds validation batches without v_r_bwd / v_i_bwd")
def test_db_model_nll_bwd_with_validation_data():
    loss, db = db_data(64)
    x, c = data(16, 3), data(16, 3)
    fit_model(db_model(), x, c, db_data=db, compile_kwargs={"db_loss": loss, "nll_bwd": True},
              x_bwd=-c, cond_bwd=-x, epochs=1, validation_data=(x[:8], c[:8]))


def test_db_model_fit_validation():
    loss, db = db_data(32)
    x, c = data(16, 3), data(16, 3)
    with pytest.raises(ValueError, match="pass cond"):
        fit_model(db_model(), x, epochs=1)
    with pytest.raises(ValueError, match="pass db_data"):
        fit_model(db_model(), x, c, compile_kwargs={"db_loss": loss}, epochs=1)
    with pytest.raises(ValueError, match="without db_loss"):
        fit_model(db_model(), x, c, db_data=db, epochs=1)
    # DB loss active but a training batch without "db" -> KeyError from the model
    model = db_model()
    model.compile("adam", db_loss=loss)
    ds, _ = make_db_dataset(x, c, batch_size=8)
    with pytest.raises(KeyError, match="db"):
        model.fit(ds.take(1), epochs=1, verbose=0)


# --- weight schedules ------------------------------------------------------------------------

def test_weight_schedules():
    np.testing.assert_allclose(ConstantWeight(0.3).get_weights(3), [0.3] * 3)
    np.testing.assert_allclose(LinearRampWeight(0, 1, 4).get_weights(6), [0, .25, .5, .75, 1, 1])
    np.testing.assert_allclose(LinearRampWeight(2, 5, 0).get_weights(2), [2, 5])
    np.testing.assert_allclose(ExpStepScheduler(0.01, 2).get_weights(5), [.01, .01, .1, .1, 1])
    loss = DBLoss(epoch_weighting=ExpStepScheduler(0.01, 2))
    assert loss.initial_weight() == pytest.approx(0.01)
    assert loss.weight_for_epoch(4) == pytest.approx(1.0)
    assert DBLoss(0.7).get_config()["loss:db:weighting:value"] == pytest.approx(0.7)


# --- training on pdf values ------------------------------------------------------------------

def test_pdf_weights():
    w = pdf_weights(pdf=[1.0, 3.0, 0.0, 4.0])
    np.testing.assert_allclose(w, [0.5, 1.5, 0.0, 2.0])
    np.testing.assert_allclose(pdf_weights(log_pdf=np.log([1.0, 3.0, 1e-300, 4.0]) + 500.0), w, rtol=1e-6)
    # importance sampling: w ∝ p / r
    np.testing.assert_allclose(pdf_weights(pdf=[2.0, 2.0], log_proposal=np.log([1.0, 3.0])), [1.5, 0.5])
    for kwargs, match in [({}, "exactly one"), (dict(pdf=[1.0], log_pdf=[0.0]), "exactly one"),
                          (dict(pdf=[-1.0, 1.0]), ">= 0"), (dict(pdf=[0.0, 0.0]), "zero"),
                          (dict(log_pdf=[np.nan]), "NaN")]:
        with pytest.raises(ValueError, match=match):
            pdf_weights(**kwargs)


def test_weighted_loss_matches_cross_entropy():
    model = RealNVP(2, hidden_layers=SMALL, n_coupling=2)
    x, w = data(32, 2), RNG.uniform(size=32).astype("float32")
    ref = -np.mean(w * model.log_prob(x).numpy())
    np.testing.assert_allclose(model.compute_losses(x, weight=w)["loss"], ref, rtol=1e-5)
    np.testing.assert_allclose(model.compute_losses(x, weight=np.ones(32, "float32"))["loss"],
                               model.compute_losses(x)["loss"], rtol=1e-6)


def test_fit_on_pdf_values_learns_gaussian():
    # uniform grid on [-5, 5]^2, target N((1, -0.5), diag(0.5², 1.2²)), log_pdf unnormalized
    g = np.linspace(-5, 5, 60)
    x = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype("float32")
    mu, sd = np.array([1.0, -0.5]), np.array([0.5, 1.2])
    log_p = -0.5 * (((x - mu) / sd) ** 2).sum(1)
    model = RealNVP(2, hidden_layers=[{"neurons": 32, "activation": "relu"}], n_coupling=4)
    hist = fit_model(model, x, log_pdf=log_p, epochs=60, batch_size=256, learning_rate=5e-3, seed=0,
                     validation_split=0.2)
    s = model.sample(20_000).numpy()
    np.testing.assert_allclose(s.mean(0), mu, atol=0.1)
    np.testing.assert_allclose(s.std(0), sd, rtol=0.1)
    # loss estimates the cross entropy -> close to the entropy of the target
    entropy = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.prod(sd ** 2))
    assert abs(hist.history["loss"][-1] - entropy) < 0.1
    assert "val_loss" in hist.history


def test_fit_on_pdf_values_conditional_and_validation():
    model = cRealNVP(2, 1, hidden_layers=SMALL, n_coupling=2)
    x, c = data(64, 2), data(64, 1)
    hist = fit_model(model, x, c, pdf=np.exp(-0.5 * (x ** 2).sum(1)), epochs=2, batch_size=32,
                     validation_data=(x[:8], c[:8]))
    assert np.all(np.isfinite(hist.history["loss"])) and "val_loss" in hist.history
    with pytest.raises(ValueError, match="one value per point"):
        fit_model(model, x, c, pdf=np.ones(10), epochs=1)
    with pytest.raises(ValueError, match="only used together"):
        fit_model(model, x, c, log_proposal=np.zeros(64), epochs=1)
    with pytest.raises(ValueError, match="does not support"):
        fit_model(VAE(2, encoder_hidden_layers=SMALL, decoder_hidden_layers=SMALL), x,
                  pdf=np.ones(64), epochs=1)
    with pytest.raises(ValueError, match="does not support"):
        fit_model(db_model(), data(16, 3), data(16, 3), pdf=np.ones(16), epochs=1)
