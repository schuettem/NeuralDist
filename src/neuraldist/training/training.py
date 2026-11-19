"""
Training helpers: fit conditional and unconditional models with a single call.

    from neuraldist.training import fit_model

    fit_model(vae, x, epochs=60)                 # unconditional: p(x)
    fit_model(cvae, x, c, epochs=60)             # conditional:   p(x | c)

Training on pdf values instead of samples (flows only): x are evaluation points (grid, uniform or
importance-sampled), pdf / log_pdf the (possibly unnormalized) target density at these points:

    fit_model(flow, x_grid, log_pdf=log_f(x_grid), epochs=60)
"""
from __future__ import annotations

import math

import keras
import numpy as np
import tensorflow as tf


def make_dataset(*arrays, weight=None, batch_size: int = 256, shuffle: bool = True,
                 seed: int | None = None):
    """(x,) for unconditional models, (x, cond) for conditional models.
    With weight (N,): dict batches {"x", "cond" (if given), "weight"} for training on pdf values."""
    n = len(arrays[0])
    if weight is not None:
        data = {"x": arrays[0], "weight": np.asarray(weight, dtype="float32")}
        if len(arrays) > 1:
            data["cond"] = arrays[1]
        ds = tf.data.Dataset.from_tensor_slices(data)
    else:
        ds = tf.data.Dataset.from_tensor_slices(arrays)
    if shuffle:
        ds = ds.shuffle(n, seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# keys of the dict format used by models with a detailed-balance loss (e.g. cRealNVPDB). The names stem from
# the gas-surface case but are generic: v_i = condition, v_r = target, log_M_* = log equilibrium weight
# (see DBLoss; SurfaceDBLoss / CollisionDBLoss build db_data in this format)
MAIN_KEYS = ("v_i", "v_r", "v_i_bwd", "v_r_bwd")
DB_KEYS = ("v_i", "v_r", "v_i_bwd", "v_r_bwd", "log_det_jacobian_fwd", "log_det_jacobian_bwd",
           "log_M_i", "log_M_r")


def make_db_dataset(x, cond, db_data=None, *, x_bwd=None, cond_bwd=None, batch_size: int = 256,
                    db_batch_size: int | None = None, seed: int | None = None):
    """Build the {"main": ..., "db": ...} batches that Keras needs as a single dataset.

    x, cond : target / condition of the (small) main data set -> keys v_r / v_i
    db_data : dict with the keys DB_KEYS (arrays), e.g. from SurfaceDBLoss / CollisionDBLoss.make_db_data;
              may be much larger than the main data
    x_bwd, cond_bwd : reversed pairs of the main data, only needed for nll_bwd=True
    Both data sets are shuffled and repeated, so the smaller one is cycled. One epoch is one pass over the
    larger one. Returns (dataset, steps_per_epoch).
    """
    f32 = lambda a: np.asarray(a, dtype="float32")
    main = {"v_r": f32(x), "v_i": f32(cond)}
    if (x_bwd is None) != (cond_bwd is None):
        raise ValueError("x_bwd and cond_bwd must be given together.")
    if x_bwd is not None:
        main.update({"v_r_bwd": f32(x_bwd), "v_i_bwd": f32(cond_bwd)})
    n_main = len(main["v_r"])
    if any(len(v) != n_main for v in main.values()):
        raise ValueError("All main arrays must have the same number of rows.")
    main_ds = tf.data.Dataset.from_tensor_slices(main).shuffle(n_main, seed=seed).repeat().batch(batch_size)
    steps = math.ceil(n_main / batch_size)

    if db_data is None:
        return main_ds.map(lambda m: {"main": m}).prefetch(tf.data.AUTOTUNE), steps

    missing = [k for k in DB_KEYS if k not in db_data]
    if missing:
        raise ValueError(f"db_data is missing the keys {missing}.")
    db = {k: f32(db_data[k]) for k in DB_KEYS}
    n_db = len(db["v_i"])
    if any(len(v) != n_db for v in db.values()):
        raise ValueError("All db_data arrays must have the same number of rows.")
    db_batch_size = db_batch_size or batch_size
    db_ds = tf.data.Dataset.from_tensor_slices(db).shuffle(n_db, seed=seed).repeat().batch(db_batch_size)
    ds = tf.data.Dataset.zip((main_ds, db_ds)).map(lambda m, d: {"main": m, "db": d})
    return ds.prefetch(tf.data.AUTOTUNE), max(steps, math.ceil(n_db / db_batch_size))


def pdf_weights(pdf=None, log_pdf=None, log_proposal=None):
    """Importance weights w_i ∝ p(x_i) / r(x_i) for training on pdf values, normalized to mean 1.

    pdf / log_pdf : target density at the points x_i (exactly one of both); it may be unnormalized,
                    the constant cancels. log_pdf is numerically better for densities with long tails.
                    Points with pdf = 0 (log_pdf = -inf) get weight 0.
    log_proposal : log density r(x_i) of how the points were placed. None means uniform placement
                    (regular grid or uniform random points in a box): then w ∝ p(x_i).
                    For importance sampling x_i ~ r, pass log r(x_i).

    With mean-1 weights, mean(w_i * -log q(x_i)) estimates the cross entropy H(p, q) = KL(p || q) + H(p).
    For conditional targets p(x | c), the weights are normalized globally: pass a density that is
    normalized in x for every c (or whose constant does not depend on c), and give log_proposal = log r(x | c)
    if the placement of the points depends on c (e.g. a box that scales with c).
    """
    if (pdf is None) == (log_pdf is None):
        raise ValueError("Pass exactly one of pdf and log_pdf.")
    if pdf is not None:
        pdf = np.asarray(pdf, dtype="float64")
        if np.any(pdf < 0) or not np.all(np.isfinite(pdf)):
            raise ValueError("pdf values must be finite and >= 0.")
        with np.errstate(divide="ignore"):
            log_w = np.log(pdf)
    else:
        log_w = np.asarray(log_pdf, dtype="float64").copy()
        if np.any(np.isnan(log_w)) or np.any(log_w == np.inf):
            raise ValueError("log_pdf must not contain NaN or +inf.")
    log_w = log_w.reshape(-1)
    if log_proposal is not None:
        log_r = np.broadcast_to(np.asarray(log_proposal, dtype="float64").reshape(-1), log_w.shape)
        if not np.all(np.isfinite(log_r)):
            raise ValueError("log_proposal must be finite.")
        log_w = log_w - log_r
    if not np.any(np.isfinite(log_w)):
        raise ValueError("All pdf values are zero.")
    log_w = log_w - log_w[np.isfinite(log_w)].max()        # stable exp
    w = np.exp(log_w)
    return (w / w.mean()).astype("float32")


def _prepare(model, x, cond):
    """Convert to float32 arrays and check shapes against the model."""
    x = np.asarray(x, dtype="float32")
    if x.ndim != 2 or x.shape[1] != model.x_dim:
        raise ValueError(f"x must have shape (N, {model.x_dim}), got {x.shape}.")
    if model.conditional:
        if cond is None:
            raise ValueError(f"{type(model).__name__} is conditional: pass cond with shape (N, {model.cond_dim}).")
        cond = np.asarray(cond, dtype="float32")
        if cond.ndim == 1:
            cond = cond[:, None]
        if cond.shape != (len(x), model.cond_dim):
            raise ValueError(f"cond must have shape ({len(x)}, {model.cond_dim}), got {cond.shape}.")
    elif cond is not None:
        raise ValueError(f"{type(model).__name__} is unconditional (cond_dim=0): cond is not allowed.")
    return x, cond


def fit_model(model, x, cond=None, *, epochs: int = 50, batch_size: int = 256,
              learning_rate: float = 1e-3, optimizer=None, compile_model: bool = True,
              compile_kwargs: dict | None = None, validation_split: float = 0.0,
              validation_data=None, callbacks=None, seed: int | None = None,
              verbose: int = 0, db_data: dict | None = None, db_batch_size: int | None = None,
              x_bwd=None, cond_bwd=None, pdf=None, log_pdf=None, log_proposal=None,
              **fit_kwargs):
    """Compile and fit a model on arrays (or a ready-made tf.data.Dataset). Returns the Keras History.

    x, cond          : arrays (N, x_dim) / (N, cond_dim); cond is required for conditional models and
                       forbidden for unconditional ones. x may also be a tf.data.Dataset (e.g. dict data
                       of the DB model); then cond must be None.
    optimizer        : keras optimizer; default Adam(learning_rate)
    compile_model    : set False if the model was already compiled (e.g. with a custom loss)
    compile_kwargs   : extra arguments for model.compile (e.g. nll_bwd=True, db_loss=DBLoss(...))
    validation_split : fraction of the arrays used for validation (arrays only)
    validation_data  : (x_val,) / (x_val, cond_val) arrays or a tf.data.Dataset
    callbacks        : list of keras callbacks (e.g. DBWeightScheduler(), EarlyStopping)
    seed             : seed for shuffling and the validation split
    db_data          : detailed-balance points (dict with DB_KEYS, e.g. from SurfaceDBLoss / CollisionDBLoss)
                       for models compiled with a DBLoss; passed
                       separately from x / cond, the {"main", "db"} batching is done internally
    db_batch_size    : batch size of the DB points (default: batch_size)
    x_bwd, cond_bwd  : reversed main pairs, only needed for nll_bwd=True
    pdf, log_pdf     : train on pdf values instead of samples (models with supports_weights, e.g.
                       RealNVP / cRealNVP): x (and cond) are then evaluation points and pdf / log_pdf the
                       (possibly unnormalized) target density p(x_i) or p(x_i | c_i) at these points.
                       The loss is the weighted NLL mean(w_i * -log q(x_i)) with w = pdf_weights(...),
                       i.e. the cross entropy H(p, q). validation_split splits points and weights;
                       validation_data (arrays) is evaluated as plain samples (unweighted NLL).
    log_proposal     : log density of the point placement (default: uniform grid / box), see pdf_weights
    """
    if compile_model:
        opt = optimizer if optimizer is not None else keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=opt, **(compile_kwargs or {}))

    weighted = pdf is not None or log_pdf is not None
    if log_proposal is not None and not weighted:
        raise ValueError("log_proposal is only used together with pdf / log_pdf.")
    if weighted and (not getattr(model, "supports_weights", False) or hasattr(model, "use_db_loss")):
        raise ValueError(f"{type(model).__name__} does not support training on pdf values.")
    if weighted and isinstance(x, tf.data.Dataset):
        raise ValueError("pdf / log_pdf need array input; for a tf.data.Dataset yield "
                         '{"x", "cond", "weight"} batches yourself.')

    if hasattr(model, "use_db_loss"):                     # models with the {"main", "db"} dict format
        return _fit_db_model(model, x, cond, db_data, x_bwd, cond_bwd, batch_size, db_batch_size, epochs,
                             validation_data, callbacks, seed, verbose, fit_kwargs)
    if db_data is not None or x_bwd is not None or cond_bwd is not None:
        raise ValueError(f"{type(model).__name__} has no DB loss: db_data / x_bwd / cond_bwd are not supported.")

    val_ds = None
    if isinstance(x, tf.data.Dataset):
        if cond is not None:
            raise ValueError("cond must be None when x is a tf.data.Dataset.")
        if validation_split:
            raise ValueError("validation_split works with arrays only; pass validation_data instead.")
        train_ds = x
    else:
        x, cond = _prepare(model, x, cond)
        arrays = (x,) if cond is None else (x, cond)
        weight = None
        if weighted:
            weight = pdf_weights(pdf, log_pdf, log_proposal)
            if len(weight) != len(x):
                raise ValueError(f"pdf / log_pdf must have one value per point ({len(x)}), got {len(weight)}.")
        if validation_split:
            if not 0.0 < validation_split < 1.0:
                raise ValueError("validation_split must be in (0, 1).")
            if validation_data is not None:
                raise ValueError("Use either validation_split or validation_data, not both.")
            perm = np.random.default_rng(seed).permutation(len(x))
            n_val = max(1, int(len(x) * validation_split))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            renorm = lambda w: w / w.mean() if w.sum() > 0 else w   # keep mean-1 weights per split
            val_w = None if weight is None else renorm(weight[val_idx])
            val_ds = make_dataset(*(a[val_idx] for a in arrays), weight=val_w, batch_size=batch_size,
                                  shuffle=False)
            arrays = tuple(a[train_idx] for a in arrays)
            weight = None if weight is None else renorm(weight[train_idx])
        train_ds = make_dataset(*arrays, weight=weight, batch_size=batch_size, seed=seed)

    if validation_data is not None:
        if isinstance(validation_data, tf.data.Dataset):
            val_ds = validation_data
        else:
            vx, vc = _prepare(model, validation_data[0], validation_data[1] if len(validation_data) > 1 else None)
            val_ds = make_dataset(*((vx,) if vc is None else (vx, vc)), batch_size=batch_size, shuffle=False)

    # shuffling happens inside tf.data, so Keras must not shuffle again
    return model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks,
                     verbose=verbose, shuffle=False, **fit_kwargs)


def _fit_db_model(model, x, cond, db_data, x_bwd, cond_bwd, batch_size, db_batch_size, epochs,
                  validation_data, callbacks, seed, verbose, fit_kwargs):
    if isinstance(x, tf.data.Dataset):               # ready-made {"main", "db"} dataset: pass through
        if cond is not None or db_data is not None:
            raise ValueError("cond / db_data must be None when x is a tf.data.Dataset.")
        return model.fit(x, epochs=epochs, validation_data=validation_data, callbacks=callbacks,
                         verbose=verbose, shuffle=False, **fit_kwargs)
    if cond is None:
        raise ValueError(f"{type(model).__name__} is conditional: pass cond (the condition of the kernel, "
                         "e.g. the incident velocity or the pre-collision relative velocity).")
    if model.use_db_loss and db_data is None:
        raise ValueError("The model was compiled with a DBLoss: pass db_data.")
    if db_data is not None and not model.use_db_loss:
        raise ValueError("db_data was given, but the model was compiled without db_loss "
                         "(pass compile_kwargs={'db_loss': SurfaceDBLoss(...)} or CollisionDBLoss(...)).")

    ds, steps = make_db_dataset(x, cond, db_data, x_bwd=x_bwd, cond_bwd=cond_bwd, batch_size=batch_size,
                                db_batch_size=db_batch_size, seed=seed)
    val_ds = None
    if validation_data is not None:                  # (x_val, cond_val) arrays or a ready-made dataset
        if isinstance(validation_data, tf.data.Dataset):
            val_ds = validation_data
        else:
            xv, cv = validation_data
            val = {"v_r": np.asarray(xv, "float32"), "v_i": np.asarray(cv, "float32")}
            val_ds = tf.data.Dataset.from_tensor_slices({"main": val}).batch(batch_size)
    fit_kwargs.setdefault("steps_per_epoch", steps)
    return model.fit(ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks,
                     verbose=verbose, shuffle=False, **fit_kwargs)