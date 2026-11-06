"""
Training helpers: fit conditional and unconditional models with a single call.

    from neuraldist.training import fit_model

    fit_model(vae, x, epochs=60)                 # unconditional: p(x)
    fit_model(cvae, x, c, epochs=60)             # conditional:   p(x | c)
"""

import keras
import numpy as np
import tensorflow as tf


def make_dataset(*arrays, batch_size: int = 256, shuffle: bool = True, seed: int | None = None):
    """(x,) for unconditional models, (x, cond) for conditional models."""
    ds = tf.data.Dataset.from_tensor_slices(arrays)
    if shuffle:
        ds = ds.shuffle(len(arrays[0]), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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
              verbose: int = 0, **fit_kwargs):
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
    """
    if compile_model:
        opt = optimizer if optimizer is not None else keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=opt, **(compile_kwargs or {}))

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
        if validation_split:
            if not 0.0 < validation_split < 1.0:
                raise ValueError("validation_split must be in (0, 1).")
            if validation_data is not None:
                raise ValueError("Use either validation_split or validation_data, not both.")
            perm = np.random.default_rng(seed).permutation(len(x))
            n_val = max(1, int(len(x) * validation_split))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            val_ds = make_dataset(*(a[val_idx] for a in arrays), batch_size=batch_size, shuffle=False)
            arrays = tuple(a[train_idx] for a in arrays)
        train_ds = make_dataset(*arrays, batch_size=batch_size, seed=seed)

    if validation_data is not None:
        if isinstance(validation_data, tf.data.Dataset):
            val_ds = validation_data
        else:
            vx, vc = _prepare(model, validation_data[0], validation_data[1] if len(validation_data) > 1 else None)
            val_ds = make_dataset(*((vx,) if vc is None else (vx, vc)), batch_size=batch_size, shuffle=False)

    # shuffling happens inside tf.data, so Keras must not shuffle again
    return model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks,
                     verbose=verbose, shuffle=False, **fit_kwargs)