import tensorflow as tf
import keras

class GenerativeModel(keras.Model):
    """
    Base class for generative models (conditional: cons_dim > 0) and unconditional (cond_dim = 0) models.
    """

    loss_names : tuple[str, ...] = ("loss",)

    def __init__(self, x_dim: int, cond_dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.x_dim = int(x_dim)
        self.cond_dim = int(cond_dim)
        self._trackers = {name: keras.metrics.Mean(name=name) for name in self.loss_names}

    # --- Interface ---
    @property
    def conditional(self) -> bool:
        return self.cond_dim > 0

    def compute_losses(self, x, cond=None, training: bool = True) -> dict[str, tf.Tensor]:
        """
        Compute losses for the model. Must be implemented in subclasses (e.g., VAE, cVAE, etc.).
        """
        raise NotImplementedError("compute_losses must be implemented in subclasses.")

    def sample(self, n_samples: int, cond=None) -> tf.Tensor:
        """
        Sample from the generative model. Must be implemented in subclasses (e.g., VAE, cVAE, etc.).
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def log_prob(self, x, cond=None) -> tf.Tensor:
        """
        Compute the log-probability of the data under the model. Must be implemented in subclasses if applicable.
        """
        raise NotImplementedError(f"{type(self).__name__} does not give an exact log-probability")

    # --- Keras-Interface ---
    @property
    def metrics(self):
        return list(self._trackers.values())

    def _unpack(self, data):
        if isinstance(data, (tuple, list)):
            x = data[0]
            cond = data[1] if len(data) > 1 else None
        else:
            x, cond = data, None
        return self._check_cond(x, cond)

    def _check_cond(self, x, cond):
        x = tf.cast(x, tf.float32)
        if self.conditional and cond is None:
            raise ValueError(f"{type(self).__name__} ist bedingt: cond muss übergeben werden.")
        if not self.conditional and cond is not None:
            raise ValueError(f"{type(self).__name__} ist unbedingt (cond_dim=0): cond nicht erlaubt.")
        return x, None if cond is None else tf.cast(cond, tf.float32)

    def _update_metrics(self, losses):
        for name, tracker in self._trackers.items():
            tracker.update_state(losses[name])
        return {name: t.result() for name, t in self._trackers.items()}

    def train_step(self, data):
        x, cond = self._unpack(data)
        with tf.GradientTape() as tape:
            losses = self.compute_losses(x, cond, training=True)
        grads = tape.gradient(losses["loss"], self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return self._update_metrics(losses)

    def test_step(self, data):
        x, cond = self._unpack(data)
        return self._update_metrics(self.compute_losses(x, cond, training=False))

    def warmup(self) -> None:
        """Baut alle Gewichte (nötig vor load_weights)."""
        x = tf.zeros((2, self.x_dim))
        cond = tf.zeros((2, self.cond_dim)) if self.conditional else None
        self.compute_losses(x, cond, training=False)
        self.built = True  # nötig für save_weights/load_weights

    # --- Helperfunctions for sample() ---
    def _expand_cond(self, n_samples: int, cond):
        """Gibt (cond_wiederholt | None, lead_shape) zurück."""
        if cond is None:
            if self.conditional:
                raise ValueError("cond muss übergeben werden.")
            return None, tf.constant([n_samples], tf.int32), n_samples
        if not self.conditional:
            raise ValueError("Unbedingtes Modell: cond nicht erlaubt.")
        cond = tf.cast(cond, tf.float32)
        b = tf.shape(cond)[0]
        lead = tf.stack([b, tf.constant(n_samples, tf.int32)])
        return tf.repeat(cond, n_samples, axis=0), lead, b * n_samples

    def _to_lead_shape(self, x, lead):
        return tf.reshape(x, tf.concat([lead, [self.x_dim]], axis=0))

    @staticmethod
    def _concat(a, cond):
        return a if cond is None else tf.concat([a, cond], axis=-1)