from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="srir")
class CharbonnierLoss(tf.keras.losses.Loss):
    def __init__(self, epsilon: float = 1e-3, name: str = "charbonnier"):
        super().__init__(name=name)
        self.epsilon = float(epsilon)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        diff = y_true - y_pred
        return tf.reduce_mean(tf.sqrt(tf.square(diff) + self.epsilon**2))

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


def get_loss(name: str, *, charbonnier_epsilon: float = 1e-3):
    name = name.lower()
    if name == "charbonnier":
        return CharbonnierLoss(epsilon=charbonnier_epsilon)
    if name in {"l1", "mae"}:
        return tf.keras.losses.MeanAbsoluteError()
    if name == "mse":
        return tf.keras.losses.MeanSquaredError()
    raise ValueError(f"Unknown SRIR loss '{name}'")
