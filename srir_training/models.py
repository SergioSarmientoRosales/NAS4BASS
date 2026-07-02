from __future__ import annotations

import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package="srir")
class PixelShuffle(layers.Layer):
    def __init__(self, scale: int, **kwargs):
        super().__init__(**kwargs)
        self.scale = int(scale)

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


@tf.keras.utils.register_keras_serializable(package="srir")
class ResidualScale(layers.Layer):
    def __init__(self, scale: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


def _activation(name: str):
    if name == "relu":
        return layers.ReLU()
    if name == "leaky_relu":
        return layers.LeakyReLU(negative_slope=0.1)
    raise ValueError(f"Unsupported activation '{name}'")


def residual_block(
    x,
    *,
    filters: int,
    residual_scale: float,
    activation: str,
    name: str,
):
    init = tf.keras.initializers.HeNormal()
    skip = x
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer=init, name=f"{name}_conv1")(x)
    x = _activation(activation)(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer=init, name=f"{name}_conv2")(x)
    x = ResidualScale(residual_scale, name=f"{name}_scale")(x)
    return layers.Add(name=f"{name}_add")([skip, x])


def upsample_block(x, *, filters: int, scale: int, activation: str, name: str):
    init = tf.keras.initializers.HeNormal()
    x = layers.Conv2D(
        filters * (scale**2),
        3,
        padding="same",
        kernel_initializer=init,
        name=f"{name}_conv",
    )(x)
    x = PixelShuffle(scale, name=f"{name}_pixel_shuffle")(x)
    x = _activation(activation)(x)
    return x


def build_residual_sr_model(
    *,
    scale: int = 2,
    channels: int = 3,
    num_filters: int = 64,
    num_res_blocks: int = 8,
    residual_scale: float = 0.1,
    activation: str = "relu",
    final_activation: str = "sigmoid",
) -> tf.keras.Model:
    if scale not in {2, 3, 4}:
        raise ValueError("scale must be one of 2, 3, or 4")

    init = tf.keras.initializers.HeNormal()
    final_init = tf.keras.initializers.RandomNormal(stddev=1e-3)

    inputs = layers.Input(shape=(None, None, channels), dtype="float32", name="lr")
    x = layers.Conv2D(
        num_filters,
        3,
        padding="same",
        kernel_initializer=init,
        name="head",
    )(inputs)
    trunk = x

    for idx in range(num_res_blocks):
        trunk = residual_block(
            trunk,
            filters=num_filters,
            residual_scale=residual_scale,
            activation=activation,
            name=f"resblock_{idx + 1:02d}",
        )

    trunk = layers.Conv2D(
        num_filters,
        3,
        padding="same",
        kernel_initializer=init,
        name="trunk_conv",
    )(trunk)
    x = layers.Add(name="global_residual")([x, trunk])

    if scale in {2, 3}:
        x = upsample_block(x, filters=num_filters, scale=scale, activation=activation, name=f"upsample_x{scale}")
    elif scale == 4:
        x = upsample_block(x, filters=num_filters, scale=2, activation=activation, name="upsample_x2_a")
        x = upsample_block(x, filters=num_filters, scale=2, activation=activation, name="upsample_x2_b")

    outputs = layers.Conv2D(
        channels,
        3,
        padding="same",
        activation=final_activation,
        dtype="float32",
        kernel_initializer=final_init,
        name="sr",
    )(x)

    return keras.Model(inputs, outputs, name=f"residual_srir_x{scale}")


def load_or_build_model(model_cfg, data_cfg) -> tf.keras.Model:
    if model_cfg.custom_model_path:
        return tf.keras.models.load_model(model_cfg.custom_model_path, compile=False)

    if model_cfg.model_type != "residual_sr":
        raise ValueError(
            f"Unsupported model_type '{model_cfg.model_type}'. "
            "Use custom_model_path for user-provided Keras models."
        )

    return build_residual_sr_model(
        scale=data_cfg.scale,
        channels=data_cfg.channels,
        num_filters=model_cfg.num_filters,
        num_res_blocks=model_cfg.num_res_blocks,
        residual_scale=model_cfg.residual_scale,
        activation=model_cfg.activation,
        final_activation=model_cfg.final_activation,
    )
