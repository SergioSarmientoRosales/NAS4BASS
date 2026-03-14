import copy
import tensorflow as tf
import keras
from keras import layers
from config import DEFAULT_UPSCALE, DEFAULT_INPUT_CHANNELS


@keras.saving.register_keras_serializable(package="sr")
class PixelShuffle(layers.Layer):
    def __init__(self, upscale_factor: int, **kwargs):
        super().__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.upscale_factor)

    def get_config(self):
        config = super().get_config()
        config.update({"upscale_factor": self.upscale_factor})
        return config


def get_branches(genotype):
    gens = copy.deepcopy(genotype)
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }

    channels = []
    for element in gens:
        channels.append(element.pop(0))

    branches = [[], [], []]

    for i in range(len(gens)):
        for layer in gens[i]:
            op_name, kernel, repeat = layer

            if op_name == "conv":
                for _ in range(repeat):
                    branches[i].append(layers.Conv2D(channels[i][1], kernel, **conv_args))

            elif op_name == "dil_conv_d2":
                for _ in range(repeat):
                    branches[i].append(layers.Conv2D(channels[i][1], kernel, dilation_rate=2, **conv_args))

            elif op_name == "dil_conv_d3":
                for _ in range(repeat):
                    branches[i].append(layers.Conv2D(channels[i][1], kernel, dilation_rate=3, **conv_args))

            elif op_name == "dil_conv_d4":
                for _ in range(repeat):
                    branches[i].append(layers.Conv2D(channels[i][1], kernel, dilation_rate=4, **conv_args))

            elif op_name == "Dsep_conv":
                for _ in range(repeat):
                    branches[i].extend([
                        layers.DepthwiseConv2D(kernel, **conv_args),
                        layers.Conv2D(channels[i][1], 1, **conv_args),
                    ])

            elif op_name == "invert_Bot_Conv_E2":
                expand = int(float(channels[i][1]) * 2)
                for _ in range(repeat):
                    branches[i].extend([
                        layers.Conv2D(expand, 1, **conv_args),
                        layers.DepthwiseConv2D(kernel, **conv_args),
                        layers.Conv2D(channels[i][1], kernel, **conv_args),
                    ])

            elif op_name == "conv_transpose":
                for _ in range(repeat):
                    branches[i].append(layers.Conv2DTranspose(channels[i][1], kernel, **conv_args))

            elif op_name == "identity":
                branches[i].append(layers.Identity())

            else:
                raise ValueError(f"Unknown primitive: {op_name}")

    bc = []
    bc.extend(branches)
    bc.append(channels[0][1])
    return bc


def get_model(genotype, upscale_factor=DEFAULT_UPSCALE, channels=DEFAULT_INPUT_CHANNELS):
    branch1, branch2, branch3, channels_mod = get_branches(genotype)

    conv_args = {
        "activation": "relu",
        "padding": "same",
    }

    inputs = layers.Input(shape=(None, None, channels), dtype="float32")
    inp = layers.Conv2D(channels_mod, 3, **conv_args)(inputs)

    b1 = inp
    for layer in branch1:
        b1 = layer(b1)

    b2 = inp
    for layer in branch2:
        b2 = layer(b2)

    b3 = inp
    for layer in branch3:
        b3 = layer(b3)

    x = layers.Add()([b1, b2, b3])
    x = layers.Conv2D(12, 3, **conv_args)(x)
    x = PixelShuffle(upscale_factor=upscale_factor)(x)
    outputs = layers.Conv2D(3, 3, **conv_args, dtype="float32")(x)

    return keras.Model(inputs, outputs, name="sr_surrogate_arch")