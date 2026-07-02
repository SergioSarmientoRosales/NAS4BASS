import os
import random
import numpy as np


def set_global_seed(seed: int, *, seed_tensorflow: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not seed_tensorflow:
        return

    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required to seed TensorFlow random state. "
            "Install the pinned requirements, including numpy<2 for TensorFlow 2.16.x."
        ) from exc

    tf.random.set_seed(seed)
