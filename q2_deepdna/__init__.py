import importlib
import tensorflow as tf
from ._version import get_versions

# Enable dynamic memory growth
for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

__version__ = get_versions()["version"]
importlib.import_module("q2_deepdna.data")
importlib.import_module("q2_deepdna.classify")
importlib.import_module("q2_deepdna.finetune")
importlib.import_module("q2_deepdna.pretrain")
