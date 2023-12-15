import importlib
from ._version import get_versions

__version__ = get_versions()["version"]
importlib.import_module("q2_deepdna.data")
importlib.import_module("q2_deepdna.classify")
importlib.import_module("q2_deepdna.finetune")
importlib.import_module("q2_deepdna.pretrain")
