import importlib
from ._version import get_versions

__version__ = get_versions()["version"]
importlib.import_module("q2_deepdna._data")
importlib.import_module("q2_deepdna._finetune")
importlib.import_module("q2_deepdna._pretrain")
