from dnadb import fasta
from deepdna.nn.models import load_model
import pickle
from qiime2.plugin import model
import tensorflow as tf
from typing import Tuple
from .._models import DeepDNAModel, DNABERTPretrainingModel, DeepDNAModelManifest
from .._registry import register_format
from ..plugin_setup import plugin, citations


@register_format
class _GenericBinaryFormat(model.BinaryFileFormat):
    def _validate_(self, level):
        pass


@register_format
class PickleFormat(model.BinaryFileFormat):
    def _validate_(self, level):
        pass


@register_format
class DNAFASTADBFormat(model.DirectoryFormat):
    data = model.File("data.mdb", format=_GenericBinaryFormat)
    lock = model.File("lock.mdb", format=_GenericBinaryFormat)


@register_format
class DeepDNASavedModelFormat(model.DirectoryFormat):
    manifest: model.File = model.File("manifest.pkl", format=PickleFormat)
    keras_metadata_pure_tf = model.File("model/keras_metadata.pb", format=_GenericBinaryFormat)
    saved_model_pure_tf = model.File("model/saved_model.pb", format=_GenericBinaryFormat)
    variables_index = model.File("model/variables/variables.index", format=_GenericBinaryFormat)
    variables_data = model.File("model/variables/variables.data-00000-of-00001", format=_GenericBinaryFormat)


# File Format Transformers Registry ----------------------------------------------------------------

@plugin.register_transformer
def _1(data: fasta.FastaDb) -> DNAFASTADBFormat:
    ff = DNAFASTADBFormat()
    ff.path.mkdir(parents=True, exist_ok=True)
    with fasta.FastaDbFactory(ff.path) as factory:
        factory.write_entries(iter(data))
    return ff

@plugin.register_transformer
def _2(ff: DNAFASTADBFormat) -> fasta.FastaDb:
    return fasta.FastaDb(ff.path)

@plugin.register_transformer
def _3(data: dict) -> PickleFormat:
    ff = PickleFormat()
    with ff.open() as f:
        pickle.dump(data, f)
    return ff

@plugin.register_transformer
def _4(ff: PickleFormat) -> dict:
    with ff.open() as f:
        return pickle.load(f)

# Model Transformers -------------------------------------------------------------------------------

# Generic save model function
def _save_model(data: DeepDNAModel) -> DeepDNASavedModelFormat:
    ff = DeepDNASavedModelFormat()
    ff.path.mkdir(parents=True, exist_ok=True)
    ff.manifest.write_data(data.manifest.to_dict(), dict)
    data.model.save(ff.path / "model")
    return ff

def _load_model(ff: DeepDNASavedModelFormat) -> Tuple[tf.keras.Model, DeepDNAModelManifest]:
    return load_model(ff.path), DeepDNAModelManifest(**(ff.manifest.view(dict) or {})) # type: ignore

@plugin.register_transformer
def _5(data: DNABERTPretrainingModel) -> DeepDNASavedModelFormat:
    return _save_model(data)

@plugin.register_transformer
def _6(ff: DeepDNASavedModelFormat) -> DNABERTPretrainingModel:
    return DNABERTPretrainingModel(*_load_model(ff))
