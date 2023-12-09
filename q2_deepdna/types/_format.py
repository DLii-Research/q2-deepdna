from dnadb import fasta
from deepdna.nn.models import load_model, dnabert
from qiime2.plugin import model
import tensorflow as tf
from ..plugin_setup import plugin, citations

class _GenericBinaryFormat(model.BinaryFileFormat):
    def _validate_(self, level):
        pass


class DNAFASTADBFormat(model.DirectoryFormat):
    data = model.File("data.mdb", format=_GenericBinaryFormat)
    lock = model.File("lock.mdb", format=_GenericBinaryFormat)


class DeepDNASavedModelFormat(model.DirectoryFormat):
    keras_metadata_pure_tf = model.File("keras_metadata.pb", format=_GenericBinaryFormat)
    saved_model_pure_tf = model.File("saved_model.pb", format=_GenericBinaryFormat)
    variables_index = model.File("variables/variables.index", format=_GenericBinaryFormat)
    variables_data = model.File("variables/variables.data-00000-of-00001", format=_GenericBinaryFormat)


plugin.register_formats(_GenericBinaryFormat)
plugin.register_formats(DNAFASTADBFormat)
plugin.register_formats(DeepDNASavedModelFormat)

# Transformer Registry -----------------------------------------------------------------------------

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

# Model Transformers -------------------------------------------------------------------------------

# Generic save model function
def _save_model(data: tf.keras.Model) -> DeepDNASavedModelFormat:
    ff = DeepDNASavedModelFormat()
    ff.path.mkdir(parents=True, exist_ok=True)
    data.save(ff.path)
    return ff

@plugin.register_transformer
def _3(data: dnabert.DnaBertPretrainModel) -> DeepDNASavedModelFormat:
    return _save_model(data)

@plugin.register_transformer
def _4(ff: DeepDNASavedModelFormat) -> dnabert.DnaBertPretrainModel:
    return load_model(ff.path, dnabert.DnaBertPretrainModel)
