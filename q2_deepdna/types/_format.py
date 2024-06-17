from dnadb import fasta, taxonomy
from deepdna.nn.models import load_model
import json
import pandas as pd
from qiime2.plugin import model
import tensorflow as tf
from typing import Tuple
from ..models import (
    DeepDNAModel, DNABERTPretrainingModel, DeepDNAModelManifest,
    DNABERTNaiveTaxonomyModel, DNABERTBERTaxTaxonomyModel, DNABERTTopDownTaxonomyModel,
    SetBERTClassificationModel,SetBERTPretrainingModel, SetBERTTaxonomyModel
)
from .._registry import register_format
from ..plugin_setup import plugin, citations


@register_format
class _GenericBinaryFormat(model.BinaryFileFormat):
    def _validate_(self, level):
        pass


@register_format
class JSONFormat(model.TextFileFormat):
    def _validate_(self, level):
        pass

@register_format
class CSVFormat(model.TextFileFormat):
    def _validate_(self, level):
        pass

CSVDirectoryFormat = model.SingleFileDirectoryFormat("CSVDirectoryFormat", "data.csv", CSVFormat)

@register_format
class LMDBFormat(model.DirectoryFormat):
    data = model.File("data.mdb", format=_GenericBinaryFormat)
    lock = model.File("lock.mdb", format=_GenericBinaryFormat)


@register_format
class DNAFASTADBFormat(LMDBFormat):
    ...


@register_format
class TaxonomyDBFormat(LMDBFormat):
    ...


@register_format
class DeepDNASavedModelFormat(model.DirectoryFormat):
    manifest: model.File = model.File("manifest.json", format=JSONFormat)
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
        factory.uuid = data.uuid
        factory.write_entries(iter(data))
    return ff

@plugin.register_transformer
def _2(ff: DNAFASTADBFormat) -> fasta.FastaDb:
    return fasta.FastaDb(ff.path)

@plugin.register_transformer
def _3(data: taxonomy.TaxonomyDb) -> TaxonomyDBFormat:
    ff = TaxonomyDBFormat()
    ff.path.mkdir(parents=True, exist_ok=True)
    assert data.fasta_db is not None, "Taxonomy DB must have a corresponding FASTA DB."
    with taxonomy.TaxonomyDbFactory(ff.path, data.fasta_db, depth=data.tree.depth) as factory:
        factory.uuid = data.uuid
        factory.write_entries(iter(data)) # type: ignore
    return ff

@plugin.register_transformer
def _4(ff: TaxonomyDBFormat) -> taxonomy.TaxonomyDb:
    return taxonomy.TaxonomyDb(ff.path)

@plugin.register_transformer
def _5(data: dict) -> JSONFormat:
    ff = JSONFormat()
    with ff.open() as f:
        json.dump(data, f)
    return ff

@plugin.register_transformer
def _6(ff: JSONFormat) -> dict:
    with ff.open() as f:
        try:
            return json.load(f)
        except:
            print("Warning: Failed to read Manifest.")
            return {}

@plugin.register_transformer
def _21(data: pd.DataFrame) -> CSVFormat:
    ff = CSVFormat()
    data.to_csv(str(ff))
    return ff

@plugin.register_transformer
def _22(ff: CSVFormat) -> pd.DataFrame:
    return pd.read_csv(str(ff), index_col=0)

# Model Transformers -------------------------------------------------------------------------------

# Generic save model function
def _save_model(data: DeepDNAModel) -> DeepDNASavedModelFormat:
    ff = DeepDNASavedModelFormat()
    ff.path.mkdir(parents=True, exist_ok=True)
    ff.manifest.write_data(data.manifest.to_dict(), dict) # type: ignore
    data.model.save(ff.path / "model")
    return ff

def _load_model(ff: DeepDNASavedModelFormat) -> Tuple[tf.keras.Model, DeepDNAModelManifest]:
    model, manifest = load_model(ff.path / "model"), DeepDNAModelManifest(**(ff.manifest.view(dict) or {})) # type: ignore
    print("Loaded model:", model)
    return model, manifest

# DNABERT Pre-training Model

@plugin.register_transformer
def _7(data: DNABERTPretrainingModel) -> DeepDNASavedModelFormat:
    return _save_model(data)

@plugin.register_transformer
def _8(ff: DeepDNASavedModelFormat) -> DNABERTPretrainingModel:
    return DNABERTPretrainingModel(*_load_model(ff))

# DNABERT Taxonomy Models

@plugin.register_transformer
def _9(data: DNABERTNaiveTaxonomyModel) -> DeepDNASavedModelFormat:
    return _save_model(data)

@plugin.register_transformer
def _10(ff: DeepDNASavedModelFormat) -> DNABERTNaiveTaxonomyModel:
    return DNABERTNaiveTaxonomyModel(*_load_model(ff))

@plugin.register_transformer
def _11(data: DNABERTBERTaxTaxonomyModel) -> DeepDNASavedModelFormat:
    return _save_model(data)

@plugin.register_transformer
def _12(ff: DeepDNASavedModelFormat) -> DNABERTBERTaxTaxonomyModel:
    return DNABERTBERTaxTaxonomyModel(*_load_model(ff))

@plugin.register_transformer
def _13(data: DNABERTTopDownTaxonomyModel) -> DeepDNASavedModelFormat:
    return _save_model(data)

@plugin.register_transformer
def _14(ff: DeepDNASavedModelFormat) -> DNABERTTopDownTaxonomyModel:
    return DNABERTTopDownTaxonomyModel(*_load_model(ff))

# SetBERT Pre-training Model

@plugin.register_transformer
def _15(data: SetBERTPretrainingModel) -> DeepDNASavedModelFormat:
    return _save_model(data)

@plugin.register_transformer
def _16(ff: DeepDNASavedModelFormat) -> SetBERTPretrainingModel:
    return SetBERTPretrainingModel(*_load_model(ff))

# SetBERT Taxonomy Model

@plugin.register_transformer
def _17(data: SetBERTTaxonomyModel) -> DeepDNASavedModelFormat:
    return _save_model(data)

@plugin.register_transformer
def _18(ff: DeepDNASavedModelFormat) -> SetBERTTaxonomyModel:
    return SetBERTTaxonomyModel(*_load_model(ff))

# SetBERT Generic Classification Model

@plugin.register_transformer
def _19(data: SetBERTClassificationModel) -> DeepDNASavedModelFormat:
    return _save_model(data)

@plugin.register_transformer
def _20(ff: DeepDNASavedModelFormat) -> SetBERTClassificationModel:
    return SetBERTClassificationModel(*_load_model(ff))
