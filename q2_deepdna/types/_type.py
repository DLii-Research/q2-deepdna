from qiime2.plugin import SemanticType
from q2_types.feature_data import FeatureData
from ._format import CSVFormat, CSVDirectoryFormat, DNAFASTADBFormat, DeepDNASavedModelFormat, TaxonomyDBFormat
from ..plugin_setup import plugin

# Some notes for myself because these are confusing:
# 1. Formats are the actual file structure of the data on disk.
#    They outline what files should be there, and can validate them as well.
# 2. Semantic types are used as a means to annotate the type of data (not the format).
#    The semantic types can then be bound to a file format.
#    When specifying input/output types for methods, we use the semantic types.
#    Think of the semantic type as the 'key' or 'name', and the format as the value.
# 3. Transformers are used to transform between usable types and formats.
#    For example, we can transform a given FASTA DB file format into a FASTA DB object.

# DB formats ---------------------------------------------------------------------------------------
SampleClassPrediction = SemanticType("SampleClassPrediction", variant_of=FeatureData.field["type"])
SequenceDB = SemanticType("SequenceDB", variant_of=FeatureData.field["type"])
TaxonomyDB = SemanticType("TaxonomyDB", variant_of=FeatureData.field["type"])

plugin.register_semantic_types(SequenceDB, TaxonomyDB)
# plugin.register_semantic_type_to_format(FeatureData[SampleClassPrediction], CSVFormat) # type: ignore
plugin.register_semantic_type_to_format(FeatureData[SequenceDB], DNAFASTADBFormat) # type: ignore
plugin.register_semantic_type_to_format(FeatureData[TaxonomyDB], TaxonomyDBFormat) # type: ignore

plugin.register_artifact_class(FeatureData[SampleClassPrediction], directory_format=CSVDirectoryFormat, description="")

# Model formats ------------------------------------------------------------------------------------

# Generic DeepDNAModel
DeepDNAModel = SemanticType("DeepDNAModel", field_names="type")

# DNABERT Models
DNABERTPretrainingModel = SemanticType("DNABERTPretrainingModel", variant_of=DeepDNAModel.field["type"])
DNABERTNaiveTaxonomyModel = SemanticType("DNABERTNaiveTaxonomyModel", variant_of=DeepDNAModel.field["type"])
DNABERTBERTaxTaxonomyModel = SemanticType("DNABERTBERTaxTaxonomyModel", variant_of=DeepDNAModel.field["type"])
DNABERTTopDownTaxonomyModel = SemanticType("DNABERTTopDownTaxonomyModel", variant_of=DeepDNAModel.field["type"])

# SetBERT Models
SetBERTPretrainingModel = SemanticType("SetBERTPretrainingModel", variant_of=DeepDNAModel.field["type"])
SetBERTTaxonomyModel = SemanticType("SetBERTTaxonomyModel", variant_of=DeepDNAModel.field["type"])
SetBERTClassificationModel = SemanticType("SetBERTClassificationModel", variant_of=DeepDNAModel.field["type"])

plugin.register_semantic_types(
    DeepDNAModel, DNABERTPretrainingModel, DNABERTNaiveTaxonomyModel, DNABERTBERTaxTaxonomyModel,
    DNABERTTopDownTaxonomyModel, SetBERTPretrainingModel, SetBERTTaxonomyModel,
    SetBERTClassificationModel)

plugin.register_semantic_type_to_format(DeepDNAModel[DNABERTPretrainingModel], DeepDNASavedModelFormat) # type: ignore
plugin.register_semantic_type_to_format(DeepDNAModel[DNABERTNaiveTaxonomyModel], DeepDNASavedModelFormat) # type: ignore
plugin.register_semantic_type_to_format(DeepDNAModel[DNABERTBERTaxTaxonomyModel], DeepDNASavedModelFormat) # type: ignore
plugin.register_semantic_type_to_format(DeepDNAModel[DNABERTTopDownTaxonomyModel], DeepDNASavedModelFormat) # type: ignore

plugin.register_semantic_type_to_format(DeepDNAModel[SetBERTPretrainingModel], DeepDNASavedModelFormat) # type: ignore
plugin.register_semantic_type_to_format(DeepDNAModel[SetBERTTaxonomyModel], DeepDNASavedModelFormat) # type: ignore
plugin.register_semantic_type_to_format(DeepDNAModel[SetBERTClassificationModel], DeepDNASavedModelFormat) # type: ignore
