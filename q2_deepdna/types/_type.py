from qiime2.plugin import SemanticType
from ._format import DNAFASTADBFormat, DeepDNASavedModelFormat
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

DNAFASTADB = SemanticType("DNAFASTADB")
DeepDNAModel = SemanticType("DeepDNAModel", field_names="type")
DNABERTPretrainingModel = SemanticType('DNABERTPretrainingModel', variant_of=DeepDNAModel.field['type'])

plugin.register_semantic_types(DNAFASTADB)
plugin.register_semantic_types(DeepDNAModel, DNABERTPretrainingModel)
plugin.register_semantic_type_to_format(DNAFASTADB, DNAFASTADBFormat)
plugin.register_semantic_type_to_format(DeepDNAModel[DNABERTPretrainingModel], DeepDNASavedModelFormat)
