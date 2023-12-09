from dnadb import fasta
from q2_types.feature_data import FeatureData, Sequence, DNAFASTAFormat
from tqdm import tqdm
from .types import (
    DNAFASTADBFormat,
    SequenceDB
)
# from .plugin_setup import plugin

from ._registry import Field, register_method


@register_method(
    "Sequences to DB",
    description="Convert a FeatureData[Sequence] artifact to a FeatureDataDB[Sequence] artifact.",
    inputs={"sequences": Field(FeatureData[Sequence], "The sequences to use for training.")}, # type: ignore
    outputs={"sequences_db": Field(FeatureData[SequenceDB], "The FASTA database to use for training.")}, # type: ignore
)
def sequences_to_db(sequences: DNAFASTAFormat) -> DNAFASTADBFormat:
    ff = DNAFASTADBFormat()
    ff._mode = "w"
    ff.path.mkdir(parents=True, exist_ok=True)
    with fasta.FastaDbFactory(ff.path) as factory:
        factory.write_entries(tqdm(fasta.entries(sequences.path)))
    return ff
