from dnadb import fasta, taxonomy
from q2_types.feature_data import DNAFASTAFormat, FeatureData, Sequence, Taxonomy
import pandas as pd
from tqdm import tqdm
from .types import (
    DNAFASTADBFormat,
    TaxonomyDBFormat,
    SequenceDB,
    TaxonomyDB
)
# from .plugin_setup import plugin

from ._registry import Field, register_method


@register_method(
    "Sequences to DB",
    description="Convert a FeatureData[Sequence] artifact to a FeatureData[SequenceDB] artifact.",
    inputs={"sequences": Field(FeatureData[Sequence], "The sequences to use for training.")}, # type: ignore
    outputs={"sequences_db": Field(FeatureData[SequenceDB], "The FASTA database to use for training.")}, # type: ignore
)
def sequences_to_db(sequences: DNAFASTAFormat) -> DNAFASTADBFormat:
    ff = DNAFASTADBFormat()
    ff.path.mkdir(parents=True, exist_ok=True)
    with fasta.FastaDbFactory(ff.path) as factory:
        factory.write_entries(tqdm(fasta.entries(sequences.path)))
    return ff

@register_method(
    "Taxonomy to DB",
    description="Convert a FeatureData[Taxonomy] artifact to a FeatureData[TaxonomyDB] artifact.",
    inputs={
        "sequences_db": Field(FeatureData[SequenceDB], "The sequences to corresponding to the taxonomies."), # type: ignore
        "taxonomies": Field(FeatureData[Taxonomy], "The taxonomies corresponding to the given sequences DB") # type: ignore
    },
    outputs={"taxonomy_db": Field(FeatureData[TaxonomyDB], "The taxonomy database.")}, # type: ignore
)
def taxonomy_to_db(sequences_db: fasta.FastaDb, taxonomies: pd.DataFrame) -> TaxonomyDBFormat:
    ff = TaxonomyDBFormat()
    ff.path.mkdir(parents=True, exist_ok=True)
    with taxonomy.TaxonomyDbFactory(ff.path, fasta_db=sequences_db) as factory:
        for sequence_id, taxon in tqdm(taxonomies.itertuples(index=True), total=len(taxonomies)):
            factory.write_sequence(sequence_id, taxon)
    return ff


