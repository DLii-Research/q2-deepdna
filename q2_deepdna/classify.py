from dnadb import dna, fasta
import numpy as np
import skbio
from qiime2.plugin import Bool, Choices, Int, Float, Range, Str
from q2_types.feature_data import DNAFASTAFormat, DNAIterator, FeatureData, Sequence, Taxonomy, TSVTaxonomyFormat
from tqdm import tqdm
from typing import Literal, Optional, Union
from . import models
from ._registry import Field, register_method
from .types import (
    DeepDNAModel,
    DNABERTBERTaxTaxonomyModel as DNABERTBERTaxTaxonomyModelType,
    DNABERTNaiveTaxonomyModel as DNABERTNaiveTaxonomyModelType,
    DNABERTTopDownTaxonomyModel as DNABERTTopDownTaxonomyModelType,
)

from deepdna.nn.models.taxonomy import AbstractTaxonomyClassificationModel

def _sequence_iterator(sequences, batch_size):
    batch = []
    with open(str(sequences)) as f:
        for entry in fasta.entries(f):
            batch.append(entry)
            if len(batch) < batch_size:
                continue
            yield batch
    if len(batch) > 0:
        yield batch

def sequence_batch_generator(sequences, batch_size, length, kmer):
    for batch in _sequence_iterator(sequences, batch_size):
        sequence_ids = [entry.identifier for entry in batch]
        sequences = np.array([dna.encode_sequence(entry.sequence[:length]) for entry in batch])
        yield sequence_ids, dna.encode_kmers(sequences, kmer)

@register_method(
    "Classify taxonomy single",
    description="Classify sequences using single-sequence taxonomy models.",
    inputs={
        "reads": Field(FeatureData[Sequence], "The sequences to classify."), # type: ignore
        "model": Field(DeepDNAModel[DNABERTBERTaxTaxonomyModelType|DNABERTNaiveTaxonomyModelType|DNABERTTopDownTaxonomyModelType], "The model to use for classification."), # type: ignore
    },
    parameters={
        "confidence": Field(Float % Range(0.0, 1.0, inclusive_start=True, inclusive_end=True) | Str % Choices(['disable']), "The confidence threshold to use for classification."), # type: ignore
        "batch_size": Field(Int % Range(1, None), "The batch size to use for classification.") # type: ignore
    },
    outputs={"classification": Field(FeatureData[Taxonomy], "The predicted taxonomy.")} # type: ignore
)
def classify_taxonomy_single(
    reads: DNAFASTAFormat,
    model: Union[models.DNABERTBERTaxTaxonomyModel, models.DNABERTNaiveTaxonomyModel, models.DNABERTTopDownTaxonomyModel],
    confidence: Union[float, Literal["disable"]] = 0.7,
    batch_size: int = 256
) -> TSVTaxonomyFormat:
    ff = TSVTaxonomyFormat()
    with open(str(reads)) as f:
        num_sequences = sum(1 for _ in fasta.entries(f))
    with ff.open() as f:
        f.write('\t'.join(ff.HEADER + ['Confidence']) + '\n')
        for batch in tqdm(sequence_batch_generator(reads, batch_size, model.model.base.sequence_length, model.model.base.kmer), total=int(np.ceil(num_sequences / batch_size))):
            sequence_ids, sequences = batch
            taxonomies, confidences = model.model.predict(sequences, verbose=0)
            for sequence_id, taxon, confidence in zip(sequence_ids, taxonomies, confidences):
                f.write('\t'.join([sequence_id, taxon.taxonomy_label, str(confidence)]) + '\n')
    return ff
