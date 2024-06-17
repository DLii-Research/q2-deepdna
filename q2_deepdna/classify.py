from dnadb import dna, fasta
import numpy as np
import pandas as pd
import skbio
from qiime2.plugin import Bool, Choices, Int, Float, Range, Str
from q2_types.feature_data import DNAFASTAFormat, DNAIterator, FeatureData, Sequence, Taxonomy, TSVTaxonomyFormat
from q2_types.feature_table import BIOMV210Format, Frequency, FeatureTable
from q2_types.per_sample_sequences import SequencesWithQuality, PairedEndSequencesWithQuality, SingleLanePerSampleSingleEndFastqDirFmt, SingleLanePerSamplePairedEndFastqDirFmt
from q2_types.sample_data import SampleData
import tensorflow as tf
from tqdm import tqdm
from typing import Literal, Optional, Union
from . import models
from ._registry import Field, register_method
from .types import (
    CSVFormat,
    CSVDirectoryFormat,
    # SampleClassPrediction,
    DeepDNAModel,
    DNABERTBERTaxTaxonomyModel as DNABERTBERTaxTaxonomyModelType,
    DNABERTNaiveTaxonomyModel as DNABERTNaiveTaxonomyModelType,
    DNABERTTopDownTaxonomyModel as DNABERTTopDownTaxonomyModelType,
    SetBERTTaxonomyModel as SetBERTTaxonomyModelType,
    SetBERTClassificationModel as SetBERTClassificationModelType
)

# from deepdna.nn.models.taxonomy import AbstractTaxonomyClassificationModel, NaiveTaxonomyPrediction, HierarchicalTaxonomyPrediction

# def _sequence_iterator(sequences, batch_size):
#     batch = []
#     with open(str(sequences)) as f:
#         for entry in fasta.entries(f):
#             batch.append(entry)
#             if len(batch) < batch_size:
#                 continue
#             yield batch
#             batch.clear()
#     if len(batch) > 0:
#         yield batch

# def sequence_batch_generator(sequences, batch_size, length, kmer):
#     for batch in _sequence_iterator(sequences, batch_size):
#         sequence_ids = [entry.identifier for entry in batch]
#         sequences = np.array([dna.encode_sequence(entry.sequence[:length]) for entry in batch])
#         yield sequence_ids, dna.encode_kmers(sequences, kmer)

# @register_method(
#     "Classify taxonomy single",
#     description="Classify sequences using single-sequence taxonomy models.",
#     inputs={
#         "reads": Field(FeatureData[Sequence], "The sequences to classify."), # type: ignore
#         "model": Field(DeepDNAModel[DNABERTBERTaxTaxonomyModelType|DNABERTNaiveTaxonomyModelType|DNABERTTopDownTaxonomyModelType], "The model to use for classification."), # type: ignore
#     },
#     parameters={
#         "confidence": Field(Float % Range(0.0, 1.0, inclusive_start=True, inclusive_end=True) | Str % Choices(['disable']), "The confidence threshold to use for classification."), # type: ignore
#         "batch_size": Field(Int % Range(1, None), "The batch size to use for classification.") # type: ignore
#     },
#     outputs={"classification": Field(FeatureData[Taxonomy], "The predicted taxonomy.")} # type: ignore
# )
# def classify_taxonomy_single(
#     reads: DNAFASTAFormat,
#     model: Union[models.DNABERTBERTaxTaxonomyModel, models.DNABERTNaiveTaxonomyModel, models.DNABERTTopDownTaxonomyModel],
#     confidence: Union[float, Literal["disable"]] = 0.7,
#     batch_size: int = 512
# ) -> TSVTaxonomyFormat:
#     ff = TSVTaxonomyFormat()
#     with open(str(reads)) as f:
#         num_sequences = sum(1 for _ in fasta.entries(f))
#     with ff.open() as f:
#         f.write('\t'.join(ff.HEADER + ['Confidence']) + '\n')
#         for batch in tqdm(sequence_batch_generator(reads, batch_size, model.model.base.sequence_length, model.model.base.kmer), total=int(np.ceil(num_sequences / batch_size))):
#             sequence_ids, sequences = batch
#             sequences = tf.constant(sequences)
#             predictions = model.model.predict(sequences, verbose=0)
#             conf = -1
#             for sequence_id, prediction in zip(sequence_ids, predictions):
#                 if isinstance(prediction, NaiveTaxonomyPrediction):
#                     taxon = prediction.taxonomy
#                 else:
#                     taxon = prediction.taxonomies[-1]
#                 if confidence != 'disable':
#                     conf = prediction.confidence[-1]
#                 f.write('\t'.join([sequence_id, taxon.taxonomy_label, str(conf)]) + '\n')
#     return ff

# @register_method(
#     "SetBERT Classify",
#     description="Classify samples using SetBERT.",
#     inputs={
#         "classifier": Field(DeepDNAModel[SetBERTClassificationModelType], "The sequences to classify."), # type: ignore
#         "reads": Field(SampleData[SequencesWithQuality|PairedEndSequencesWithQuality], "The reads to classify"), # type: ignore
#     },
#     parameters={
#         "confidence": Field(Float % Range(0.0, 1.0, inclusive_start=True, inclusive_end=True) | Str % Choices(['disable']), "The confidence threshold to use for classification."), # type: ignore
#     },
#     outputs={"classification": Field(FeatureData[SampleClassPrediction], "The predicted taxonomy.")} # type: ignore
# )
# def setbert_classify(
#     classifier: models.SetBERTClassificationModel,
#     reads: Union[SingleLanePerSampleSingleEndFastqDirFmt,SingleLanePerSamplePairedEndFastqDirFmt],
#     # Parameters
#     confidence: Union[float, Literal["disable"]] = "disable",
# ) -> CSVFormat:

#     results = pd.DataFrame(columns=["Predicted Class", "Confidence"])
#     results.index.name = "Sample"

#     ff = CSVFormat()
#     with ff.open() as f:
#         f.write(results.to_csv(index=True))
#     return ff

def _trim(sequence, length, rng):
    assert len(sequence) >= length, "Sequence is too short"
    start = rng.integers(0, len(sequence) - length + 1)
    return sequence[start:start+length]

@register_method(
    "Classify taxonomy",
    description="Classify sequences using single-sequence taxonomy models.",
    inputs={
        # "reads": Field(FeatureData[Sequence], "The sequences to classify."), # type: ignore
        "model": Field(DeepDNAModel[SetBERTTaxonomyModelType], "The model to use for classification."), # type: ignore
        # "samples": Field(SampleData[SequencesWithQuality|PairedEndSequencesWithQuality], "The sequences to classify."),
        "sequences": Field(FeatureData[Sequence], "The sequences to classify."),
        "frequency_table": Field(FeatureTable[Frequency], "The frequency table of the DNA sequences.")
    },
    parameters={
        "confidence": Field(Float % Range(0.0, 1.0, inclusive_start=True, inclusive_end=True) | Str % Choices(['disable']), "The confidence threshold to use for classification."), # type: ignore
        "batch_size": Field(Int % Range(1, None), "The batch size to use for classification."), # type: ignore
        "subsample_size": Field(Int % Range(1, None), "The subsample size to use for classification.") # type: ignore
    },
    outputs={"classification": Field(FeatureData[Taxonomy], "The predicted taxonomy.")} # type: ignore
)
def classify_taxonomy(
    # reads: DNAFASTAFormat,
    model: models.SetBERTTaxonomyModel,
    # samples: SingleLanePerSampleSingleEndFastqDirFmt,
    sequences: DNAFASTAFormat,
    frequency_table: BIOMV210Format,
    # sequences: Union[
    #     SampleData[PairedEndSequencesWithQuality]
    # ],
    confidence: Union[float, Literal["disable"]] = 0.7,
    batch_size: int = 1,
    subsample_size: int = 1000
) -> TSVTaxonomyFormat:
    print("Processing...", model, model.model)
    rng = np.random.default_rng()
    sequence_map = {}
    with sequences.open() as f:
        for entry in fasta.entries(f):
            sequence_map[entry.identifier] = entry.sequence
    frequency: pd.DataFrame = frequency_table.view(pd.DataFrame)
    taxonomy_predictions = {}
    for _, row in frequency.iterrows():
        # Only keep non-zero row values
        abundances = row[row > 0].astype(int)
        n = abundances.sum()
        relative_abundance = abundances / n
        indices = rng.choice(abundances.index, size=subsample_size-(n%subsample_size), p=relative_abundance, replace=True)
        indices, counts = np.unique(indices, return_counts=True)
        abundances[indices] += counts
        x = np.array([
            dna.encode_sequence(dna.augment_ambiguous_bases(_trim(sequence_map[sequence_id], model.model.base.base.sequence_length, rng), rng))
            for sequence_id, count in abundances.items() for _ in range(count)])
        sequence_indices = np.array([sequence_id for sequence_id, count in abundances.items() for _ in range(count)])
        indices = rng.permutation(len(x))
        x = x[indices]
        sequence_indices = sequence_indices[indices]
        x = dna.encode_kmers(x, model.model.base.base.kmer)
        x = x.reshape((-1, subsample_size, x.shape[-1]))
        taxa = model.model.predict(x, batch_size=batch_size).flatten()
        for sequence_id, prediction in zip(sequence_indices, taxa):
            if sequence_id not in taxonomy_predictions:
                taxonomy_predictions[sequence_id] = [{} for _ in range(len(prediction.taxonomies))]
            for taxon, conf in zip(prediction.taxonomies, prediction.confidence):
                label = taxon.taxonomy_label
                if label not in taxonomy_predictions[sequence_id][taxon.rank]:
                    taxonomy_predictions[sequence_id][taxon.rank][label] = []
                taxonomy_predictions[sequence_id][taxon.rank][taxon.taxonomy_label].append(conf)

    ff = TSVTaxonomyFormat()
    with ff.open() as f:
        f.write('\t'.join(ff.HEADER + ['Confidence']) + '\n')
        # Aggregate predictions
        for sequence_id in frequency.columns:
            if sequence_id not in taxonomy_predictions:
                continue
            for rank in range(len(taxonomy_predictions[sequence_id]) - 1, -1, -1):
                # Aggregate confidences across each label
                for taxon, confidences in taxonomy_predictions[sequence_id][rank].items():
                    taxonomy_predictions[sequence_id][rank][taxon] = np.mean(confidences)
                # Fetch the highest confidence taxon for each rank
                taxonomy_predictions[rank] = max(taxonomy_predictions[sequence_id][rank].items(), key=lambda x: x[1])
                if confidence == 'disable':
                    f.write('\t'.join(map(str, [sequence_id, taxonomy_predictions[rank][0], -1])) + '\n')
                    break
                elif taxonomy_predictions[rank][1] >= confidence:
                    f.write('\t'.join(map(str, [sequence_id, taxonomy_predictions[rank][0], taxonomy_predictions[rank][1]])) + '\n')
                    break
            else:
                if confidence == 'disable':
                    f.write('\t'.join(map(str, [sequence_id, 'Unassigned', -1])) + '\n')
                else:
                    f.write('\t'.join(map(str, [sequence_id, 'Unassigned', 1 - taxonomy_predictions[0][1]])) + '\n')
    return ff
