# Create a custom action used with Qiime2

# Import types from qiime
from dnadb import fasta
from deepdna.nn.models import dnabert
import os
from pathlib import Path
from qiime2.plugin import Int, Float, Range
from q2_types.feature_data import FeatureData, Sequence, DNAFASTAFormat
from .types import DeepDNAModel, DNABERTPretrainingModel
from .plugin_setup import plugin

def finetune_dnabert(dnabert_pretraining_model: dnabert.DnaBertPretrainModel) -> dnabert.DnaBertPretrainModel:
    dnabert_pretraining_model.summary()
    return dnabert_pretraining_model

plugin.methods.register_function(
    function=finetune_dnabert,
    inputs={'dnabert_pretraining_model': DeepDNAModel[DNABERTPretrainingModel]},
    parameters={},
    outputs={'model': DeepDNAModel[DNABERTPretrainingModel]},
    input_descriptions={'dnabert_pretraining_model': 'The pre-trained DNABERT model to start from.'},
    parameter_descriptions={},
    output_descriptions={},
    name='Fine-tune test',
    description="Fine-tune a model.",
    citations=[]
)

