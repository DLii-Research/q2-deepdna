# Create a custom action used with Qiime2

# Import types from qiime
from dnadb import fasta
import os
from pathlib import Path
from qiime2.plugin import Int, Float, Range
from q2_types.feature_data import FeatureData, Sequence, DNAFASTAFormat
from .plugin_setup import plugin


def pretrain(sequences: DNAFASTAFormat, test_value: float) -> DNAFASTAFormat:
    return sequences

plugin.methods.register_function(
    function=pretrain,
    inputs={"sequences": FeatureData[Sequence]},
    parameters={"test_value": Float % Range(0.0, 1.0, inclusive_end=True)},
    outputs=[("output", FeatureData[Sequence])],
    input_descriptions={"sequences": 'The feature sequences to be classified.'},
    parameter_descriptions={"test_value": "A test value"},
    output_descriptions={'output': 'The resulting sequences.'},
    name="Pre-train",
    description="Pre-train a deepdna model.",
    citations=[]
)

