from dnadb import fasta
import os
from pathlib import Path
from qiime2.plugin import Int, Float, Range
from q2_types.feature_data import FeatureData, Sequence, DNAFASTAFormat
import time
from tqdm import tqdm
from .types import DNAFASTADB, DNAFASTADBFormat
from .plugin_setup import plugin

def load_training_sequences(sequences: DNAFASTAFormat) -> DNAFASTADBFormat:
    print("Loading the training data")
    ff = DNAFASTADBFormat()
    ff._mode = "w"
    ff.path.mkdir(parents=True, exist_ok=True)
    with fasta.FastaDbFactory(ff.path) as factory:
        factory.write_entries(tqdm(fasta.entries(sequences.path)))
    return ff
    # fasta_db_path = Path(fasta_path.path).with_suffix(".fasta.db")
    # assert os.access(fasta_db_path.parent, os.W_OK)
    # with fasta.FastaDbFactory(fasta_db_path) as factory:
    #     factory.write_entries(fasta.entries(fasta_path.path))
    # return fasta.FastaDb(fasta_db_path)

plugin.methods.register_function(
    function=load_training_sequences,
    inputs={"sequences": FeatureData[Sequence]},
    parameters={},
    outputs=[("training_sequences", DNAFASTADB)],
    input_descriptions={"sequences": "The sequences to use for training."},
    parameter_descriptions={},
    output_descriptions={"training_sequences": "The FASTA database to use for training."},
    name="Load training sequences",
    description="Load the training data into a FASTA database.",
)
