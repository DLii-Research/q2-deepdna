# Create a custom action used with Qiime2

# Import types from qiime
from dnadb import fasta
from deepdna.nn.models import dnabert
import os
from pathlib import Path
from typing import Literal, Optional, Union
from qiime2.plugin import Choices, Int, Float, Range, Str
from q2_types.feature_data import FeatureData, Sequence, DNAFASTAFormat
import tensorflow as tf
from .types import DNAFASTADBFormat, DNAFASTADB, DeepDNAModel, DNABERTPretrainingModel
from .plugin_setup import plugin

def pretrain_dnabert(
    sequences: fasta.FastaDb,
    # Model hyperparameters
    model_sequence_length: int = 150,
    model_kmer: int = 3,
    model_embed_dim: int = 64,
    model_num_transformer_blocks: int = 8,
    model_num_attention_heads: int = 8,
    # Training hyperparameters
    train_checkpoint_path: Optional[str] = None,
    train_checkpoint_frequency: Union[int, Literal["epoch"]] = "epoch",
    train_mask_ratio: float = 0.15,
    train_batch_size: int = 256,
    val_batch_size: int = 256,
    # Wandb
    wandb_mode: str = "disabled",
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
) -> dnabert.DnaBertPretrainModel:
    import tensorflow as tf
    import wandb
    print("Creating the model")
    model = dnabert.DnaBertPretrainModel(
        dnabert.DnaBertModel(
            sequence_length=model_sequence_length,
            kmer=model_kmer,
            embed_dim=model_embed_dim,
            stack=model_num_transformer_blocks,
            num_heads=model_num_attention_heads)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    model.summary()
    if wandb_mode != "disabled":
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            entity=wandb_entity,
            group=wandb_group,
            mode=wandb_mode
        )
        wandb.config.update({
            "model_sequence_length": model_sequence_length,
            "model_kmer": model_kmer,
            "model_embed_dim": model_embed_dim,
            "model_num_transformer_blocks": model_num_transformer_blocks,
            "model_num_attention_heads": model_num_attention_heads,
            "train_mask_ratio": train_mask_ratio,
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
        })
    model(tf.zeros((1, model.sequence_length - model.kmer + 1), dtype=tf.int32))
    return model


plugin.methods.register_function(
    function=pretrain_dnabert,
    inputs={
        "sequences": DNAFASTADB
    },
    parameters={
        # Model hyperparameters
        "model_sequence_length": Int % Range(1, None),
        "model_kmer": Int % Range(1, None),
        "model_embed_dim": Int % Range(1, None),
        "model_num_transformer_blocks": Int % Range(1, None),
        "model_num_attention_heads": Int % Range(1, None),

        # Training hyperparameters
        "train_checkpoint_path": Str,
        "train_checkpoint_frequency": Int % Range(1, None) | Str % Choices(["epoch"]),
        "train_mask_ratio": Float % Range(0, 1),
        "train_batch_size": Int % Range(1, None),
        "val_batch_size": Int % Range(1, None),

        # Wandb
        "wandb_mode": Str % Choices(["disabled", "online", "offline"]),
        "wandb_project": Str,
        "wandb_name": Str,
        "wandb_entity": Str,
        "wandb_group": Str
    },
    outputs={
        "model": DeepDNAModel[DNABERTPretrainingModel]
    },
    input_descriptions={
        "sequences": 'The feature sequences to be classified.'
    },
    parameter_descriptions={
        # Model hyperparameters
        "model_sequence_length": "The length of the sequence to be used for the model.",
        "model_kmer": "The kmer size to be used for the model.",
        "model_embed_dim": "The embedding dimension to be used for the model.",
        "model_num_transformer_blocks": "The number of transformer blocks to be used for the model.",
        "model_num_attention_heads": "The number of attention heads within each transformer block to be used for the model.",

        # Training hyperparameters
        "train_checkpoint_path": "The path to the checkpoint to be used for training.",
        "train_checkpoint_frequency": "The checkpoint frequency to use for training.",
        "train_batch_size": "The batch size to be used for training.",
        "train_mask_ratio": "The ratio of the sequences to be masked during pre-training.",
        "val_batch_size": "The batch size to be used for validation.",

        # Wandb
        "wandb_mode": "The wandb mode to be used for logging.",
        "wandb_project": "The wandb project to be used for logging.",
        "wandb_name": "The wandb run name to be used for logging.",
        "wandb_entity": "The wandb entity to be used for logging.",
        "wandb_group": "The wandb group to be used for logging.",
    },
    output_descriptions={
        'model': 'The path to store the pre-trained model'
    },
    name="Pre-train DNABERT",
    description="Pre-train a DNABERT model.",
    citations=[]
)

