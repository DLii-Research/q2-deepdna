from qiime2.plugin import Bool, Choices, Int, Float, Range, Str
from q2_types.feature_data import FeatureData
from typing import Optional
from .types import DeepDNAModel, DNABERTPretrainingModel as DNABERTPretrainingModelType, SequenceDB
from ._models import DNABERTPretrainingModel
from ._registry import Field, register_method

from dnadb import fasta
import wandb


@register_method(
    "Pretrain DNABERT",
    description="Pre-train a DNABERT model.",
    inputs={"sequences": Field(FeatureData[SequenceDB], "The feature sequences to be classified.")}, # type: ignore
    parameters={
        # Model hyperparameters
        "model_sequence_length": Field(Int % Range(1, None), "The length of the sequence to be used for the model."), # type: ignore
        "model_kmer": Field(Int % Range(1, None), "The kmer size to be used for the model."), # type: ignore
        "model_embed_dim": Field(Int % Range(1, None), "The embedding dimension to be used for the model."), # type: ignore
        "model_num_transformer_blocks": Field(Int % Range(1, None), "The number of transformer blocks to be used for the model."), # type: ignore
        "model_num_attention_heads": Field(Int % Range(1, None), "The number of attention heads within each transformer block to be used for the model."), # type: ignore

        # Training hyperparameters
        "train_epochs": Field(Int % Range(1, None), "The number of epochs to be used for training."), # type: ignore
        "train_steps_per_epoch": Field(Int % Range(1, None), "The number of steps per epoch to be used for training."), # type: ignore
        "train_batch_size": Field(Int % Range(1, None), "The batch size to be used for training."), # type: ignore
        "train_mask_ratio": Field(Float % Range(0, 1, inclusive_start=False, inclusive_end=False), "The ratio of the sequences to be masked during pre-training."), # type: ignore
        "train_val_batch_size": Field(Int % Range(1, None), "The batch size to be used for validation."), # type: ignore
        "train_val_frequency": Field(Int % Range(1, None), "The validation frequency to use for training."), # type: ignore
        "train_val_steps": Field(Int % Range(1, None), "The number of steps to use for validation."), # type: ignore
        "train_show_progress": Field(Bool, "Whether to show the training progress or not (--verbose required)."),

        # Wandb
        "wandb_mode": Field(Str % Choices(["disabled", "online", "offline"]), "The wandb mode to be used for logging."), # type: ignore
        "wandb_project": Field(Str, "The wandb project to be used for logging."),
        "wandb_name": Field(Str, "The wandb run name to be used for logging."),
        "wandb_entity": Field(Str, "The wandb entity to be used for logging."),
        "wandb_group": Field(Str, "The wandb group to be used for logging."),
    },
    outputs={"model": Field(DeepDNAModel[DNABERTPretrainingModelType], "The pre-trained DNABERT model.")}, # type: ignore
)
def pretrain_dnabert(
    sequences: fasta.FastaDb,
    # Model hyperparameters
    model_sequence_length: int = 150,
    model_kmer: int = 3,
    model_embed_dim: int = 64,
    model_num_transformer_blocks: int = 8,
    model_num_attention_heads: int = 8,
    # Training hyperparameters
    train_epochs: int = 2000,
    train_steps_per_epoch: int = 100,
    train_mask_ratio: float = 0.15,
    train_batch_size: int = 256,
    train_val_batch_size: int = 256,
    train_val_frequency: int = 20,
    train_val_steps: int = 20,
    train_show_progress: bool = True,
    # Wandb
    wandb_mode: str = "disabled",
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
) -> DNABERTPretrainingModel:
    container = DNABERTPretrainingModel.create(
        sequence_length=model_sequence_length,
        kmer=model_kmer,
        embed_dim=model_embed_dim,
        stack=model_num_transformer_blocks,
        num_heads=model_num_attention_heads)
    container.summary()
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        entity=wandb_entity,
        group=wandb_group,
        mode=wandb_mode,
        config=container.manifest.to_dict())
    container.fit(
        sequences,
        val_sequences=None,
        epochs=train_epochs,
        steps_per_epoch=train_steps_per_epoch,
        mask_ratio=train_mask_ratio,
        batch_size=train_batch_size,
        val_batch_size=train_val_batch_size,
        val_frequency=train_val_frequency,
        val_steps=train_val_steps,
        verbose=1 if train_show_progress else 0)
    return container
