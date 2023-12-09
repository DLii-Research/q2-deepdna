from qiime2.plugin import Bool, Choices, Int, Float, Range, Str
from q2_types.feature_data import FeatureData
from typing import Literal, Optional, Union
from .types import DeepDNAModel, DNABERTPretrainingModel, SequenceDB
from ._registry import Field, register_method

from deepdna.nn import data_generators as dg
from deepdna.nn.callbacks import SafelyStopTrainingCallback
from deepdna.nn.models import dnabert
from dnadb import fasta
import tensorflow as tf
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
        "train_checkpoint_path": Field(Str, "The path to the checkpoint to be used for training."), # type: ignore
        "train_checkpoint_frequency": Field(Int % Range(1, None) | Str % Choices(["epoch"]), "The checkpoint frequency to use for training."), # type: ignore
        "train_batch_size": Field(Int % Range(1, None), "The batch size to be used for training."), # type: ignore
        "train_mask_ratio": Field(Float % Range(0, 1, inclusive_start=False, inclusive_end=False), "The ratio of the sequences to be masked during pre-training."), # type: ignore
        "train_val_batch_size": Field(Int % Range(1, None), "The batch size to be used for validation."), # type: ignore
        "train_val_frequency": Field(Int % Range(1, None), "The validation frequency to use for training."), # type: ignore
        "train_show_progress": Field(Bool, "Whether to show the training progress or not (--verbose required)."),

        # Wandb
        "wandb_mode": Field(Str % Choices(["disabled", "online", "offline"]), "The wandb mode to be used for logging."), # type: ignore
        "wandb_project": Field(Str, "The wandb project to be used for logging."),
        "wandb_name": Field(Str, "The wandb run name to be used for logging."),
        "wandb_entity": Field(Str, "The wandb entity to be used for logging."),
        "wandb_group": Field(Str, "The wandb group to be used for logging."),
    },
    outputs={"model": Field(DeepDNAModel[DNABERTPretrainingModel], "The pre-trained DNABERT model.")}, # type: ignore
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
    train_checkpoint_path: Optional[str] = None,
    train_checkpoint_frequency: Union[int, Literal["epoch"]] = "epoch",
    train_mask_ratio: float = 0.15,
    train_batch_size: int = 256,
    train_val_batch_size: int = 256,
    train_val_frequency: int = 1,
    train_show_progress: bool = True,
    # Wandb
    wandb_mode: str = "disabled",
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
) -> dnabert.DnaBertPretrainModel:
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
        })
    train_data = dg.BatchGenerator(train_batch_size, 100, [
        dg.random_samples(sequences),
        dg.random_sequence_entries(),
        dg.sequences(model_sequence_length),
        dg.augment_ambiguous_bases(),
        dg.encode_sequences(),
        dg.encode_kmers(model_kmer),
        lambda encoded_kmer_sequences: (encoded_kmer_sequences, encoded_kmer_sequences)
    ])
    val_data = dg.BatchGenerator(train_val_batch_size, 20, [
        dg.random_samples(sequences),
        dg.random_sequence_entries(),
        dg.sequences(model_sequence_length),
        dg.augment_ambiguous_bases(),
        dg.encode_sequences(),
        dg.encode_kmers(model_kmer),
        lambda encoded_kmer_sequences: (encoded_kmer_sequences, encoded_kmer_sequences)
    ], shuffle=False)
    print("Training the model")
    callbacks = [SafelyStopTrainingCallback()]
    if wandb_mode != "disabled":
        callbacks.append(wandb.keras.WandbCallback())
    if train_checkpoint_path is not None:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=train_checkpoint_path,
            save_weights_only=False,
            save_freq=train_checkpoint_frequency*100,
        ))
    model.fit(
        train_data,
        validation_data=val_data,
        validation_freq=train_val_frequency,
        callbacks=callbacks,
        verbose=1 if train_show_progress else 0)

    return model
