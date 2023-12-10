from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar

from deepdna.nn import data_generators as dg
from deepdna.nn.callbacks import SafelyStopTrainingCallback
from deepdna.nn.models import dnabert
from dnadb import fasta
import tensorflow as tf
import wandb

ModelType = TypeVar("ModelType", bound="tf.keras.Model")

@dataclass
class DeepDNAModelManifest:
    config: Dict[Any, Any] = field(default=dict) # type: ignore
    wandb_run_id: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "DeepDNAModelManifest":
        return cls(
            config=data["config"],
            wandb_run_id=data["wandb_run_id"],
        )

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "wandb_run_id": self.wandb_run_id,
        }

@dataclass
class DeepDNAModel(Generic[ModelType]):
    model: ModelType
    manifest: DeepDNAModelManifest

    def summary(self):
        return self.model.summary()

# Model Definitions --------------------------------------------------------------------------------

@dataclass
class DNABERTPretrainingModel(DeepDNAModel[dnabert.DnaBertPretrainModel]):
    ...
    @classmethod
    def create(
        cls,
        sequence_length: int = 150,
        kmer: int = 3,
        embed_dim: int = 64,
        stack: int = 8,
        num_heads: int = 8,
    ) -> "DNABERTPretrainingModel":
        model = dnabert.DnaBertPretrainModel(
            dnabert.DnaBertModel(
                sequence_length=sequence_length,
                kmer=kmer,
                embed_dim=embed_dim,
                stack=stack,
                num_heads=num_heads))
        return cls(model, DeepDNAModelManifest({
            "model": {
                "sequence_length": sequence_length,
                "kmer": kmer,
                "embed_dim": embed_dim,
                "stack": stack,
                "num_heads": num_heads
            }
        }))

    def fit(
            self,
            train_sequences: fasta.FastaDb,
            val_sequences: Optional[fasta.FastaDb],
            epochs: int = 2000,
            steps_per_epoch: int = 100,
            mask_ratio: float = 0.15,
            batch_size: int = 256,
            val_batch_size: int = 256,
            val_frequency: int = 20,
            val_steps: int = 20,
            verbose: int = 0
    ):
        self.manifest.config["train"] = {
            "train_sequences_uuid": train_sequences.uuid,
            "val_sequences_uuid": val_sequences.uuid if val_sequences else None,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "mask_ratio": mask_ratio,
            "batch_size": batch_size,
            "val_batch_size": val_batch_size,
            "val_frequency": val_frequency,
            "val_steps": val_steps
        }
        self.model.masking.mask_ratio.assign(mask_ratio)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
        train_data = dg.BatchGenerator(batch_size, steps_per_epoch, [
            dg.random_samples(train_sequences),
            dg.random_sequence_entries(),
            dg.sequences(self.model.sequence_length),
            dg.augment_ambiguous_bases(),
            dg.encode_sequences(),
            dg.encode_kmers(self.model.kmer),
            lambda encoded_kmer_sequences: (encoded_kmer_sequences, encoded_kmer_sequences)
        ])
        val_data = dg.BatchGenerator(val_batch_size, val_steps, [
            dg.random_samples(val_sequences or train_sequences),
            dg.random_sequence_entries(),
            dg.sequences(self.model.sequence_length),
            dg.augment_ambiguous_bases(),
            dg.encode_sequences(),
            dg.encode_kmers(self.model.kmer),
            lambda encoded_kmer_sequences: (encoded_kmer_sequences, encoded_kmer_sequences)
        ], shuffle=(val_sequences is not None))
        return self.model.fit(
            train_data,
            validation_data=val_data,
            validation_freq=val_frequency,
            epochs=epochs,
            callbacks=[
                SafelyStopTrainingCallback(),
                wandb.keras.WandbMetricsLogger(),
            ],
            verbose=verbose)
