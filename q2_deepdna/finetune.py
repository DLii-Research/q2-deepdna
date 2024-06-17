import csv
from qiime2.plugin import Bool, Choices, Int, Range, Str
from q2_types.feature_data import FeatureData
from q2_types.per_sample_sequences import SequencesWithQuality, PairedEndSequencesWithQuality, SingleLanePerSampleSingleEndFastqDirFmt, SingleLanePerSamplePairedEndFastqDirFmt
from q2_types.sample_data import SampleData
from typing import Optional, Union
from ._registry import Field, register_method
from .models import SetBERTPretrainingModel, SetBERTClassificationModel
from .types import DeepDNAModel, SetBERTClassificationModel as SetBERTClassificationModelType, SetBERTPretrainingModel as SetBERTPretrainingModelType, SequenceDB

# Classification -----------------------------------------------------------------------------------

@register_method(
    "Finetune SetBERT for Classification",
    description="Fine-tune a SetBERT model for classification.",
    inputs={
        "pretrained_model": Field(DeepDNAModel[SetBERTPretrainingModelType], "The pre-trained SetBERT model."),
        "sequences": Field(SampleData[SequencesWithQuality|PairedEndSequencesWithQuality], "The feature sequences to be classified."),
        # "feature_data": Field(FeatureData[])
    }, # type: ignore
    parameters={
        # @TODO Will move to artifact later
        "metadata_file": Field(Str, "The metadata file to be used for classification."), # type: ignore
        "feature": Field(Str, "The feature to be used for classification."), # type: ignore

        # Training hyperparameters
        "train_epochs": Field(Int % Range(1, None), "The number of epochs to be used for training."), # type: ignore
        "train_steps_per_epoch": Field(Int % Range(1, None), "The number of steps per epoch to be used for training."), # type: ignore
        "train_batch_size": Field(Int % Range(1, None), "The batch size to be used for training."), # type: ignore
        "train_show_progress": Field(Bool, "Whether to show the training progress or not (--verbose required)."),

        # Wandb
        "wandb_mode": Field(Str % Choices(["disabled", "online", "offline"]), "The wandb mode to be used for logging."), # type: ignore
        "wandb_project": Field(Str, "The wandb project to be used for logging."),
        "wandb_name": Field(Str, "The wandb run name to be used for logging."),
        "wandb_entity": Field(Str, "The wandb entity to be used for logging."),
        "wandb_group": Field(Str, "The wandb group to be used for logging."),
    },
    outputs={"model": Field(DeepDNAModel[SetBERTClassificationModelType], "The pre-trained DNABERT model.")}, # type: ignore
)
def finetune_setbert_classifier(
    pretrained_model: SetBERTPretrainingModel,
    sequences: Union[SingleLanePerSampleSingleEndFastqDirFmt,SingleLanePerSamplePairedEndFastqDirFmt],
    metadata_file: str,
    feature: str,
    # Training hyperparameters
    train_epochs: int,
    train_steps_per_epoch: int = 100,
    train_batch_size: int = 3,
    train_show_progress: bool = True,
    # Wandb
    wandb_mode: str = "disabled",
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
) -> SetBERTClassificationModel:
    targets = {}
    with open(metadata_file, "r") as f:
        header = f.readline().strip().split(',')
        i = header.index(feature)
        for row in f:
            row = row.strip().split(',')
            targets[row[0]] = row[i]
    labels = list(set(targets.values()))
    container = SetBERTClassificationModel.create(
        pretrained_model,
        labels,
    )
    import numpy as np
    container.model(np.zeros((1, 1000, 148), dtype=np.int64))
    container.summary()
    return container
    # wandb.init(
    #     project=wandb_project,
    #     name=wandb_name,
    #     entity=wandb_entity,
    #     group=wandb_group,
    #     mode=wandb_mode,
    #     config=container.manifest.to_dict())
    # container.fit(
    #     sequences,
    #     val_sequences=None,
    #     epochs=train_epochs,
    #     steps_per_epoch=train_steps_per_epoch,
    #     mask_ratio=train_mask_ratio,
    #     batch_size=train_batch_size,
    #     val_batch_size=train_val_batch_size,
    #     val_frequency=train_val_frequency,
    #     val_steps=train_val_steps,
    #     verbose=1 if train_show_progress else 0)
    # return container
    ...


# Taxonomy -----------------------------------------------------------------------------------------

# @register_method(
#     "Fine-tune DNABERT Naive Taxonomy Model",
#     description="Fine-tune a DNABERT model using the naive classification method.",
#     inputs={
#         "sequences": Field(FeatureData[SequenceDB], "The feature sequences to be classified."),
#         "taxonomy": Field(FeatureData[TaxonomyDB], "The taxonomy to be used for classification."),
#     },
#     outputs={"model": Field(DeepDNAModel[DNABERTNaiveTaxonomyModel], "The pre-trained DNABERT model.")}
# )
# def finetune_dnabert_taxonomy_naive()
