from ._registry import Field, register_method

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
