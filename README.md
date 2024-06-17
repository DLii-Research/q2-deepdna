# q2-deepdna

A QIIME 2 plugin for [deepdna models](https://github.com/DLii-Research/deep-dna).

## Taxonomy Classification via SetBERT

Taxonomy classification can be performed with a SetBERT taxonomy model artifact based on abundance in samples.

```bash
qiime deepdna classify-taxonomy \
    --i-model setbert_taxonomy_model.qza \
    --i-sequences rep-seqs.qza \
    --i-frequency-table frequency-table.qza \
    --p-confidence 0.7 \
    --o-classification taxonomy.qza \
    --verbose # for progress
```
