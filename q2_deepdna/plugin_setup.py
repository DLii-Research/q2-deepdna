from qiime2.plugin import Citations, Plugin
import q2_deepdna

citations = Citations.load("citations.bib", package="q2_deepdna")
plugin = Plugin(
    name="deepdna",
    version=q2_deepdna.__version__,
    website="https://github.com/DLii-Research/q2-deepdna",
    package="q2_deepdna",
    description="A plugin to interface deep-learning models provided by deep-dna.",
    citations=[]
)
