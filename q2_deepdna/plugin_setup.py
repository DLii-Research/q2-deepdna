from qiime2.plugin import Citations, Plugin
import q2_deepdna

citations = Citations.load("citations.bib", package="q2_deepdna")
plugin = Plugin(
    name='deepdna',
    version=q2_deepdna.__version__,
    website='',
    package='q2_deepdna',
    description='',
    citations=[]
)
