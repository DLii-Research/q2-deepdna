from setuptools import setup, find_packages
import versioneer

setup(
    name="q2-deepdna",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    author="David Ludwig",
    author_email="davidludwigii@gmail.com",
    description="deepdna plugin for QIIME 2",
    license='',
    url="",
    entry_points={
        'qiime2.plugins':
        ['q2-deepdna=q2_deepdna.plugin_setup:plugin']
    },
    package_data={
        'q2_deepdna': ['citations.bib'],
        # 'q2_deepdna.tests': ['data/*'],
    },
    zip_safe=False,
)
