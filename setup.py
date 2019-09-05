from setuptools import setup, find_packages

d = {}
exec(open("spikecomparison/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

pkg_name = "spikecomparison"

setup(
    name=pkg_name,
    version=version,
    author="Cole Hurwitz, Jeremy Magland, Alessio Paolo Buccino, Matthias Hennig, Samuel Garcia",
    author_email="alessiop.buccino@gmail.com",
    description="Python toolkit for analysis, visualization, and comparison of spike sorting output",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alejoe91/spikecomparison",
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numpy', 
        'scipy',
        'pandas',
        'networkx',
        'joblib',
        'spikeextractors',
        'spiketoolkit',
        'spikesorters',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
