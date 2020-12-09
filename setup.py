from setuptools import setup, find_packages

d = {}
exec(open("spikecomparison/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

pkg_name = "spikecomparison"

setup(
    name=pkg_name,
    version=version,
    author="Alessio Buccino, Cole Hurwitz, Samuel Garcia, Jeremy Magland, Matthias Hennig",
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
        'spikeextractors>=0.9.3',
        'spiketoolkit>=0.7.2',
        'spikesorters>=0.4.3',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
