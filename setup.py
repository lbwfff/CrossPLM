"""Single install for the whole project.

  pip install -e .

installs BOTH the `single` package (SAE module, importable from anywhere) and the
`crossplm` console command (the unified CLI). After that:

  crossplm training eval ...       # unified CLI
  from single.label_maps import ...  # package importable anywhere

Without installing, use `python crossplm.py ...` from the repository root.
"""
from setuptools import setup, find_packages

setup(
    name="crossplm",
    version="0.1.0",
    py_modules=["crossplm"],                      # repo-root unified CLI
    package_dir={"single": "Single/single"},      # `single` package lives in Single/single/
    packages=find_packages("Single"),             # -> single, single.sae, ...
    entry_points={"console_scripts": ["crossplm = crossplm:main"]},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.9",
)
