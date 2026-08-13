from setuptools import setup, find_packages

setup(
    name="single",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tqdm>=4.65.0",
        "transformers>=4.35.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.9",
)
