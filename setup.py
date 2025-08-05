# /tennis-betting-model/setup.py
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

### IMPROVEMENT: Consolidate all dependencies into a single source of truth.
install_deps = required + ["lightgbm", "pydantic", "flumine"]

setup(
    name="tennis_betting_model",
    version="1.1.0",
    author="Luca Panozzo",
    author_email="luca.panozzo13@gmail.com",
    description="An automated trading bot for the Betfair Tennis exchange.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tennis-betting-model",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=install_deps,
    entry_points={"console_scripts": ["tennis-value-bot=main:main"]},
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
