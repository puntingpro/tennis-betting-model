from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="tennis_betting_model",
    version="1.0.0",
    author="Luca PuntingPro",
    author_email="example@example.com",
    description="An automated trading bot for the Betfair Tennis exchange.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tennis-betting-model",  # Replace with your repo URL
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=required,
    entry_points={
        "console_scripts": [
            "puntingpro=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
