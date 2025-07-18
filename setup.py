from setuptools import find_packages, setup

setup(
    name="puntingpro",
    version="0.1.0",
    # FIX: Tell setuptools that packages are under the 'src' directory
    package_dir={"": "src"},
    # FIX: Find packages within the 'src' directory
    packages=find_packages(where="src"),
)
