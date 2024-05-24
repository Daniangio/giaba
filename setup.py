from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "herobm/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="giaba",
    version=version,
    author="Daniele Angioletti, ---",
    description="Giaba",
    python_requires=">=3.8",
    packages=find_packages(include=[]),
    install_requires=[
        "ipykernel",
        "numpy",
    ],
    zip_safe=True,
)