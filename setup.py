try:
    import torch
except ImportError:
    raise ImportError("Please install torch before installing neuromancer")

from setuptools import setup, find_packages

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()
    # remove lines that start with #
    requirements = [
        r
        for r in requirements
        if not (r.startswith("#") or r.startswith("-e git+") or r.startswith("git+"))
    ]

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="neuromancer",
    version="1.3",
    description="Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularization",
    long_description=long_description,
    url="https://pnnl.github.io/neuromancer/",
    author="Aaron Tuor, Jan Drgona, Mia Skomski, Stefan Dernbach, James Koch, Zhao Chen, Christian Møldrup Legaard, Draguna Vrabie",
    author_email="aaron.tuor@pnnl.gov",
    license="BSD2",
    packages=find_packages(),  # or list of package paths from this directory
    zip_safe=False,
    classifiers=["Programming Language :: Python :: 3.10"],
    install_requires=requirements,
    keywords=[
        "Deep Learning",
        "Pytorch",
        "Linear Models",
        "Dynamical Systems",
        "Data-driven control",
    ],
)
