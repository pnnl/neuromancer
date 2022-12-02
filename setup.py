from setuptools import setup, find_packages

setup(
    name="neuromancer",
    version=1.2.1,
    description="Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularization",
    url="https://pnnl.github.io/neuromancer/",
    author="Aaron Tuor, Jan Drgona, Mia Skomski, Stefan Dernbach, James Koch, Zhao Chen, Christian Møldrup Legaard, Draguna Vrabie",
    author_email="aaron.tuor@pnnl.gov",
    license="BSD2",
    packages=find_packages(),  # or list of package paths from this directory
    zip_safe=False,
    classifiers=['Programming Language :: Python :: 3.10'],
    keywords=[
        "Deep Learning",
        "Pytorch",
        "Linear Models",
        "Dynamical Systems",
        "Data-driven control",
    ],
)
