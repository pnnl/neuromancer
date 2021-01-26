from setuptools import setup, find_packages

setup(name='neuromancer',
      version=0.01,
      description='Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularization',
      url='https://pnnl.github.io/neuromancer/',
      author='Aaron Tuor, Jan Drgona, Elliott Skomski',
      author_email='aaron.tuor@pnnl.gov',
      license='BSD2',
      packages=find_packages(), # or list of package paths from this directory
      zip_safe=False,
      classifiers=['Programming Language :: Python'],
      keywords=['Deep Learning', 'Pytorch', 'Linear Models', 'Dynamical Systems', 'Data-driven control'])