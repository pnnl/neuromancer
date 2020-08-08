from setuptools import setup, find_packages

setup(name='neuromancer',
      version=0.01,
      description='Constrained linear map parametrizations in pytorch',
      url='http://aarontuor.site',
      author='Jan Drgona, Aaron Tuor',
      author_email='aaron.tuor@pnnl.gov',
      license='MIT',
      packages=find_packages(), # or list of package paths from this directory
      zip_safe=False,
      classifiers=['Programming Language :: Python'],
      keywords=['Deep Learning', 'Pytorch', 'Linear Models'])