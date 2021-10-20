#!/bin/bash

# Load modules
source /etc/profile.d/modules.sh
module purge
module load python/miniconda3.9
ulimit
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh

# Activate the conda environment
env_dir='/qfs/projects/dadaist/conda-env/neuromancer'
conda activate $env_dir

# Assumes that the submodules are already cloned...
# To clone the submodules directly when cloning neuromancer:
#   - git clone --recurse-submodules <git-url>
# To initialize submodules + clone 
#   - git submodule update --init (--recursive)

# Submodules are set up to track master in respective remote
# To integrate new changes from psl/slim master:
#   - git submodule update --remote
# You may have to update the branch pointer for PSL/slim after:
#   - cd (psl/slim); git clean -ffdx; cd - # If you have changes in submodule
#   - git add (psl/sim) # To update HEAD pointer for submodule
#   - git commit ...; git push...

# Run this in order to update psl / slim with new master branches
# git submodule update --remote
# In order to commit these updated branch pointers:
# git add psl
# git add slim

# Note that `pip install` is required in CI
# Feel free to use an alternative to develop w/ PSL/slim:
#   - python setup.py develop
#   - conda develop .

cd psl # Change this directory for non-submodule psl
pip install --user -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd -

cd slim # Change this directory for non-submodule slim
pip install --user -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd -

pip install --user -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run tests
exit_code=0

pytest --workers 16 --tests-per-worker 1 tests
if [ $? != 0 ]; then
    exit_code=1
fi

# All jobs with status 0 are good, while 1 denote failure
echo BUILD_STATUS:$exit_code

