#!/bin/bash

# Load modules
source /etc/profile.d/modules.sh
module purge
module load python/anaconda3.2019.3
ulimit
source /share/apps/python/anaconda3.2019.3/etc/profile.d/conda.sh

# Activate the conda environment
env_dir='/qfs/projects/dadaist/conda-env/neuromancer'
conda activate $env_dir

# Assumes that the submodules are already cloned...
# To clone the submodules directly when cloning neuromancer:
#   - git clone --recurse-submodules <git-url>
# To initialize submodules + clone 
#   - git submodule update --init (--recursive)

# Note that `pip install` is required in CI
# Feel free to use an alternative to develop w/ PSL/slim:
#   - python setup.py develop
#   - conda develop .

cd psl # Change this directory for non-submodule psl
pip install -e . --prefix .
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd -

cd slim # Change this directory for non-submodule slim
pip install -e . --prefix .
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd -

pip install -e . --prefix .
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run tests
exit_code=0

pytest --workers 64 --tests-per-worker 1 tests
if [ $? != 0 ]; then
    exit_code=1
fi

# All jobs with status 0 are good, while 1 denote failure
echo BUILD_STATUS:$exit_code

