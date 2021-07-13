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
cd psl
pip install -e . --prefix .
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..

cd slim
pip install -e . --prefix .
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..

pip install -e . --prefix .
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run tests
exit_code=0

pytest tests
if [ $? != 0 ]; then
    exit_code=1
fi

# All jobs with status 0 are good, while 1 denote failure
echo BUILD_STATUS:$exit_code

