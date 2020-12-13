# Constrained Block Nonlinear Neural Dynamical Models
_Elliott Skomski, Soumya Vasisht, Colby Wight, Aaron Tuor, Ján Drgoňa, Draguna Vrabie_

The main driver script for this work is `system_id.py`, which can be found in the `neuromancer/train_scripts` directory of this repo. For more details on the creation of the model components and objective/constraint terms used, these are defined in `common.py`. Note that the estimator arrival loss $Q_e$ is unused in this work&mdash;*i.e.* using argument `-Q_e 0`.

Hyperparameter optimization was performed via random search and asynchronous genetic search, both implemented by [APE](https://github.com/pnnl/ape); the search space configuration file has been included here as `cssm_space.yml`. The best hyperparameters obtained by these searches grouped by structured and unstructured model classes have been included in CSV files under the `analysis` directory (`best_systems.csv` and `best_blackbox.csv`, respectively).

## Ablation Study
Scripts ablating objective constraints and model structure can be found in the `analysis/` directory. Each script takes a CSV file of model hyperparameters and generates a Bash script containing a series of `system_id.py` invocations with arguments to perform various ablations over each model in the CSV.
- `nonlin_ablation.py`: individually ablates objective constraint weights and linear map structure.
- `nonlin_allablate.py`: ablates all objective constraint weights and linear map structure at once to validate how models perform without constraints.
- `nonlin_blackbox_ablate.py`: ablates all objective constraint weights, linear map structure, and SSM structure at once to validate how models perform without constraints and algebraic structure.
- `nonlin_noablation.py`: performs no ablation, meant to provide benchmarks against which ablations can be compared.

## Eigenvalue Analysis
Also under the `analysis/` directory is a script called `eigval_plots.py`, which generates eigenvalue spectrum plots for the best-observed block nonlinear models and their corresponding ablated models.