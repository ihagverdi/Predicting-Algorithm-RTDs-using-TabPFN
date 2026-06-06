'''
This module defines global constants and configurations that are used/shared across the project.
'''

from tabpfn_project.helper.data_source_release import get_sc_dict
from tabpfn_project.paths import DISTNET_DATA_DIR

# general constants and configurations for the project
SCENARIOS = [
    "clasp_factoring",
    "saps-CVVAR",
    "spear_qcp",
    "yalsat_qcp",
    "spear_swgcp",
    "yalsat_swgcp",
    "lpg-zeno",
]  # list of scenarios (i.e., datasets) to evaluate on (we use all scenarios from the distnet data)
MODELS = ["distnet", "tabpfn", "bayesian_distnet", "random_forest", "lognormal", "gp"]  # list of available models to evaluate
METRICS = ["NLLH", "CRPS", "Wasserstein", "KS", "MAE"]  # list of metrics to evaluate the models on (NLLH: negative log-likelihood, CRPS: continuous ranked probability score, Wasserstein: Wasserstein distance, KS: Kolmogorov-Smirnov statistic, MAE: mean absolute error)
TARGET_SCALES = ["log", "max", "original"]  # list of available target scales for the models (i.e., how to scale the target variable, which is the runtime). "log" means log1p-scaling, "max" Mini-Max scaling with min=0, and "original" means no scaling.
DISTNET_SCENARIOS = list(get_sc_dict(DISTNET_DATA_DIR).keys())  # list of scenarios in the distnet data (i.e., the actual datasets).
RANDOM_STATE = 0  # random state for 10-fold CV in distnet data and for ML models.
N_FOLDS = 10  # number of folds for cross-validation in distnet data.
N_GRID_POINTS = 15000  # number of grid points for computing the distributional metrics (CRPS, Wasserstein, KS).
LLH_EPSILON = 1e-10  # epsilon value to avoid numerical issues (such as log(0)) when computing the LLH (log-likelihood).

# Random Foreset specific constants and configurations
MAX_HPO_TRIALS = 1000000  # maximum number of HPO trials for Random Forest (we use early stopping based on WCT (wall-clock-time), so this is just a safeguard).
MAX_HPO_WCT = 3600  # maximum wall-clock-time (in seconds) for HPO of Random Forest (we use early stopping based on MAX_HPO_WCT, so this is the main constraint for HPO).

# DistNet specific constants and configurations
DISTNET_N_EPOCHS = 1000  # maximum number of epochs for training distnet (we use early stopping based on DISTNET_WCT, so this is just a safeguard).
DISTNET_BATCH_SIZE = 16  # batch size for training distnet
DISTNET_WCT = 3600  # maximum wall-clock-time (in seconds) for training distnet (we use early stopping based on DISTNET_WCT, so this is the main constraint for training distnet).
DISTNET_ES_PATIENCE = 20  # patience for early stopping of distnet (i.e., number of epochs with no improvement on the validation set before stopping).

# TabPFN specific constants and configurations
TABPFN_VAL_BATCH_SIZE = 1000  # validation batch size for TabPFN
