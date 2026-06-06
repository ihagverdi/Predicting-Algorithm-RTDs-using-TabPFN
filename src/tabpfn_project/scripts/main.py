import argparse
import pickle
import time

from tabpfn_project.experiment_config import ExperimentConfig
from tabpfn_project.globals import DISTNET_SCENARIOS, MODELS, TARGET_SCALES
from tabpfn_project.helper.utils import TargetScale, generate_experiment_id
from tabpfn_project.paths import RESULTS_DIR
from tabpfn_project.scripts.model_handler import (
    BayesianDistNetHandler,
    DistNetHandler,
    GPHandler,
    LognormalHandler,
    RFHandler,
    TabPFNHandler,
)
from tabpfn_project.scripts.prepare_data import prepare_datasets


def train_test_model(cfg: ExperimentConfig):
    save_dir = RESULTS_DIR / cfg.save_dir.lstrip("/\\")
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train_flat, X_test, y_train_flat, y_test, train_group_ids_flat = prepare_datasets(
        cfg
    )

    handlers = {
        "distnet": DistNetHandler(),
        "bayesian_distnet": BayesianDistNetHandler(),
        "tabpfn": TabPFNHandler(),
        "random_forest": RFHandler(),
        "lognormal": LognormalHandler(),
        "gp": GPHandler(),
    }

    if cfg.model_name not in handlers:
        raise ValueError(f"Unsupported model: {cfg.model_name}")

    handler = handlers[cfg.model_name]
    model_results = handler.run(
        cfg, X_train_flat, X_test, y_train_flat, y_test, train_group_ids_flat
    )

    results_dict = {
        "model_name": cfg.model_name,
        "scenario": cfg.scenario,
        "fold": cfg.fold,
        "seed_context_size": cfg.seed_context_size,
        "seed_feature_drop_rate": cfg.seed_feature_drop_rate,
        "seed_samples_per_instance": cfg.seed_samples_per_instance,
        "jitter_x": cfg.jitter_x,
        "jitter_val": cfg.jitter_val,
        "rand_extend_x": cfg.rand_extend_x,
        "n_rand_cols": cfg.n_rand_cols,
        "subsample_unflattened": cfg.subsample_unflattened,
        "feature_drop_rate": cfg.feature_drop_rate,
        "n_features_keep": cfg.n_features_keep,
        "context_size": cfg.context_size,
        "target_scale": cfg.target_scale.value,
        "num_samples_per_instance": cfg.num_samples_per_instance,
        "use_cpu": cfg.use_cpu,
        "save_dir": str(save_dir),
        "remove_duplicates": cfg.remove_duplicates,
        "oracle": cfg.oracle,
        "do_hpo": cfg.do_hpo,
        "train_size": (X_train_flat.shape[0], X_train_flat.shape[1]),
        "test_size": (X_test.shape[0], X_test.shape[1]),
        **model_results,
    }

    exp_id = generate_experiment_id(cfg)
    res_filename = f"{exp_id}_metadata.pkl"

    meta_dir = save_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)

    with open(meta_dir / res_filename, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {meta_dir / res_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate predictive models (e.g. TabPFN, DistNet, Bayes DistNet, Random Forest, Gaussian Process) on algorithm runtime scenarios."
    )
    parser.add_argument(
        "--scenario", type=str, required=True, choices=DISTNET_SCENARIOS,
        help="Dataset/Scenario to use."
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=MODELS,
        help="The predictive model to run."
    )
    parser.add_argument(
        "--remove_duplicates", action="store_true",
        help="Keep only 1 sample per instance in training data."
    )
    parser.add_argument(
        "--oracle", action="store_true",
        help="Run model in oracle mode (uses test targets directly for perfect prediction). Applicable only for TabPFN and Lognormal models."
    )
    parser.add_argument(
        "--fold", type=int, required=True, choices=range(10),
        help="The cross-validation fold to run."
    )
    parser.add_argument(
        "--num_samples_per_instance", type=int, default=100,
        help="Number of runtime samples to use per instance (1-100)."
    )
    parser.add_argument(
        "--target_scale", type=str, required=True, choices=TARGET_SCALES,
        help="Type of transformation to apply to the targets (i.e., the runtime values)."
    )
    parser.add_argument(
        "--subsample_unflattened", action="store_true",
        help="Subsamples the training data at the instance level before flattening."
    )
    parser.add_argument(
        "--jitter_x", action="store_true",
        help="Applies gaussian baseline jitter to training features."
    )
    parser.add_argument(
        "--jitter_val", type=float, default=None,
        help="Required if `jitter_x` is set, controls the intensity (std dev) of the gaussian feature jitter."
    )
    parser.add_argument(
        "--rand_extend_x", action="store_true",
        help="Appends random noise columns to the features."
    )
    parser.add_argument(
        "--n_rand_cols", type=int, default=None,
        help="Required if `rand_extend_x` is set, controls the number of random noise columns to append."
    )
    parser.add_argument(
        "--context_size", type=int, default=None,
        help="Training samples OR instances to subsample; specified by `subsample_unflattened`."
    )
    parser.add_argument(
        "--feature_drop_rate", type=float, default=None,
        help="Fraction of features to randomly drop (0.0 to 1.0)."
    )
    parser.add_argument(
        "--n_features_keep", type=int, default=None,
        help="Number of features to keep in a random manner (takes precedence over `feature_drop_rate`)."
    )
    parser.add_argument(
        "--seed_context_size", type=int, default=None,
        help="Random seed for data subsampling. Required if `context_size` is set."
    )
    parser.add_argument(
        "--seed_feature_drop_rate", type=int, default=None,
        help="Random seed for feature dropping. Required if `feature_drop_rate` or `n_features_keep` is set."
    )
    parser.add_argument(
        "--seed_samples_per_instance", type=int, default=None,
        help="Random seed for subsampling targets per instance. Required if `num_samples_per_instance` is set and less than 100."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Subdirectory name to save experiment results and metadata."
    )
    parser.add_argument(
        "--early_stopping", action="store_true",
        help="Enable early stopping during training by using a validation split."
    )
    parser.add_argument(
        "--use_cpu", action="store_true",
        help="Force model execution on CPU even if a GPU is available."
    )
    parser.add_argument(
        "--do_hpo", action="store_true",
        help="Enable SMAC hyperparameter optimization (for Random Forest)."
    )
    parser.add_argument(
        "--rf_new_default", action="store_true",
        help="Use updated default hyperparameters (for Random Forest)."
    )

    args = parser.parse_args()
    config = ExperimentConfig(
        scenario=args.scenario,
        model_name=args.model,
        fold=args.fold,
        save_dir=args.save_dir,
        num_samples_per_instance=args.num_samples_per_instance,
        context_size=args.context_size,
        use_cpu=args.use_cpu,
        target_scale=TargetScale.from_str(args.target_scale),
        subsample_unflattened=args.subsample_unflattened,
        early_stopping=args.early_stopping,
        seed_context_size=args.seed_context_size,
        seed_feature_drop_rate=args.seed_feature_drop_rate,
        feature_drop_rate=args.feature_drop_rate,
        seed_samples_per_instance=args.seed_samples_per_instance,
        do_hpo=args.do_hpo,
        oracle=args.oracle,
        remove_duplicates=args.remove_duplicates,
        jitter_x=args.jitter_x,
        rand_extend_x=args.rand_extend_x,
        n_rand_cols=args.n_rand_cols,
        jitter_val=args.jitter_val,
        n_features_keep=args.n_features_keep,
        rf_new_default=args.rf_new_default,
    )

    start = time.perf_counter()
    train_test_model(config)
    print(f"✅ Experiment completed in {time.perf_counter() - start:.2f} seconds.")
