from .common import run_experiment
from .smartnoise import compare_baselines, comparison_args


def main() -> None:
    args = comparison_args(mode="mia")
    run_experiment(args, baseline_comparison=compare_baselines if args.compare_smartnoise else None)


if __name__ == "__main__":
    main()