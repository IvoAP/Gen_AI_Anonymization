from .common import parse_args, run_experiment


def main() -> None:
    run_experiment(parse_args(mode="dptal"))


if __name__ == "__main__":
    main()