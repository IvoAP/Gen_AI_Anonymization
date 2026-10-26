import pandas as pd

from load_data import load_data


def main() -> None:
	for dataset, csv_path in load_data().items():
		data = pd.read_csv(csv_path, nrows=5, low_memory=False)
		print(f"\n{dataset} dataset: {csv_path.name}")
		print(data.head())


if __name__ == "__main__":
	main()
