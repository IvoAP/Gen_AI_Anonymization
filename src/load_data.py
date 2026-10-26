from pathlib import Path

import kagglehub

DATASET_HANDLE = "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
DATASET_FILES = {
	"ML": "ML-EdgeIIoT-dataset.csv",
	"DL": "DNN-EdgeIIoT-dataset.csv",
}


def load_data() -> dict[str, Path]:
	dataset_directory = Path(kagglehub.dataset_download(DATASET_HANDLE))
	csv_paths = {}

	for dataset, filename in DATASET_FILES.items():
		matches = list(dataset_directory.rglob(filename))
		if len(matches) != 1 or not matches[0].is_file():
			raise FileNotFoundError(
				f"Expected one {filename} in {dataset_directory}; found {len(matches)}."
			)
		csv_paths[dataset] = matches[0]

	return csv_paths
