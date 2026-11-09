from .model import LatentGenerator, ModelConfig

__all__ = ["LatentGenerator", "ModelConfig", "TrainingConfig", "train_generator"]


def __getattr__(name: str):
	if name in ("TrainingConfig", "train_generator"):
		from . import training

		return getattr(training, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")