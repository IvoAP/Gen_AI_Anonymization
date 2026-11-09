# Gen_AI_Anonymization

## Project organization

Reusable implementations stay in `src/gen_ai_anonymization/`: `dptal/` contains
the proposed model, training and evaluation utilities; `baseline/` contains
the separate SmartNoise adapters. Dataset inspection tools stay in `scripts/`.

Experiment orchestration is separate:

- `experiments/dptal.py`: real-feature reference and comparison of classifiers
	trained on real versus synthetic latents, without MIA.
- `experiments/mia.py`: membership audits for the real-feature, real-latent and
	synthetic-latent classifier pipelines, with their utility metrics.
- `experiments/smartnoise.py`: DP-TAL synthetic latents versus MWEM, MST, AIM and
	DP-CTGAN synthetic latents in a shared public-binned latent space.
- `experiments/common.py`: shared arguments, DP-TAL training, evaluation and
	persistence; `experiments/smartnoise_worker.py` is an internal isolated worker.

`tests/` holds unit and integration tests. `data/` holds input datasets and
`results/` holds internal research outputs. The root `pyproject.toml` configures
DP-TAL; `environments/baselines/pyproject.toml` and its lockfile configure the
incompatible legacy SmartNoise environment. They are not a combined uv workspace.
Run commands from the repository root using `python -m experiments.<name>`.

## Latent classification experiment

Run from the project root in Ubuntu/WSL. Keep the Linux environment separate
from the Windows `.venv`:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/gen-ai-anonymization"
uv sync --python 3.12
```

Short smoke test:

```bash
uv run python -m experiments.dptal --max-rows 1000 --epochs 1 --noise-multipliers 0 0.5 --synthetic-samples 600 --classifier-max-iter 20 --verbose 2
```

Full dataset, ten epochs per generator, with non-private and DP baselines:

```bash
uv run python -m experiments.dptal --epochs 10 --noise-multipliers 0 0.1 0.3 0.5 1 3 5 --verbose 2
```

Use `--seeds 42 43 44` for repeated runs. Use `--help` for all options.
Classifier iteration limits are independent of the generator epoch count.
`--verbose 0` hides progress, `1` reports phases and epochs, and `2` adds
minibatch progress and native classifier output. Warnings and the final
results table remain visible.

The experiment evaluates Logistic Regression and an MLP in three scenarios:

- Train on real features; test on real `X_test` (fixed preprocessing in both).
- Train on real latents; test on `X_test_latent` (deterministic encoder means in both).
- Train on synthetic latents; test on `X_test_latent` (encoded held-out real records).

The final table, logs and output files identify these combinations using
`evaluation_scenario`. CSV/JSON and grouped summaries also retain separate
`train_representation` and `test_representation` fields. The original
`representation` field remains available in saved results.
Direct real-feature-to-latent or latent-to-real-feature classifier evaluations
are not performed: these are different coordinate spaces and typically have
different dimensions. Matching their dimensions alone would not align them.

All classifiers use the same held-out real test records. Latent classifiers
receive the test records' encoder means without using their labels.
The seeded random split is 60% training, 20% nonmembers for membership
inference, and 20% classification testing. There is no temporal/device grouping or duplicate
removal. Correlated or duplicate flows across partitions can overstate utility.
The encoder and classifier are never fitted on test records.

Original label `1` is positive by default. Verify the dataset's attack mapping;
use `--positive-label 0` if attacks correspond to zero. F1 is binary positive-class
F1, not a macro average.

Each execution writes a timestamped directory under `results/dptal/`,
`results/mia/` or `results/smartnoise/`, depending on the entry point:

- `metrics.csv`: accuracy, binary F1, Macro-F1, positive precision and recall,
  false-positive rate, ROC-AUC, Cohen's kappa, explicit train/test scenarios,
	confusion counts, classifier times and generator privacy budget.
- `summary.csv`: mean, sample standard deviation and seed count by configuration.
	Standard deviation is undefined for a single seed.
- `report.json`: arguments, training histories, timings, split details and
	classifier parameters/convergence warnings. Partial results survive failures.
- `membership_inference.csv`: per-classifier attack results, membership scope,
	calibration/evaluation counts, ROC-AUC, accuracy, F1, TPR, observed FPR,
	advantage (TPR minus FPR), thresholds and timing.
- `membership_summary.csv`: mean, sample standard deviation and seed count for
	completed attacks, grouped by classifier, representation, noise and scope.

Models are not saved by default. `--save-model` stores encoder checkpoints
for internal research; classifiers and record-level latent arrays are not saved.
Short smoke runs may hit classifier iteration limits; inspect `converged`
before drawing conclusions.

## Membership inference audits

The `experiments.mia` entry point runs membership inference for each classifier
and representation. The DP-TAL and SmartNoise comparison entry points do not run MIA.
It uses two black-box, known-label attacks: a threshold on the probability of
the correct task label (equivalently, negative log loss), and a logistic
attacker trained on log loss and prediction entropy. The attacker has access
to a labeled calibration set with known membership; this is a supervised audit,
not an implementation of shadow-model training.

Membership targets depend on the scenario:

- Real features: participation in classifier training.
- Real latents: participation in both generator and classifier training.
- Synthetic latents: participation in original generator training, queried
	through the real-record encoder and synthetic-trained classifier. The
	candidates are original records, not samples from the synthetic training set.

Member candidates come from the 60% training partition. Nonmembers come only
from the separate 20% attack partition, never from the classification test.
Each group is sampled with matching counts for task classes zero and one,
then divided into disjoint calibration and evaluation records. The same seed
keeps candidate selection consistent across classifiers and noise levels.
Insufficient candidates produce an explicit skipped result.

The nonmember calibration scores determine a threshold for a target FPR of 1%.
Strict threshold comparison handles ties conservatively. Evaluation reports
the actual TPR and FPR at that threshold: it does not claim that evaluation FPR
equals 1%. A warning flags samples with fewer than ten expected false positives
at the target FPR. Small smoke tests are not suitable for low-FPR conclusions.

```bash
uv run python -m experiments.mia --mia-max-records 2000 --mia-target-fpr 0.01
```

`--mia-max-records` limits the total candidates per membership group before
the calibration/evaluation split. The MIA entry point rejects `--skip-membership`;
use `experiments.dptal` for a utility-only experiment.
ROC-AUC near 0.5 means these attacks did not distinguish membership well;
it is not proof of privacy. Confidence AUC is not flipped using evaluation
labels. More capable attackers, other signals and repeated seeds may reveal
additional leakage. Identical or correlated flows across partitions remain
a limitation. No per-record confidence scores are saved.

## Research limitations and privacy

The priors are fixed, not learned from traffic. With the same seed, synthetic
samples are identical across noise levels; this prototype evaluates how well
the encoder aligns real data with that reference space, not whether the prior
learns the full traffic distribution. Neighborhood preservation, coverage
and diversity are not evaluated by this script yet.

DP applies to the generator's record-level training mechanism. Noise zero is
non-private and has no finite epsilon claim. Epsilon is reported separately
for each generator: releasing several trained models requires composition.
Secure RNG is disabled for experimentation. Loss logs, metrics, baseline models,
data-dependent metadata and latents directly computed from private records are
not automatically DP-protected releases. Treat all saved results as internal
research artifacts, not anonymized outputs. In particular, classifiers fitted
on real latents and private labels do not automatically inherit the encoder's
training guarantee; their attack results describe the combined pipeline.

## SmartNoise baselines

Independent adapters live in `src/gen_ai_anonymization/baseline/`:

- `mwem.py`: `MWEMBaseline`.
- `mst.py`: `MSTBaseline`.
- `aim.py`: `AIMBaseline`.
- `dpctgan.py`: `DPCTGANBaseline`.

Each instance exposes `fit(data, ...)`, `sample(n_samples)`,
`fit_sample(data, n_samples, ...)` and `get_report()`. Sampling returns a
DataFrame with the original columns. The shared adapter validates budgets,
column hints and output shape, and records fit/sample times and backend warnings.
The `experiments.smartnoise` entry point uses these adapters through an isolated
worker process, so both Opacus versions are never loaded into the same process.

SmartNoise Synth 1.0.8 requires legacy Opacus 0.14, whereas DP-TAL uses
Opacus >=1.5. Do not install the baseline requirements into the DP-TAL
environment or downgrade its Opacus. Create a separate Ubuntu/WSL environment
from the project root; the CPU index avoids CUDA downloads:

```bash
UV_PROJECT_ENVIRONMENT="$HOME/.venvs/gen-ai-baselines" \
	uv sync --project environments/baselines --python 3.12
```

The command above overrides the environment only for that installation command.
Keep the main shell's `UV_PROJECT_ENVIRONMENT` pointing to the DP-TAL environment.
Run the comparison using the main interpreter; the dedicated baseline interpreter
defaults to `$HOME/.venvs/gen-ai-baselines/bin/python` on Linux:

```bash
uv run python -m experiments.smartnoise --epochs 10 --noise-multipliers 1 \
	--baseline-epsilon 3 --baseline-python "$HOME/.venvs/gen-ai-baselines/bin/python"
```

Short end-to-end check of all four methods (not a scientific benchmark):

```bash
uv run python -m experiments.smartnoise --max-rows 300 --epochs 1 \
	--noise-multipliers 0.5 --latent-dim 2 --hidden-dim 8 --anchors 4 \
	--latent-bins 4 --synthetic-samples 128 --batch-size 32 \
	--baseline-epsilon 10 --mwem-iterations 2 --aim-rounds 4 \
	--dpctgan-epochs 1 --classifiers logistic_regression --classifier-max-iter 20
```

MWEM, MST, AIM and DP-CTGAN run by default. Select a subset with
`--smartnoise-methods mwem mst aim`, for example, to omit DP-CTGAN.
Both Logistic
Regression and MLP are evaluated by default, using the existing accuracy, binary
F1, macro-F1, positive precision/recall, FPR, ROC-AUC, Cohen's kappa, confusion
counts, convergence and timing metrics. All methods generate the same requested
number of training samples and use the same held-out test records.

The comparison trains one encoder per seed/noise setting, encodes only the
training partition for SmartNoise fitting, and applies fixed public clipping
and binning to real latents, DP-TAL synthetic latents and test latents alike.
`--latent-bound` (default 4) and `--latent-bins` (default 16) control this shared
representation. SmartNoise synthesizes the joint latent/label table; DP-TAL uses
balanced fixed conditional priors, so generated class frequencies can differ.
This benchmarks latent-space utility, not synthetic tables in original feature
coordinates. Quantization can erase useful signal; the separate DP-TAL
experiment evaluates continuous latents without this discretization.

Worker inputs containing real encodings are temporary and removed after the
worker exits. Metrics, reports and worker logs remain internal diagnostics.
`report.json` includes `baseline_runs`, backend warnings/failures and the protocol;
`smartnoise_seed_*.json` contains backend versions and detailed worker evaluations.
Failures are recorded per method and produce `completed_with_failures` rather
than silently disappearing. Epsilon for encoder training and the requested
SmartNoise epsilon are reported separately; this is not a matched-budget
comparison. Explicit AIM rounds must cover all latent columns plus the label
column, or its initial marginals can exhaust the budget.

Direct adapter API example using a public artificial table, not private traffic
(run in the baseline environment with `PYTHONPATH=src`):

```python
import pandas as pd
from gen_ai_anonymization.baseline import (
	MWEMBaseline, MSTBaseline, AIMBaseline, DPCTGANBaseline,
)

instances = {
	"mwem": MWEMBaseline(epsilon=3, iterations=10, split_factor=2),
	"mst": MSTBaseline(epsilon=3, delta=1e-6),
	"aim": AIMBaseline(epsilon=3, delta=1e-6, rounds=2),
	"dpctgan": DPCTGANBaseline(epsilon=3, epochs=1, batch_size=32, cuda=False),
}
public_data = pd.DataFrame({
	"feature": [0, 1] * 64,
	"Label": [0, 0, 1, 1] * 32,
})
baseline = instances["mst"]
synthetic = baseline.fit_sample(
	public_data, 32, categorical_columns=["feature", "Label"],
)
print(synthetic.head())
print(baseline.get_report())
```

For private tables, supply a SmartNoise `transformer` with appropriate public
domains/bounds, or explicitly budget supported private preprocessing through
`preprocessor_eps`. This allocation must be less than `epsilon` and is deducted
by SmartNoise, not added on top. Column type hints alone do not make
data-dependent domains or bounds public. MWEM, MST and AIM operate on discrete
representations; a large joint domain can exhaust MWEM memory. Splitting columns
reduces that cost but loses dependencies across partitions.

Reports contain requested budgets, not independently verified effective privacy
guarantees (`privacy_budget_verified` is false). The real DP-CTGAN smoke test
emitted a legacy Opacus warning that the returned privacy budget is incorrect
when the sample rate is derived from batch/sample sizes. It also used a
nonsecure RNG and legacy backward hooks. Treat this backend as experimental;
successful fit/sample does not certify privacy or support matched-epsilon
comparisons. Review `backend_warnings`, account for repeated-fit composition,
and do not treat reports or data-dependent metadata as DP-protected releases.