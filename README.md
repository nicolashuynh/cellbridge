# CellBRIDGE

Code for our paper **CellBRIDGE: Learning Cellular
Trajectories via Interaction-Aware Alignment** (ICML 2026).

CellBRIDGE aligns cellular snapshots with an interaction-aware optimal transport
objective, then uses the learned couplings for marginal interpolation and
flow-matching velocity pushforward. The repository includes lightweight AnnData
inputs, Hydra configurations, reproduction scripts, and figure notebooks.

## Setup

This project uses `uv` and Python 3.13.

```bash
uv python install 3.13
uv sync
```

Check that the package imports correctly:

```bash
uv run python -c "import cellbridge; print('ok')"
```

To make the `uv` environment available as a VS Code/Jupyter notebook kernel:

```bash
uv run python -m ipykernel install --user \
  --name cellbridge \
  --display-name "Python (cellbridge)"
```


## Data

The repository ships compact AnnData files in `data/` for the camera-ready
experiment scripts. Each file keeps only the fields used by the pipelines:
start/eval/end cells, the dataset domain column, `.obsm["X_pca_whiten"]`, and
`.layers["cp10k"]`.

| Dataset config | Included input | Snapshot labels |
| --- | --- | --- |
| `cancer` | `data/cancer_lite.h5ad` | `SIGAA5`, `SIGAC5`, `SIGAF5` |
| `immune` | `data/immune_lite.h5ad` | `0`, `2`, `6` |
| `light` | `data/light_lite.h5ad` | `0`, `1`, `4` |
| `embryo` | `data/embryo_lite.h5ad` | `Day 00-03`, `Day 06-09`, `Day 12-15` |



## Quickstart

Run an alignment sweep for one dataset:

```bash
uv run python src/cellbridge/pipeline/sweep_alpha_align.py \
  inputs=cancer \
  'align.alphas=[0.0,0.5,1.0]'
```

The command writes an `artifacts/` directory containing shared CCI/cost artifacts
and one `alpha_*/coupling.npy` folder per alpha. Use the printed artifact path to
evaluate marginal interpolation:

```bash
uv run python src/cellbridge/pipeline/sample_interpolation.py \
  inputs=cancer \
  folder_artifacts=/path/to/artifacts/alpha_0.500
```

Train and sample a flow-matching velocity model from an existing alpha folder:

```bash
uv run python src/cellbridge/pipeline/train_flow.py \
  inputs=cancer \
  alpha='"0.500"' \
  inputs.folder_artifacts=/path/to/artifacts

uv run python src/cellbridge/pipeline/sample_with_velocity.py \
  inputs=cancer \
  alpha='"0.500"' \
  inputs.folder_artifacts=/path/to/artifacts
```

Flow-matching jobs use the W&B logger in offline mode by default, so they can
run without a W&B login. To sync runs online, pass `wandb.offline=false`.

## Using a New Dataset

The easiest path is to store the start, evaluation, and end snapshots in one
AnnData file and add a Hydra input config under `conf/inputs/`.

Your AnnData file should contain:

- an `.obs` column identifying the snapshot or timepoint for each cell,
- a precomputed embedding in `.obsm`, usually `.obsm["X_pca_whiten"]`,
- gene names in `.var_names`,
- an expression layer for CCI construction, usually `.layers["cp10k"]`.

Create `conf/inputs/my_dataset.yaml`:

```yaml
mode: from_h5ad
h5ad_path: data/my_dataset.h5ad
domain_key: timepoint
start_label: "day0"
eval_label: "day3"
end_label: "day7"
t_eval: 0.43
pair_mode: lr
name: my_dataset
subsample: false
rep_transform: null
folder_artifacts: experiments/my_dataset/align_sweep/latest/artifacts
```

Set `t_eval` to the relative location of the evaluation snapshot between the
start and end snapshots. For example, if the snapshots are days 0, 3, and 7,
then `t_eval = 3 / 7`.

Choose `pair_mode` based on the interaction pairs you want to use. The paper
experiments use the curated modes `liana_cancer`, `liana_immune`, `liana_light`,
and `liana_embryo`; the ablation scripts also use `random_lr`. For a new
dataset, start with `lr` to use OmniPath ligand-receptor pairs filtered to
genes in `.var_names`. If your dataset needs a custom or species-specific
ligand-receptor set, add it in `src/cellbridge/cci/lr_pairs_config.py` and
expose it from `src/cellbridge/cci/pair_extraction.py`.

If your embedding key is not `X_pca_whiten`, override `rep.obsm_key` when you
run the pipeline. If your expression layer is not `cp10k`, override `cci.layer`;
if the expression matrix is in `.X`, use `cci.layer=null`.

Run the alignment sweep:

```bash
EXPERIMENTS_ROOT=$PWD/experiments \
uv run python src/cellbridge/pipeline/sweep_alpha_align.py \
  inputs=my_dataset \
  rep.obsm_key=X_pca_whiten \
  cci.layer=cp10k \
  'align.alphas=[0.0,0.5,1.0]'
```

The alignment command prints the generated `artifacts/` directory. Use that path
for downstream sampling and flow matching:

```bash
uv run python src/cellbridge/pipeline/sample_interpolation.py \
  inputs=my_dataset \
  folder_artifacts=/path/to/artifacts/alpha_0.500

uv run python src/cellbridge/pipeline/train_flow.py \
  inputs=my_dataset \
  alpha='"0.500"' \
  inputs.folder_artifacts=/path/to/artifacts

uv run python src/cellbridge/pipeline/sample_with_velocity.py \
  inputs=my_dataset \
  alpha='"0.500"' \
  inputs.folder_artifacts=/path/to/artifacts
```

For repeated runs, update `folder_artifacts` in `conf/inputs/my_dataset.yaml` to
the generated artifact root, or keep passing `inputs.folder_artifacts=...` on the
command line.

## Experiment Scripts

These scripts are the higher-level entry points used for reproduction runs.
They resolve the repository from their own file locations, but many defaults
still point at experiment roots under `/mnt/data/nvth2.data/experiments`;
inspect the script headers for path and parallelism overrides before rerunning
large jobs.

### Path Overrides

The shell scripts discover the repository with `SCRIPT_DIR`/`CELLBRIDGE_ROOT`,
so the checkout location usually does not need to be edited. The paths that do
need to match your machine are the experiment output roots and the precomputed
artifact roots consumed by later stages.

Alignment scripts use Hydra's run directory. By default this is under
`/mnt/data/nvth2.data/experiments`, but you can override it without editing the
script:

```bash
EXPERIMENTS_ROOT=$PWD/experiments \
RUN_GROUP=my_run_name \
bash scripts/marginal_interpolation/cfm/"V1 Light.sh"
```

This writes to:

```text
${EXPERIMENTS_ROOT}/${inputs.name}/align_sweep/${RUN_GROUP}/artifacts/
```

If `RUN_GROUP` is omitted, Hydra uses a timestamped folder. The alignment
scripts print `Artifacts directory: ...`; downstream commands expect either
that root `artifacts/` directory or one of its `alpha_*` subdirectories:

```bash
# Marginal interpolation expects a single alpha folder.
uv run python src/cellbridge/pipeline/sample_interpolation.py \
  inputs=light \
  folder_artifacts=/path/to/artifacts/alpha_0.500

# Flow training/sampling expect the artifact root containing alpha_* folders.
uv run python src/cellbridge/pipeline/train_flow.py \
  inputs=light \
  alpha='"0.500"' \
  inputs.folder_artifacts=/path/to/artifacts

uv run python src/cellbridge/pipeline/sample_with_velocity.py \
  inputs=light \
  alpha='"0.500"' \
  inputs.folder_artifacts=/path/to/artifacts
```

For the parallel velocity scripts, provide one artifact root per dataset if you
are not using the checked-in defaults:

```bash
CANCER_ARTIFACTS=/path/to/cancer/artifacts \
LIGHT_ARTIFACTS=/path/to/light/artifacts \
IMMUNE_ARTIFACTS=/path/to/immune/artifacts \
bash scripts/velocity_pushforward/cfm/run_parallel_cfm.sh
```

`scripts/velocity_pushforward/mfm/run_parallel_geopath.sh` and
`scripts/velocity_pushforward/sf2m/run_parallel_sf2m.sh` accept the same
`CANCER_ARTIFACTS`, `LIGHT_ARTIFACTS`, `IMMUNE_ARTIFACTS`, and
`EMBRYO_ARTIFACTS` environment variables. Give them the base CellBRIDGE
artifact root; the scripts create/use sibling `_mfm` or `_sf2m` artifact
folders. `scripts/velocity_pushforward/uot/run_parallel_uot_fm.sh` stores its
defaults in the `ARTIFACTS_FOLDERS` block near the top of the script, so update
that block if your UOT artifacts live elsewhere.

The ablation wrapper also accepts path and run-size overrides:

```bash
EXPERIMENTS_ROOT=$PWD/experiments \
RUN_ID=test_run \
OUTPUT_DIR=$PWD/scripts/ablations/results/test_run \
DATASETS="cancer light" \
SEEDS="42" \
bash scripts/ablations/ablations.sh
```


| Script or folder | What it runs |
| --- | --- |
| `scripts/marginal_interpolation/` | Balanced CFM and unbalanced UOT coupling generation plus interpolation sampling. |
| `scripts/velocity_pushforward/cfm/run_parallel_cfm.sh` | CellBRIDGE CFM train+sample runs for `cancer`, `light`, and `immune` across alphas and seeds. |
| `scripts/velocity_pushforward/uot/run_parallel_uot_fm.sh` | Flow-matching velocity pushforward using UOT-generated artifacts. |
| `scripts/velocity_pushforward/mfm/run_parallel_geopath.sh` | MFM/GeoPath train+sample runs using `_mfm` artifact directories. |
| `scripts/velocity_pushforward/sf2m/run_parallel_sf2m.sh` | SF2M training with stochastic sampling. |
| `scripts/ablations/ablations.sh` | Shuffle, random ligand-receptor, and metacell ablations. |
| `scripts/knockout_experiment/reproduce_normalized_percent_decrease.sh` | Recreates the normalized percent-decrease knockout validation figure from lightweight result artifacts. |



## Configuration

Hydra configuration files live in `conf/`.

| Path | Contents |
| --- | --- |
| `conf/inputs/` | Dataset labels, AnnData paths, evaluation times, ligand-receptor pair modes, and artifact roots. |
| `conf/cci/` | CCI construction from AnnData expression layers and cluster assignments. |
| `conf/cci_transform/` | CCI matrix transforms. |
| `conf/cluster/` | Identity and Leiden clustering pipelines. |
| `conf/sweep_align_multi_channel*.yaml` | Balanced, unbalanced, and ablation alignment sweeps. |
| `conf/flow_matching.yaml` | Flow-matching training, dataloaders, optimizer, and periodic sampling evaluation. |
| `conf/sampling_marginal.yaml` | Coupling-based marginal interpolation evaluation. |
| `conf/sampling_velocity.yaml` | Velocity-model pushforward sampling and evaluation. |
| `conf/geopath/` | Optional GeoPath bridge configuration. |
| `conf/model/`, `conf/trainer/`, `conf/wandb/` | Neural network, PyTorch Lightning, and logging settings. |

## Outputs

Alignment sweeps write to Hydra run directories such as:

```text
${EXPERIMENTS_ROOT}/${inputs.name}/align_sweep/<run>/artifacts/
```

Typical alignment artifacts include `labels_X.npy`, `labels_Y.npy`,
`CCI_X.npy`, `CCI_Y.npy`, `cost_C.npy`, `cost_D1.npy`, `cost_D2.npy`,
`pairs.csv`, `sweep_index.tsv`, and per-alpha folders with `coupling.npy`,
`artifacts.json`, and `summary.txt`.

Sampling jobs write `metrics.json` and `samples.pt` under `sample_marginal/` or
`sample_velocity/`. Flow-matching runs also write Lightning checkpoints and
logger outputs under each alpha/seed folder.

The lightweight knockout validation bundle is stored under
`artifacts/knockout_experiment/normalized_percent_decrease_repro/`.
The regenerated knockout validation PDF is written to
`figures/knockout_experiment/normalized_percent_decrease_all_conditions.pdf`.

Rendered figures are in `figures/`; the corresponding post-processing notebooks
are in:

- `notebooks/ablations/`
- `notebooks/marginal_interpolation/`
- `notebooks/synthetic_experiment/`
- `notebooks/velocity_pushforward/`

## Citation

If you use this code, please cite the accompanying CellBRIDGE paper.

```bibtex
@inproceedings{estevez2026cellbridge,
  title = {CellBRIDGE: Learning Cellular Trajectories via Interaction-Aware Alignment},
  author = {Silas Ruhrberg Estévez and Nicolas Huynh and Tennison Liu and Roderik M. Kortlever and Gerard I. Evan and David L. Bentley and Mihaela van der Schaar},
  booktitle = {Proceedings of the 43rd International Conference on Machine learning},
  year = {2026}
}
```
