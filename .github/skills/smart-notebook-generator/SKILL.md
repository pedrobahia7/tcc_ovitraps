---
name: smart-notebook-generator
description: >
  Generate modular, AI-friendly Jupyter notebooks with co-located Python modules for
  this repository. Use this skill whenever the user wants to create a new notebook,
  scaffold a notebook structure, set up a data exploration/preprocessing/modeling/
  analysis notebook, or mentions "smart notebook", "modular notebook", or "notebook
  with modules". Also trigger when the user asks to organize notebook code into
  separate files, or wants a clean notebook template with proper imports and autoreload.
  This skill creates thin orchestration notebooks backed by well-structured Python
  modules, not monolithic notebook code.
---

# Smart Notebook Generator

## Purpose

Create modular Jupyter notebooks that act as **thin orchestration layers** importing from
co-located Python modules. This keeps notebooks clean, testable, and AI-friendly while
separating concerns properly.

## How to Use

Run the generator script bundled with this skill from the repository root:

```bash
conda activate venv_ovitraps
python <skill-path>/scripts/create_smart_notebook.py \
  --name <notebook_name> \
  --title "<notebook_title>" \
  --type <notebook_type> \
  --project-root <project_root_path>
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--name` | Yes | Name of the notebook (used as directory name under `notebooks/`) |
| `--title` | Yes | Human-readable title displayed in the notebook header |
| `--type` | Yes | One of: `exploration`, `preprocessing`, `modeling`, `analysis`, `signal_analysis` |
| `--project-root` | No | Project root path (defaults to current working directory) |
| `--description` | No | Brief description of the notebook's purpose |

### What Gets Created

```
notebooks/
├── <notebook_name>/
│   ├── notebook.ipynb               # Thin orchestration notebook
│   ├── <type-specific modules>.py   # Python modules with function stubs
│   ├── temp/                        # Results & dill sessions (gitignored)
│   └── __init__.py
```

### Shared Utilities — `utils/`

Instead of a separate `notebook/shared/` folder, all cross-notebook utility functions
live in `utils/`. Before creating any new helper function in a notebook module:

1. **Check `utils/project_utils.py`** first — shared project-specific helpers
2. **Check `utils/df_operations.py`** — DataFrame manipulation helpers
3. **Check `utils/NN_preprocessing.py`** — neural-network preprocessing helpers
4. **Check `utils/visualization.py`** — plotting and visualization helpers
5. **Check `utils/generic.py`** — general-purpose notebook helpers
6. **Check `utils/mlflow_utils.py`** and the NN helper modules when the task touches model training or tracking
7. If the function already exists there, import it directly
8. If a function is used across multiple notebooks, add it to the appropriate file in
  `utils/` rather than duplicating it in notebook modules

The generated notebooks already import from `utils/` via path setup and load
parameters from `params.yaml` at the repository root.

### Project Data Conventions

- **Raw data**: Files under `data/raw/`
- **Processed data**: Files under `data/processed/`
- **Other derived data**: Files under `data/` subfolders such as `correlation/`, `geo/`, and `external_data/`
- **Parameters**: `params.yaml` at project root — loaded via `yaml.safe_load()`
- **Data loading**: Prefer reading paths from `params.yaml` instead of hardcoding file locations when possible

### Notebook Types and Generated Modules

**exploration** — For data exploration and profiling:
- `data_loader.py` → `load_data()`, `load_pot_data()`, `load_from_h5()`
- `data_profiler.py` → `generate_profile()`, `summarize_statistics()`, `check_data_quality()`
- `visualizations.py` → `plot_distributions()`, `plot_correlations()`, `plot_time_series()`

**preprocessing** — For data cleaning and feature engineering:
- `cleaner.py` → `clean_data()`, `handle_missing()`, `remove_outliers()`
- `feature_engineering.py` → `create_features()`, `encode_categorical()`, `scale_features()`
- `transformers.py` → `CustomTransformer` class with `fit()`, `transform()`

**modeling** — For model training and evaluation:
- `train.py` → `ModelTrainer` class with `train()`, `save_model()`
- `evaluate.py` → `evaluate_model()`, `compute_metrics()`, `plot_results()`
- `predict.py` → `load_model()`, `predict()`, `batch_predict()`

**analysis** — For general analysis and reporting:
- `data_loader.py` → `load_data()`, `load_from_h5()`
- `analyzer.py` → `run_analysis()`, `compute_statistics()`, `generate_insights()`
- `visualizations.py` → `create_charts()`, `create_summary_table()`, `export_figures()`

**signal_analysis** — For pot signal analysis and failure diagnostics:
- `signal_loader.py` → `load_pot_signals()`, `load_reference_data()`, `load_failure_data()`
- `signal_processor.py` → `compute_resistance()`, `compute_noise()`, `detect_anomalies()`
- `diagnostic_plots.py` → `plot_pot_signals()`, `plot_failure_comparison()`, `plot_diagnostic_timeline()`

### Generated Notebook Structure

Each notebook contains these cells in order:

1. **Markdown header** — Title, purpose, creation date
2. **Setup cell** — `%autoreload 2`, path configuration, repository root detection
3. **Params cell** — Loads `params.yaml` configuration
4. **Imports cell** — Local module imports, `utils/` imports, standard libraries
5. **Execution cells** — Type-specific placeholder cells with markdown headers
6. **Session save cell** — Uses `dill` to save the entire session to `notebooks/<notebook_name>/temp/session.dill`

### Cell Testing

After generating a notebook, **always test each cell for bugs** by running them
sequentially. The only exception is when a cell's execution time is prohibitively high
(e.g., large model training). In that case, skip it and note that it was not tested.

### After Creation

1. Implement the function stubs in the `.py` module files
2. Run the notebook cells sequentially — autoreload ensures changes in `.py` files are picked up immediately
3. Check `utils/` before adding new utility functions — reuse what exists
4. Results and session state are saved to `notebooks/<notebook_name>/temp/` (gitignored)
5. In VS Code, open the generated `.ipynb` file with the Jupyter extension and run the cells in order

### Key Design Principles

- **Notebooks are orchestrators, not containers** — Business logic lives in `.py` files
- **Autoreload is always on** — Edit `.py` files and re-run cells without restarting the kernel
- **Path setup is automatic** — Repository root and `utils/` are added to `sys.path`
- **Shared utilities live in `utils/`** — Check existing functions before creating new ones
- **Configuration via params.yaml** — All data paths and parameters loaded from `params.yaml`
- **Repo-first conventions** — Prefer the existing `notebooks/`, `utils/`, `data/processed/`, and `results/` structure
- **Session persistence via dill** — Save/restore full session state from `temp/` inside the notebook folder
- **Results go to `<notebook_name>/temp/`** — All intermediate outputs are gitignored