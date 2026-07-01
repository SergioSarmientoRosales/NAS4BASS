# Troubleshooting

## `You must provide --n-gen when --early-stop is disabled`

`main.py` requires `--n-gen` unless you enable `--early-stop`.

Use:

```bash
python main.py --eval model_based --search nsga3 --n-gen 2
```

## `No .pkl files were found inside 'models'`

Model-based evaluation needs serialized surrogate models in `models/*.pkl`.

Options:

- Make sure the `models/` folder exists.
- Use `--models-dir` to point to another folder.
- Use `--eval zero_cost` if you want to run without surrogate models.

## TensorFlow Import Or GPU Errors

First verify TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
```

If this fails, reinstall TensorFlow in a clean virtual environment. GPU support depends on your OS, driver, CUDA, cuDNN, and TensorFlow version.

If the traceback mentions `numpy.core.umath`, `numpy.core._multiarray_umath`, or a module compiled with NumPy 1.x, NumPy 2.x is probably installed. Recreate the environment and reinstall the pinned dependencies:

```powershell
python -m pip install --upgrade pip
pip install --force-reinstall -r requirements.txt
```

## Windows Opens The Microsoft Store Instead Of Python

On some Windows systems, `python` points to a Microsoft Store alias instead of the real Python installation.

Check:

```powershell
where.exe python
```

If it points to `WindowsApps\python.exe`, either disable the Python app execution alias in Windows settings or call the real Python executable directly, for example:

```powershell
C:\Users\<USER>\AppData\Local\Programs\Python\Python312\python.exe --version
```

## Pickle Compatibility Errors

The surrogate models in `models/*.pkl` were serialized with specific machine-learning libraries. If loading fails, install versions close to those in `requirements.txt`, especially:

- `scikit-learn`
- `xgboost`
- `joblib`

## Benchmark Outputs Missing

`Plots.py` expects outputs produced by `Benchmark.py`.

Run:

```bash
python Benchmark.py
python Plots.py
```

## Generated Files Appearing In Git

Generated outputs should stay out of commits. Check:

```bash
git status --ignored
```

If needed, remove generated files from the working tree or update `.gitignore`.
