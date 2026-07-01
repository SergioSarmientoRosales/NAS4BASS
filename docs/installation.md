# Installation

This project is a Python research-code repository. It uses TensorFlow/Keras for model construction and zero-cost predictor evaluation, scikit-learn and XGBoost for surrogate predictors, and pandas/matplotlib for analysis outputs.

## Python Version

The local source folder was inspected with Python 3.12.2 available on the machine. Use Python 3.12 when possible unless your TensorFlow installation requires a different compatible version.

## Windows PowerShell

```powershell
cd NAS4BASS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

then activate the environment again.

## Linux Or macOS

```bash
cd NAS4BASS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## GPU Notes

Zero-cost metrics and model construction can run on CPU for small tests, but full experiments are faster on a GPU-enabled TensorFlow installation.

TODO: document the exact CUDA, cuDNN, GPU, and driver versions used for the paper experiments once the environment is finalized.

## Verify The Installation

```bash
python -m compileall .
python main.py --help
```

If `python main.py --help` fails while importing TensorFlow, check the TensorFlow installation first.
