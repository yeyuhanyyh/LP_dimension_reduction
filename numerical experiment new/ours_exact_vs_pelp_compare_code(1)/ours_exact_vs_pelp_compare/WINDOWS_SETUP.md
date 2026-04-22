# Windows + VSCode setup

## 1. Install Python

Recommended: install 64-bit Python 3.11 or 3.12 from python.org.

During installation, check:

- Add python.exe to PATH
- pip
- py launcher

Then open a new PowerShell and run:

```powershell
py --version
python --version
py -0p
```

If `python` opens Microsoft Store or is not found, open Windows Settings and search for
"App execution aliases"; turn off the aliases for `python.exe` and `python3.exe`.

## 2. Install VSCode extensions

Install Visual Studio Code, then install these extensions:

- Python, by Microsoft
- Pylance, by Microsoft

Open this folder in VSCode: `File -> Open Folder`.

## 3. Create a virtual environment

Open VSCode Terminal: `Terminal -> New Terminal`.

PowerShell commands:

```powershell
cd path\to\ours_exact_vs_pelp_compare
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

If PyTorch installation is slow or fails, use the CPU wheel command from PyTorch:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy scipy pandas matplotlib
```

## 4. Select interpreter in VSCode

Press `Ctrl+Shift+P`, run:

```text
Python: Select Interpreter
```

Choose:

```text
.\.venv\Scripts\python.exe
```

## 5. Run the code

Quick run:

```powershell
python compare_ours_exact_vs_pelp_fixedX.py --quick --run_sharedp --run_fcnn --make_plots --out_dir results_quick
```

Medium run:

```powershell
python compare_ours_exact_vs_pelp_fixedX.py --n_vars 60 --n_cons 16 --dstar 8 --n_train 50 --n_val 8 --n_test 15 --k_list 4,8,12 --epochs 12 --batch_size 4 --run_sharedp --run_fcnn --make_plots --out_dir results_medium
```

If the neural baselines are too slow, omit `--run_sharedp --run_fcnn`, or reduce `--epochs`.
