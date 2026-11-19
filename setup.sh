#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "  Student Python Environment Setup"
echo "=========================================="
echo

if [[ ! -f requirements.txt ]]; then
  echo "ERROR: requirements.txt was not found."
  echo "Keep this setup script in the same folder as requirements.txt."
  exit 1
fi

PYTHON_CMD=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_CMD="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_CMD" ]]; then
  echo "ERROR: Python was not found."
  echo "Install Python 3 from https://www.python.org/downloads/ and run this script again."
  exit 1
fi

echo "Using $("$PYTHON_CMD" --version)"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_CMD" -m venv .venv
else
  echo "Existing .venv environment found. Reusing it."
fi

echo
echo "Upgrading pip ..."
".venv/bin/python" -m pip install --upgrade pip

echo
echo "Installing packages from requirements.txt ..."
".venv/bin/python" -m pip install -r requirements.txt

echo
echo "Checking package imports ..."
".venv/bin/python" -c "import mlflow, sklearn, pandas, numpy; print('MLflow:', mlflow.__version__); print('scikit-learn:', sklearn.__version__); print('pandas:', pandas.__version__); print('NumPy:', numpy.__version__)"

echo
echo "SETUP COMPLETE!"
echo
echo "In PyCharm, select this interpreter:"
echo "$(pwd)/.venv/bin/python"
echo "Usually: Settings/Preferences > Project > Python Interpreter"
