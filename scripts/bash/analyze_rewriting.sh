set -e # Exit on error
set -u # Treat unset variables as an error

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -u src/evaluation/rewriting_analysis.py