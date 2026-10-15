#!/usr/bin/env bash
# One-shot environment setup. Run from the project root:
#
#     bash setup.sh
#
# Creates the virtual environment, installs everything, and runs the tests.
# Does NOT log you in to Hugging Face; that step needs your token and has to
# be you.

set -euo pipefail

cd "$(dirname "$0")"

echo "==> Python version"
python3 --version

echo
echo "==> Creating virtual environment (.venv)"
if [ -d .venv ]; then
    echo "    .venv already exists, reusing it"
else
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo
echo "==> Upgrading pip"
pip install -q -U pip

echo
echo "==> Installing the project and its dependencies (a few minutes)"
pip install -q -e ".[dev]"

echo
echo "==> Running the test suite"
pytest

cat <<'DONE'

===========================================================================
Environment is ready.

IMPORTANT: this script ran in its own shell. In YOUR terminal you still need:

    source .venv/bin/activate

Do that in every new terminal window, or nothing below will find its
packages. Your prompt shows (.venv) when it is active.

Next, in order:

    huggingface-cli login     # paste your read token
    python check.py           # tokenizers, base model, language codes

If check.py looks clean:

    python -m tokpremium.survey.run --limit 20    # smoke test
    python -m tokpremium.survey.run               # real run
    python -m tokpremium.survey.figures

Results land in results/, figures in results/figures/.
===========================================================================
DONE
