#!/bin/bash
# =============================================================================
# setup.sh  —  Automated environment setup script for Amazon Linux / Linux.
#
#     bash setup.sh
# =============================================================================
set -e
cd "$(dirname "$0")"

echo "=============================================================="
echo " Momentum Quant Setup"
echo "=============================================================="

# 0. Safety: ensure no private keys exist in working directory
if find . -name '*.pem' -not -path './.venv/*' | grep -q .; then
    echo
    echo "ERROR: Private key (.pem) found in repository root:" >&2
    find . -name '*.pem' -not -path './.venv/*' >&2
    exit 1
fi

# 1. System packages
echo
echo "--- 1. System Packages ---"
sudo dnf install -y -q python3.12 python3.12-pip git cronie 2>/dev/null || true
echo "python3.12 : $(python3.12 -V 2>/dev/null || python3 -V)"
echo "git        : $(git --version)"

# 2. Python virtual environment
echo
echo "--- 2. Python Virtual Environment ---"
if [ ! -x .venv/bin/python ]; then
    python3.12 -m venv .venv 2>/dev/null || python3 -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists"
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python - <<'PY'
import duckdb, matplotlib, numpy, pandas, pyarrow
print(f"pandas {pandas.__version__}  numpy {numpy.__version__}  "
      f"matplotlib {matplotlib.__version__}  duckdb {duckdb.__version__}  "
      f"pyarrow {pyarrow.__version__}")
PY

# 3. Create required directories
mkdir -p web/data web/charts logs

# 4. Verify data availability
echo
echo "--- 3. Price Data Repository ---"
CSV_COUNT=$(ls data/*.csv 2>/dev/null | wc -l)
echo "$CSV_COUNT CSV files in data/ ($(du -sh data 2>/dev/null | cut -f1))"

echo
echo "=============================================================="
echo " Ready."
echo "=============================================================="
cat <<'EOF'

Execution pipeline:
    bash steps/00_hello_server.sh
    .venv/bin/python steps/01_load_and_plot.py
    .venv/bin/python steps/02_metrics.py
    .venv/bin/python steps/03_backtest.py
    .venv/bin/python steps/04_ic_and_ir.py
    .venv/bin/python steps/05_multifactor.py
    .venv/bin/python steps/06_database_load.py
    .venv/bin/python steps/07_sql_vs_pandas.py

Or build and serve dashboard:
    .venv/bin/python tools/pipeline.py
    .venv/bin/python tools/serve.py

Run test suite:
    .venv/bin/python steps/check.py

EOF
