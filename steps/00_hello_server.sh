#!/bin/bash
# =============================================================================
# steps/00_hello_server.sh  —  Server environment and host verification
#
# Run:   bash steps/00_hello_server.sh
#
# Prints system specifications, OS details, Python environment status,
# and market time zones to verify execution environment.
# =============================================================================

echo "=== ENVIRONMENT ==="
echo "user      : $(whoami)"
echo "hostname  : $(hostname)"
echo "directory : $(pwd)"

echo
echo "=== SYSTEM SPECIFICATIONS ==="
echo "os        : $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "kernel    : $(uname -r)"
echo "arch      : $(uname -m)"
echo "cpu cores : $(nproc)"
echo "memory    : $(free -h | awk '/^Mem:/ {print $3 " used of " $2}')"
echo "disk      : $(df -h / | awk 'NR==2 {print $3 " used of " $2 " (" $5 " full)"}')"
echo "uptime    : $(uptime -p 2>/dev/null || echo n/a)"

echo
echo "=== TIME & MARKET CLOCKS ==="
echo "server    : $(date -u '+%Y-%m-%d %H:%M:%S %Z')"
echo "new york  : $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"

echo
echo "=== RUNTIME TOOLCHAIN ==="
echo "python    : $(python3.12 -V 2>&1 || python3 -V 2>&1)"
echo "git       : $(git --version 2>&1)"
if [ -x .venv/bin/python ]; then
    echo "venv      : $(.venv/bin/python -V 2>&1)"
    .venv/bin/python - <<'PY'
import pandas, numpy, matplotlib, duckdb
print(f"  pandas {pandas.__version__}  numpy {numpy.__version__}  "
      f"matplotlib {matplotlib.__version__}  duckdb {duckdb.__version__}")
PY
else
    echo "venv      : NOT BUILT YET -- run  bash setup.sh"
fi

echo
echo "=== DATA REPOSITORY ==="
echo "csv files : $(ls data/*.csv 2>/dev/null | wc -l)  (expect 109)"
echo "data size : $(du -sh data 2>/dev/null | cut -f1)"
echo
