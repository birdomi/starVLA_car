#!/bin/bash
# One fetch_agibot.py process per line of a list (login nodes kill a process after 2 h of CPU time):
#   fetch_driver.sh tars <root>/logs/list_w0.txt   |   fetch_driver.sh pr <root>/logs/list_pr.txt
TOOLS=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-/scratch/t509a04/data/AgiBotWorld2026}
PY=${PY:-/home01/t509a04/envs/starVLA/bin/python}
mode=$1; list=$2
while read -r item; do
  [ -z "${item}" ] && continue
  if [ "${mode}" = tars ]; then
    OMP_NUM_THREADS=1 ${PY} -u ${TOOLS}/fetch_agibot.py tars --root ${ROOT} --only "${item}" 2>&1 | grep -v -e jemalloc -e "unauthenticated requests"
  else
    OMP_NUM_THREADS=1 ${PY} -u ${TOOLS}/fetch_agibot.py pr --root ${ROOT} --sub "${item}" 2>&1 | grep -v -e jemalloc -e "unauthenticated requests" | tr '\r' '\n' | grep -v "Fetching"
  fi
done < "${list}"
echo "[$(date)] driver ${mode} ${list} finished"
