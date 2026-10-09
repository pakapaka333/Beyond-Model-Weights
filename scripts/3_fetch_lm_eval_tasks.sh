#!/usr/bin/env bash
# Copy the MMLU and IFEval tasks of lm_eval 0.4.8 (the version of dOPSD's lm-eval
# fork) into the fork, which ships neither. IFEval must live inside the package
# because its yaml refers to `utils.process_results` as a package submodule.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=0.4.8
DEST="third_party/dOPSD/eval/Dream/eval_instruct/lm_eval/tasks"
TASKS=(mmlu ifeval)

missing=0
for t in "${TASKS[@]}"; do [ -d "${DEST}/${t}" ] || missing=1; done
if [ "${missing}" -eq 0 ] && [ "${1:-}" != "--force" ]; then
    echo "[fetch_lm_eval_tasks] ${TASKS[*]} already in ${DEST}"
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
python3 - "${VERSION}" "${WORK}" <<'PY'
import hashlib, json, pathlib, sys, urllib.request
version, work = sys.argv[1], pathlib.Path(sys.argv[2])
meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/lm_eval/{version}/json", timeout=120))
sdist = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
data = urllib.request.urlopen(sdist["url"], timeout=600).read()
if hashlib.sha256(data).hexdigest() != sdist["digests"]["sha256"]:
    sys.exit("sha256 mismatch for the lm_eval sdist")
(work / sdist["filename"]).write_bytes(data)
PY
tar xzf "${WORK}/lm_eval-${VERSION}.tar.gz" -C "${WORK}"
SRC="${WORK}/lm_eval-${VERSION}/lm_eval/tasks"
for t in "${TASKS[@]}"; do
    rm -rf "${DEST:?}/${t}"
    cp -r "${SRC}/${t}" "${DEST}/${t}"
    echo "[fetch_lm_eval_tasks] ${t} -> ${DEST}/${t}"
done
