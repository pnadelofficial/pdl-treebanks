#!/usr/bin/env bash
#SBATCH --job-name=pdl-treebanks-ingest
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL   
#SBATCH --mail-user=peter.nadel@tufts.edu

#
# Usage:
#   1. bash cluster/prepare.sh          (once, on a login node)
#   2. edit the "EDIT ME" sections below
#   3. sbatch cluster/ingest.sbatch

set -euo pipefail

# --- EDIT ME: paths to your clones on the cluster -----------------------
NLP_PIPELINE_DIR="${NLP_PIPELINE_DIR:-/cluster/tufts/perseuslab/pnadel01/perseus/nlp_pipeline}"
PDL_TREEBANKS_DIR="${PDL_TREEBANKS_DIR:-/cluster/tufts/perseuslab/pnadel01/perseus/pdl-treebanks}"
# Must match prepare.sh's STANZA_MODEL_DIR -- see that script's comment on
# why this has to be $NLP_PIPELINE_DIR/stanza_models specifically (the
# server has no model_dir override, only a hardcoded relative default).

# --- EDIT ME: what to ingest ---------------------------------------------
# A single work's proto-pages directory is enough -- doesn't need to be
# the whole corpus. See run_tokenizer.py's layout convention in
# MinimumViablePerseus: proto-pages/<textgroup>/<work>/<version>/.
PROTO_DIR="${PROTO_DIR:-/cluster/tufts/perseuslab/pnadel01/perseus/proto-pages}"
TARGET_LANG="${TARGET_LANG:-la}"                                # grc or la -- NOT named LANG, that's a reserved locale env var
DB_PATH="${DB_PATH:-$PDL_TREEBANKS_DIR/treebanks.db}"
SOURCE="stanza:perseus"
NLP_PORT=8123
# --------------------------------------------------------------------------

NLP_URL="http://127.0.0.1:${NLP_PORT}"
NLP_PID=""

cleanup() {
  if [[ -n "$NLP_PID" ]] && kill -0 "$NLP_PID" 2>/dev/null; then
    echo "Stopping nlp_pipeline server (pid $NLP_PID)"
    kill "$NLP_PID"
  fi
}
trap cleanup EXIT

echo "== starting nlp_pipeline on $NLP_URL =="
(
  # cd here matters, not just for uv: NLPPipeline() (no args, hardcoded in
  # nlp_pipeline.server) resolves its model_dir default ("./stanza_models")
  # relative to this CWD, which is where prepare.sh must have downloaded them.
  cd "$NLP_PIPELINE_DIR"
  uv run uvicorn nlp_pipeline.server:app --host 127.0.0.1 --port "$NLP_PORT" \
    > nlp_pipeline.log 2>&1
) &
NLP_PID=$!

echo "== waiting for nlp_pipeline to become reachable =="
for i in $(seq 1 60); do
  if curl -sf "$NLP_URL/" > /dev/null 2>&1; then
    echo "nlp_pipeline is up (after ${i}0s or less)"
    break
  fi
  if [[ "$i" -eq 60 ]]; then
    echo "nlp_pipeline never came up -- check $NLP_PIPELINE_DIR/nlp_pipeline.log" >&2
    exit 1
  fi
  sleep 10
done

echo "== ingesting $PROTO_DIR (lang=$TARGET_LANG) into $DB_PATH =="
(
  cd "$PDL_TREEBANKS_DIR"
  uv run pdl-treebanks-ingest \
    --proto-dir "$PROTO_DIR" \
    --db "$DB_PATH" \
    --nlp-url "$NLP_URL" \
    --lang "$TARGET_LANG"
)

echo "== aggregating =="
(
  cd "$PDL_TREEBANKS_DIR"
  uv run pdl-treebanks-aggregate --db "$DB_PATH" --source "$SOURCE"
)

echo "== done. db at $DB_PATH =="
