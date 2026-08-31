#!/usr/bin/env bash
#
# Usage: edit the paths below, then:
#   bash cluster/prepare.sh
set -euo pipefail

# --- EDIT ME: paths to your clones on the cluster -----------------------
NLP_PIPELINE_DIR="${NLP_PIPELINE_DIR:-/cluster/tufts/perseuslab/pnadel01/perseus/nlp_pipeline}"
PDL_TREEBANKS_DIR="${PDL_TREEBANKS_DIR:-/cluster/tufts/perseuslab/pnadel01/perseus/pdl-treebanks}"
# nlp_pipeline.server always builds NLPPipeline() with no arguments, which
# defaults model_dir to "./stanza_models" relative to its CWD -- there's no
# env var hook for this (only the importable NLPPipeline class takes
# model_dir directly, which is what this script uses below). So models
# must live at exactly this path, and ingest.sbatch must `cd` into
# NLP_PIPELINE_DIR before starting the server, for the two to agree on
# where to find them.
STANZA_MODEL_DIR="${STANZA_MODEL_DIR:-$NLP_PIPELINE_DIR/stanza_models}"
# Which languages you'll actually ingest -- add/remove as needed.
LANGUAGES=("grc" "la")
# --------------------------------------------------------------------------

module load uv

echo "== uv sync: nlp_pipeline =="
(cd "$NLP_PIPELINE_DIR" && uv sync)

echo "== uv sync: pdl-treebanks =="
(cd "$PDL_TREEBANKS_DIR" && uv sync)

echo "== downloading Stanza models into $STANZA_MODEL_DIR =="
mkdir -p "$STANZA_MODEL_DIR"
for lang in "${LANGUAGES[@]}"; do
  echo "  -- $lang --"
  # Constructing NLPPipeline with download_method's default (REUSE_RESOURCES)
  # downloads whatever's missing and reuses whatever's already there -- the
  # same call the batch job itself will make later, just with internet
  # available this time so it can actually fetch anything missing.
  (cd "$NLP_PIPELINE_DIR" && uv run python -c "
from nlp_pipeline.pipeline import NLPPipeline
pipeline = NLPPipeline()
pipeline.analyze_str('test', lang='$lang') 
print('  $lang OK')
")
done

echo "== done. Stanza models cached in $STANZA_MODEL_DIR =="
echo "Now submit the batch job: sbatch cluster/ingest.sbatch"
