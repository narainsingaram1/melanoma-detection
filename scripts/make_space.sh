#!/usr/bin/env bash
# Assemble a self-contained Hugging Face Space in build/hf_space.
#   bash scripts/make_space.sh
#   hf upload <your-username>/melanoma-detection build/hf_space . --repo-type space
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/hf_space
for f in models/model_info.json models/ood_stats.npz models/classifier.pth; do
  [ -f "$f" ] || { echo "missing $f: run python -m scripts.build_release --run <run dir> first" >&2; exit 1; }
done
rm -rf "$OUT" && mkdir -p "$OUT"/{src,models,results/ablation,assets}
cp deploy/hf_space/README.md deploy/hf_space/requirements.txt app.py "$OUT"/
cp src/__init__.py src/constants.py src/utils.py src/inference.py src/transfer_model.py src/gradcam.py "$OUT"/src/
cp models/model_info.json models/ood_stats.npz models/classifier.pth "$OUT"/models/
cp -R assets/samples "$OUT"/assets/
cp results/leakage_report.json "$OUT"/results/
[ -f results/ablation/summary.md ] && cp results/ablation/summary.md "$OUT"/results/ablation/
echo "Space assembled in $OUT ($(du -sh "$OUT" | cut -f1))"
