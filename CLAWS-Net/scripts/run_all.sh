#!/usr/bin/env bash
# Full pipeline: preprocess -> SimCLR pre-train -> 5-fold CLAWS-Net + ablations -> report.
set -e
LIDC=data/lidc; LUNGX=data/lungx
python scripts/extract_lidc.py --out $LIDC
# python scripts/prepare_lungx.py --annotations lungx_annotations.csv --out $LUNGX   # if LUNGx available
MAN="$LIDC/manifest.csv"; [ -f $LUNGX/manifest.csv ] && MAN="$MAN $LUNGX/manifest.csv"
python scripts/pretrain_simclr.py --manifest $MAN --epochs 100
EXT=""; [ -f $LUNGX/manifest.csv ] && EXT="--external $LUNGX/manifest.csv"
for F in 0 1 2 3 4; do
  python scripts/train.py --manifest $LIDC/manifest.csv $EXT --fold $F --pretrained checkpoints/simclr_encoder.pt --out results/claws_net_fold$F
  python scripts/train.py --manifest $LIDC/manifest.csv $EXT --fold $F --out results/abl_no_simclr_fold$F
  python scripts/train.py --manifest $LIDC/manifest.csv $EXT --fold $F --pretrained checkpoints/simclr_encoder.pt --no_ctf --out results/abl_no_ctf_fold$F
  python scripts/train.py --manifest $LIDC/manifest.csv $EXT --fold $F --pretrained checkpoints/simclr_encoder.pt --no_awdf --out results/abl_no_awdf_fold$F
  for V in axial coronal sagittal; do
    python scripts/train.py --manifest $LIDC/manifest.csv $EXT --fold $F --pretrained checkpoints/simclr_encoder.pt --views $V --out results/abl_single_${V}_fold$F
  done
done
python scripts/make_report.py results
