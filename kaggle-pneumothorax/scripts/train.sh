set -e

export PROJECT_ROOT="/root/repo/siim-help/kaggle-pneumothorax"
for FOLD in {1..1}
do
PYTHONPATH="${PROJECT_ROOT}" \
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0 \
python "${PROJECT_ROOT}"/src/train.py "${PROJECT_ROOT}"/configs/resnet34_768_unet.yaml --fold=$FOLD
done
