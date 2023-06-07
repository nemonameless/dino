#python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" main_dino.py \
#CUDA_VISIBLE_DEVICES=1 python main_dino.py \
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" main_dino.py \
    --patch_size 16 --epochs 100 \
    --data_path /paddle/dataset/ILSVRC2012 \
    --batch_size 64 \
    --num_workers 4 \
    --output_dir ./out
