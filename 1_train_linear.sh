
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" \
    eval_linear.py --patch_size 16 \
    --epochs 100 \
    --data_path /paddle/dataset/ILSVRC2012 \
    --pretrained_weights ./out/dino_deitsmall16_pretrain.pdparams \
    --checkpoint_key teacher \
    --batch_size 32 \
    --num_workers 4 \
    --output_dir ./out
