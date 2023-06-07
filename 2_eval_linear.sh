
CUDA_VISIBLE_DEVICES=1 python eval_linear.py --evaluate \
    --patch_size 16 \
    --data_path /paddle/dataset/ILSVRC2012 \
    --pretrained_weights ./out/dino_deitsmall16_pretrain_full_ckp.pdparams \
    --checkpoint_key teacher 
    --pretrained_linear ./out/dino_deitsmall16_linearweights.pdparams \
    --batch_size 32