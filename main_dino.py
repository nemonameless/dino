# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import paddle
from paddle import nn
import paddle.distributed as dist

import paddle.nn.functional as F
from paddle.vision import transforms
import paddle.optimizer
from paddle.vision.datasets import ImageFolder
from vision_transformer import vit_small, DINOHead

import utils


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    parser.add_argument('--load_args', type=str, default="",
                        help="load args from 'args.json'")

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_small'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=bool,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=bool, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--base_lr', type=float, default=0.00075, help="Base value of learning rate.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/paddle/dataset/ILSVRC2012', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--resume_path', default="", type=str,help='If resume from checkpoint.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser.parse_args()


def train_dino(args):
    dist.init_parallel_env()
    utils.fix_random_seeds(args.seed)

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = paddle.io.DistributedBatchSampler(dataset, args.batch_size, shuffle=True, drop_last=True)
    data_loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        use_shared_memory=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks  ============
    # only support vit_s8 and vit_s16 currently
    student = vit_small(
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,  # stochastic depth
    )
    teacher = vit_small(patch_size=args.patch_size)
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    ckt = paddle.load("initial_weight/torch_init_weight.pdparams")
    student.set_dict(ckt)

    # vit_s8 and vit_s16 are batch norm free models. here, we don't check bn
    teacher_without_ddp = teacher
    student = paddle.DataParallel(student)
    # teacher and student start with the same weights
    teacher_without_ddp.load_dict(student.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.stop_gradient = True
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    )

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    clip = paddle.nn.ClipGradByGlobalNorm(args.clip_grad) if args.clip_grad != 0 else None

    opt = paddle.optimizer.AdamW(learning_rate=args.base_lr, parameters=params_groups, grad_clip=clip)
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = paddle.amp.GradScaler(init_loss_scaling=65536.0, incr_every_n_steps=2000)

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size * dist.get_world_size() / 256,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        args.resume_path,
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=opt,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training!")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, opt, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': opt.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        if dist.get_rank() == 0:
            path = os.path.join(args.output_dir, 'dino_deitsmall16_pretrain_full_ckp_epoch_lastest.pdparams')
            paddle.save(save_dict, path)
        if epoch == args.epochs or epoch % args.saveckp_freq == 0:
            if dist.get_rank() == 0:
                path = os.path.join(args.output_dir, f'dino_deitsmall16_pretrain_full_ckp_epoch_{epoch}.pdparams')
                paddle.save(save_dict, path)

        epoch_end_time = time.time()
        used_time = f"{(epoch_end_time - epoch_start_time)/3600:.6f} h"

        # ============ write train log ============
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'used_time': used_time}
        print(log_stats)
        if dist.get_rank() == 0:
            log_path = os.path.join(args.output_dir, "train_backbone_log_pd.txt")
            with open(log_path, "a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(
        student, teacher, teacher_without_ddp, dino_loss, data_loader,
        optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
        fp16_scaler, args):
    metric_logger = utils.MetricLogger(" ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate
        # compute global training iteration
        cur_iter_num = len(data_loader) * epoch + it
        optimizer.set_lr(lr_schedule[cur_iter_num])
        for i, param_group in enumerate(optimizer._param_groups):
            if i == 0:
                # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[cur_iter_num]

        # teacher and student forward passes + compute dino loss
        with paddle.amp.auto_cast(fp16_scaler is not None):
            teacher_output = teacher(images[0][:2])  # only the 2 global views pass through the teacher
            student_output = student(images[0])      # all views pass through the student
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.clear_grad()

        if fp16_scaler is None:
            loss.backward()
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad != 0:
                fp16_scaler.unscale_(optimizer)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with paddle.no_grad():
            m = momentum_schedule[cur_iter_num]  # momentum parameter
            for ema_v, model_v in zip(teacher_without_ddp.state_dict().values(), student.state_dict().values()):
                paddle.assign(m * ema_v + (1-m) * model_v, ema_v)

        # logging
        paddle.device.cuda.synchronize() ###
        metric_logger.update(train_lr=optimizer._learning_rate) # param_groups[0]["lr"]?
        metric_logger.update(train_wd=optimizer._param_groups[0]["weight_decay"])
        metric_logger.update(train_loss=loss.item())

    # gather the stats from all processes
    if dist.is_initialized():
        metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Layer):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", paddle.zeros((1, out_dim)))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        warm_up_teacher_sch = np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs)
        teacher_sch = np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        self.teacher_temp_schedule = np.concatenate([warm_up_teacher_sch, teacher_sch]).astype("float32")

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, axis=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = paddle.sum(-q * F.log_softmax(student_out[v], axis=-1), axis=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @paddle.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = paddle.sum(teacher_output, axis=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            utils.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            utils.RandomGrayscale(0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation="bicubic"),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation="bicubic"),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation="bicubic"),
            flip_and_color_jitter,
            utils.GaussianBlur(0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == "__main__":
    args = get_args_parser()
    train_dino(args)
