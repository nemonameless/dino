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
import os
import argparse
import json
from pathlib import Path

import paddle
from paddle import nn
import paddle.distributed as dist
from vision_transformer import vit_small
import utils
from models import LinearClassifier
from paddle.vision import transforms
import os
from utils import MetricLogger
from dataset import ImageNet2012Dataset
import json
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def eval_linear(args):
    dist.init_parallel_env()

    # ============ building network ... ============
    # only support vit_s8 and vit_s16 currently
    model = vit_small(
        patch_size=args.patch_size, num_classes=0
    )
    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    print(f"vit_s{args.patch_size} has build!")
    model.eval()

    # load weight to evaluate
    utils.load_pretrained_weights(model, args.checkpoint_key, args.pretrained_weights)

    linear_clf = LinearClassifier(embed_dim, args.num_labels)
    linear_clf = paddle.DataParallel(linear_clf) ### DataParallel

    # ============ preparing data ... ============
    val_transform = transforms.Compose([
        transforms.Resize(size=256, interpolation='bicubic'),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = ImageNet2012Dataset(args.data_path, mode="val", transform=val_transform)
    val_loader = paddle.io.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_shared_memory=True,
    )

    if args.evaluate:
        utils.load_linear_clf_weights(linear_clf, args.pretrained_linear)
        test_stats = valid(val_loader, model, linear_clf, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ImageNet2012Dataset(args.data_path, mode="train", transform=train_transform)
    sampler = paddle.io.DistributedBatchSampler(dataset_train, args.batch_size, shuffle=True)
    train_loader = paddle.io.DataLoader(
        dataset_train,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        use_shared_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    base_lr = args.lr * (args.batch_size * dist.get_world_size() / 256)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=args.epochs, eta_min=0)
    optimizer = paddle.optimizer.Momentum(
        parameters=linear_clf.parameters(),
        learning_rate=scheduler,
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    if args.resume_path != "":
        utils.restart_from_checkpoint(
            os.path.join(args.resume_path),
            run_variables=to_restore,
            state_dict=linear_clf,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(f"resume from epoch {to_restore['epoch']}")

    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_stats = train(model, linear_clf, optimizer, train_loader, epoch,
                            args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = valid(val_loader, model, linear_clf, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            print(log_stats)

        if dist.get_rank() == 0:
            with open(os.path.join(args.output_dir, "train_linear_log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_clf.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            paddle.save(save_dict, os.path.join(args.output_dir, "dino_deitsmall16_linearweights.pdparams"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.4f}".format(acc=best_acc))


def train(model, linear_clf, optimizer, loader, epoch, n, avgpool):
    linear_clf.train()
    metrics_logger = MetricLogger(" ")
    metrics_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = 'Epoch: [{}]'.format(epoch)
    for (image, y) in metrics_logger.log_every(loader, 20, header):

        # forward
        with paddle.no_grad():
            intermediate_output = model.get_intermediate_layers(image, n)
            output = paddle.concat([x[:, 0] for x in intermediate_output], axis=-1)
            if avgpool:
                output = paddle.concat(
                    (output.unsqueeze(-1), paddle.mean(intermediate_output[-1][:, 1:], axis=1).unsqueeze(-1)),
                    axis=-1
                )
                output = output.reshape((output.shape[0], -1))
        output = linear_clf(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, y)

        # compute the gradients
        optimizer.clear_grad()
        loss.backward()

        # step
        optimizer.step()

        if dist.is_initialized():
            paddle.device.cuda.synchronize()
        metrics_logger.update(loss=loss.item())
        metrics_logger.update(lr=optimizer._learning_rate.last_lr)
    # gather the stats from all processes
    if dist.is_initialized():
        metrics_logger.synchronize_between_processes()
    print("Averaged stats:", metrics_logger)
    return {k: meter.global_avg for k, meter in metrics_logger.meters.items()}


@paddle.no_grad()
def valid(valid_loader, model, linear_clf, n_last_blocks, avgpool_patchtokens):
    linear_clf.eval()
    metrics_logger = MetricLogger(" ")
    header = 'Test:'
    for images, y in metrics_logger.log_every(valid_loader, 20, header):

        # forward
        with paddle.no_grad():
            intermediate_output = model.get_intermediate_layers(images, n_last_blocks)
            output = paddle.concat([x[:, 0] for x in intermediate_output], axis=-1)
            if avgpool_patchtokens:
                output = paddle.concat(
                    (output.unsqueeze(-1), paddle.mean(intermediate_output[-1][:, 1:], axis=1).unsqueeze(-1)),
                    axis=-1
                )
                output = output.reshape((output.shape[0], -1))

        output = linear_clf(output)
        loss = nn.CrossEntropyLoss()(output, y)

        if args.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, y, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, y, topk=(1,))

        batch_size = images.shape[0]
        metrics_logger.update(loss=loss.item())
        metrics_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if args.num_labels >= 5:
            metrics_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    # gather the stats from all processes
    if dist.is_initialized():
        metrics_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metrics_logger.meters.items()}


class LinearClassifier(nn.Layer):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        normal_(self.linear.weight)
        zeros_(self.linear.bias)

    def forward(self, x):
        # flatten
        x = x.reshape((x.shape[0], -1))

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=bool,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='./dino_deitsmall16_pretrain_full_ckp.pdparams',
                        type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--pretrained_linear', default='', type=str, help='Path to pretrained linear clf weights.')
    parser.add_argument('--resume_path', default='', type=str, help='Path to checkpoint.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.003, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size', default=32, type=int, help='total batch-size')
    parser.add_argument('--data_path', default='/paddle/dataset/ILSVRC2012', type=str)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    eval_linear(args)
