import os
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
from engine import train_one_epoch
from models import build_model
from datasets import create_dataset
import cv2
from eval.evaluate import eval
from util.logger import get_logger
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_args_parser():
    parser = argparse.ArgumentParser('DFECrack', add_help=False)

    parser.add_argument('--BCELoss_ratio', default=0.83, type=float)
    parser.add_argument('--DiceLoss_ratio', default=0.17, type=float)
    parser.add_argument('--dataset_path', default="../../data/datasets/TUT",
                        help='Root directory path for dataset')
    parser.add_argument('--batch_size_train', type=int, default=4)
    parser.add_argument('--batch_size_test', type=int, default=1)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--output_dir', default='./checkpoints/weights')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_mode', type=str, default='crack')
    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--num_threads', default=1, type=int)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--load_width', type=int, default=512)
    parser.add_argument('--load_height', type=int, default=512)
    parser.add_argument('--test_checkpoint', type=str,
                        default='./checkpoints/weights/checkpoint_best.pth')
    parser.add_argument('--test_results_dir', type=str, default='./results/results_test')

    # Model architecture
    parser.add_argument('--freq_module_types', type=str,
                        default='identity,identity,identity,identity',
                        help='Frequency module type per layer. Released model uses identity for all layers.')
    parser.add_argument('--decoder_type', type=str, default='fasf_simple',
                        choices=['fasf_simple', 'fasf_full'])
    parser.add_argument('--fusion_type', type=str, default='acsf',
                        help='Feature fusion type. Released model uses acsf.')

    # Boundary supervision
    parser.add_argument('--use_boundary', action='store_true',
                        help='Enable boundary-weighted supervision')
    parser.add_argument('--boundary_boost', type=float, default=2.0)
    return parser


def main(args):
    checkpoints_path = "./checkpoints"
    cur_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    dataset_name = (args.dataset_path).split('/')[-1]
    process_folder_path = os.path.join(checkpoints_path, cur_time + '_' + dataset_name)
    args.phase = 'train'
    if not os.path.exists(process_folder_path):
        os.makedirs(process_folder_path)

    log_train = get_logger(process_folder_path, 'train')
    log_test = get_logger(process_folder_path, 'test')
    log_eval = get_logger(process_folder_path, 'eval')

    log_train.info("args -> " + str(args))
    print(f"Dataset: {args.dataset_path}")
    print(f"BCELoss_ratio: {args.BCELoss_ratio}, DiceLoss_ratio: {args.DiceLoss_ratio}")

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)
    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    dataset_size = len(train_dataLoader)
    print('The number of training images = %d' % dataset_size)

    param_dicts = [{"params": [p for n, p in model.named_parameters()], "lr": args.lr}]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    output_dir = args.output_dir + '/' + cur_time + '_' + dataset_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)

    print("Start training!")
    start_time = time.time()
    max_mIoU = 0
    max_Metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch}/{args.epochs-1} =====")
        train_loss = train_one_epoch(model, criterion, train_dataLoader, optimizer, epoch, args, log_train)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # Test
        results_path = cur_time + '_' + dataset_name
        save_root = f'./results/{results_path}/results_{epoch}'
        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)
        pbar = tqdm(total=len(test_dl), desc="Testing")

        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        with torch.no_grad():
            model.eval()
            for batch_idx, (data) in enumerate(test_dl):
                x = data["image"]
                target = data["label"]
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                out = model(x)
                target = target[0, 0, ...].cpu().numpy()
                out = out[0, 0, ...].cpu().numpy()
                root_name = data["A_paths"][0].split("/")[-1][0:-4]
                target = 255 * (target / np.max(target))
                out = 255 * (out / np.max(out))
                cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
                cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)
                pbar.update(1)
        pbar.close()

        # Evaluate
        metrics = eval(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        if max_mIoU < metrics['mIoU']:
            max_Metrics = metrics
            max_mIoU = metrics['mIoU']
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, output_dir / 'checkpoint_best.pth')
            print(f"  ** New best mIoU: {max_mIoU:.4f} at epoch {epoch}")

        print(f"  Best mIoU: {max_Metrics['mIoU']} (epoch {max_Metrics['epoch']})")

    for key, value in max_Metrics.items():
        log_eval.info(f"{key} -> {value}")

    total_time = time.time() - start_time
    print(f'Total training time: {datetime.timedelta(seconds=int(total_time))}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DFECrack', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
