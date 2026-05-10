import numpy as np
import torch
import argparse
import os
import cv2
import time
from tqdm import tqdm
from datasets import create_dataset
from models import build_model
from main import get_args_parser
from util.checkpoint import normalize_state_dict_keys

parser = argparse.ArgumentParser('DFECrack Inference', parents=[get_args_parser()])
args = parser.parse_args()


if __name__ == '__main__':
    print(f"Dataset: {args.dataset_path}")
    print(f"Checkpoint: {args.test_checkpoint}")
    print(f"Output Dir: {args.test_results_dir}")

    args.phase = 'test'
    args.batch_size = 1
    device = torch.device(args.device)
    test_dl = create_dataset(args)
    model, criterion = build_model(args)
    state_dict = torch.load(args.test_checkpoint, map_location=device)
    model.load_state_dict(normalize_state_dict_keys(state_dict["model"]))
    model.to(device)
    print("Model loaded successfully!")

    dataset_name = args.dataset_path.split('/')[-1]
    save_root = os.path.join(args.test_results_dir, f"{dataset_name}_results")
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    pbar = tqdm(total=len(test_dl), desc="Testing")
    total_time = 0
    total_frames = 0

    with torch.no_grad():
        model.eval()
        for batch_idx, (data) in enumerate(test_dl):
            x = data["image"]
            target = data["label"]
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()

            start_time = time.time()
            out = model(x)
            inference_time = time.time() - start_time
            total_time += inference_time
            total_frames += 1

            pbar.set_description(f"FPS: {1.0/inference_time:.1f}")
            pbar.update(1)

            target = target[0, 0, ...].cpu().numpy()
            out = out[0, 0, ...].cpu().numpy()
            root_name = data["A_paths"][0].split("/")[-1][0:-4]
            target = 255 * (target / np.max(target))
            out = 255 * (out / np.max(out))

            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)

    pbar.close()
    avg_fps = total_frames / total_time if total_time > 0 else 0
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Results saved to: {save_root}")
