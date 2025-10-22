# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

Ref:
    https://github.com/sihyun-yu/REPA/blob/main/generate.py
"""

import argparse
import gc
import json
import math
import os

from dictdot import dictdot
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from tqdm import tqdm

from models.sit import SiT_models
from models.autoencoder import vae_models
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_encoders, denormalize_latents


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.exp_path is None or args.train_steps is None:
        if rank == 0:
            print("The `exp_path` or `train_steps` is not provided, setting `exp_path` and `train_steps` to default values.")
        args.exp_path = "pretrained/sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k"
        args.train_steps = 400_000

    with open(os.path.join(args.exp_path, "args.json"), "r") as f:
        config = dictdot(json.load(f))

    # Load model:
    if config.vae == "f8d4":
        latent_size = config.resolution // 8
        in_channels = 4
    elif config.vae == "f16d32":
        latent_size = config.resolution // 16
        in_channels = 32
    else:
        raise NotImplementedError()

    # Load the encoder(s) to get the latent dimension(s)
    encoders, _, _ = load_encoders(config.enc_type, "cpu", config.resolution)
    z_dims = [encoder.embed_dim for encoder in encoders] if config.enc_type != 'None' else [0]
    del encoders
    gc.collect()

    block_kwargs = {"fused_attn": config.fused_attn, "qk_norm": config.qk_norm}
    model = SiT_models[config.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=config.num_classes,
        class_dropout_prob=config.cfg_prob,
        z_dims=z_dims,
        encoder_depth=config.encoder_depth,
        bn_momentum=config.bn_momentum,
        **block_kwargs,
    ).to(device)

    exp_name = os.path.basename(args.exp_path)
    train_step_str = str(args.train_steps).zfill(7)
    state_dict = torch.load(
        os.path.join(args.exp_path, "checkpoints", train_step_str +'.pt'),
        map_location=f"cuda:{device}",
    )
    model.load_state_dict(state_dict['ema'])
    model.eval()  # Important! To disable label dropout during sampling

    # Load the VAE and latent stats
    vae = vae_models[config.vae]().to(device)
    if "vae" in state_dict:
        # REPA-E checkpoints, VAE is in the checkpoint
        vae_state_dict = state_dict['vae']

        latents_scale = state_dict["ema"]["bn.running_var"].rsqrt().view(1, in_channels, 1, 1).to(device)
        latents_bias = state_dict["ema"]["bn.running_mean"].view(1, in_channels, 1, 1).to(device)
    else:
        # LDM-training-only checkpoints, VAE checkpoint should be in the config
        vae_state_dict = torch.load(config.vae_ckpt, map_location=f"cuda:{device}")

        latents_stats = torch.load(
            config.vae_ckpt.replace(".pt", "-latents-stats.pt"),
            map_location=f"cuda:{device}"
        )
        latents_scale = latents_stats["latents_scale"].to(device)
        latents_bias = latents_stats["latents_bias"].to(device)
        del latents_stats

    vae.load_state_dict(vae_state_dict)
    vae.eval()

    del state_dict, vae_state_dict
    gc.collect()
    torch.cuda.empty_cache()

    assert args.cfg_scale >= 1.0, "cfg_scale should be >= 1.0"

    sample_folder_dir = (f"{args.sample_dir}/{exp_name}_{train_step_str}_cfg{args.cfg_scale}"
                        f"-{args.guidance_low}-{args.guidance_high}-labelsampling-{args.label_sampling}")
    skip = torch.tensor([False], device=device)
    if rank == 0:
        if os.path.exists(f"{sample_folder_dir}.npz"):
            skip[0] = True
            print(f"Skipping sampling as {sample_folder_dir}.npz already exists.")
        else:
            os.makedirs(sample_folder_dir, exist_ok=True)
            print(f"Saving .png samples at {sample_folder_dir}")

    # Broadcast the skip flag to all processes
    dist.broadcast(skip, src=0)
    if skip.item():
        dist.destroy_process_group()
        return
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.pproc_batch_size
    world_size = dist.get_world_size()

    # Exact class balance: 50_000 images, 1_000 classes => 50 per class
    assert args.num_fid_samples % args.num_classes == 0, \
        f"num_fid_samples ({args.num_fid_samples}) must be divisible by num_classes ({args.num_classes})."
    per_class = args.num_fid_samples // args.num_classes  # 50 when 50k/1k

    # Build a global label schedule with exact counts, then (optionally) shuffle it.
    # IMPORTANT: all ranks must see the same permutation => use a rank-independent seed or broadcast.
    if rank == 0:
        if args.label_sampling == "equal":
            y_all = torch.arange(args.num_classes, device=device).repeat_interleave(per_class)  # [0..999] each repeated 50x
            gen = torch.Generator(device=device).manual_seed(args.global_seed)  # SAME seed across ranks
            y_all = y_all[torch.randperm(y_all.numel(), generator=gen, device=device)]
        elif args.label_sampling == "random":
            y_all = torch.randint(0, args.num_classes, (args.num_fid_samples,), device=device)
        else:
            raise NotImplementedError(f"Unknown label_sampling: {args.label_sampling}")
    else:
        y_all = torch.empty(args.num_fid_samples, device=device, dtype=torch.long)

    # Broadcast the global label schedule to all ranks
    dist.broadcast(y_all, src=0)

    # Equal shard per rank
    labels_per_rank = args.num_fid_samples // world_size   # 12_500 (4 GPUs) or 6_250 (8 GPUs)
    assert args.num_fid_samples % world_size == 0, \
        f"num_fid_samples ({args.num_fid_samples}) must be divisible by world_size ({world_size})."
    start = rank * labels_per_rank
    end = start + labels_per_rank
    y_this_rank = y_all[start:end]                         # shape: (labels_per_rank,)

    # Iteration planning with possible partial last batch (no need to force divisibility by n)
    total_to_make = y_this_rank.numel()                    # exactly 12,500 or 6,250
    iterations = int(math.ceil(total_to_make / n))
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    if rank == 0:
        print(f"Generating exactly {args.num_fid_samples} images across {world_size} GPUs.")
        print(f"Per GPU: {labels_per_rank} images. Classes globally: {args.num_classes} with {per_class} per class.")

    offset = 0  # offset within this rank's shard

    for _ in pbar:
        m = min(n, total_to_make - offset)  # batch size for this iteration (may be < n on the last step)

        # Sample inputs:
        z = torch.randn(m, model.in_channels, latent_size, latent_size, device=device)
        y = y_this_rank[offset : offset + m]

        assert not args.heun or args.mode == "ode", "Heun's method is only available for ODE sampling."

        # Sample images:
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
        )
        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            samples = vae.decode(denormalize_latents(samples, latents_scale, latents_bias)).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
            ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = start + offset + i
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        offset += m

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed params
    parser.add_argument("--global-seed", type=int, default=0)

    # precision params
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving params
    parser.add_argument("--sample-dir", type=str, default="samples")

    # ckpt params
    parser.add_argument("--exp-path", type=str, default=None, help="Path to the specific experiment directory.")
    parser.add_argument("--train-steps", type=str, default=None, help="The checkpoint of the model to sample from.")

    # number of samples
    parser.add_argument("--pproc-batch-size", type=int, default=128)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False,
                        help="Use Heun's method for ODE sampling.")
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    parser.add_argument(
        "--label-sampling",
        type=str,
        choices=["random", "equal"],
        default="equal",
        help="Choose how to sample class labels when generating images.",
    )

    args = parser.parse_args()
    main(args)
