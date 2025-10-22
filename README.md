<h1 align="center"> REPA-E: Unlocking VAE for End-to-End Tuning of Latent Diffusion Transformers </h1>

<p align="center">
  <a href="https://www.linkedin.com/in/xingjian-leng" target="_blank">Xingjian&nbsp;Leng</a><sup>1*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://1jsingh.github.io/" target="_blank">Jaskirat&nbsp;Singh</a><sup>1*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://hou-yz.github.io/" target="_blank">Yunzhong&nbsp;Hou</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://people.csiro.au/X/Z/Zhenchang-Xing/" target="_blank">Zhenchang&nbsp;Xing</a><sup>2</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?hl=en&user=Y2GtJkAAAAAJ&view_op=list_works" target="_blank">Saining&nbsp;Xie</a><sup>3</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=vNHqr3oAAAAJ&hl=en" target="_blank">Liang&nbsp;Zheng</a><sup>1</sup>&ensp;
</p>

<p align="center">
  <sup>1</sup> Australian National University &emsp; <sup>2</sup>Data61-CSIRO &emsp; <sup>3</sup>New York University &emsp; <br>
  <sub><sup>*</sup>Project Leads &emsp;</sub>
</p>

<p align="center">
  <a href="https://End2End-Diffusion.github.io">🌐 Project Page</a> &ensp;
  <a href="https://huggingface.co/REPA-E">🤗 Models</a> &ensp;
  <a href="https://arxiv.org/abs/2504.10483">📃 Paper</a> &ensp;
  <br><br>
  <!-- <a href="https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=repa-e-unlocking-vae-for-end-to-end-tuning-of"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/repa-e-unlocking-vae-for-end-to-end-tuning-of/image-generation-on-imagenet-256x256" alt="PWC"></a> -->
</p>

![](assets/vis-examples.jpg)

## Overview
We address a fundamental question: ***Can latent diffusion models and their VAE tokenizer be trained end-to-end?*** While training both components jointly with standard diffusion loss is observed to be ineffective — often degrading final performance — we show that this limitation can be overcome using a simple representation-alignment (REPA) loss. Our proposed method, **REPA-E**, enables stable and effective joint training of both the VAE and the diffusion model.

![](assets/overview.jpg)

**REPA-E** significantly accelerates training — achieving over **17×** speedup compared to REPA and **45×** over the vanilla training recipe. Interestingly, end-to-end tuning also improves the VAE itself: the resulting **E2E-VAE** provides better latent structure and serves as a **drop-in replacement** for existing VAEs (e.g., SD-VAE), improving convergence and generation quality across diverse LDM architectures. Our method achieves state-of-the-art FID scores on ImageNet 256×256: **1.12** with CFG and **1.69** without CFG.

<section id="news">
    <h2>News</h2>
    <div class="row">
        <div> 📅 <b>[Oct 2025]</b> 🚨 Released <a href="https://end2end-diffusion.github.io/repa-e-t2i/">REPA-E for T2I</a> 🚨 — a family of End-to-End Tuned VAEs:
            <ul style="margin-top: 8px; margin-bottom: 4px;">
                <li><b>Family of end-to-end tuned VAEs</b>:
                    <ul style="margin-top: 4px; margin-bottom: 4px;">
                        <li>T2I VAEs: <a href="https://huggingface.co/REPA-E/e2e-flux-vae">FLUX-VAE</a>, <a href="https://huggingface.co/REPA-E/e2e-sd3.5-vae">SD-3.5-VAE</a>, <a href="https://huggingface.co/REPA-E/e2e-qwenimage-vae">Qwen-Image-VAE</a></li>
                        <li>ImageNet VAEs: <a href="https://huggingface.co/REPA-E/e2e-sdvae-hf">SD-VAE</a>, <a href="https://huggingface.co/REPA-E/e2e-invae-hf">IN-VAE</a>, <a href="https://huggingface.co/REPA-E/e2e-vavae-hf">VA-VAE</a></li>
                    </ul>
                </li>
                <li><b>End-to-end training generalizes to T2I</b>: E2E-VAEs achieve better T2I generation quality across multiple resolutions (256×256, 512×512) compared to their standard VAE counterparts, without requiring additional representation alignment losses</li>
                <li><b>SOTA results on ImageNet 256×256</b>: FID <b>1.12</b> with CFG and <b>1.69</b> without CFG</li>
                <!-- <li><b>Improved latent space structure</b> with enhanced semantic spatial details compared to standard VAEs</li> -->
                <li>All models available as <b>Hugging Face-compatible AutoencoderKL</b> checkpoints — load directly with <code>diffusers</code> API, no custom wrapper needed</li>
            </ul>
        </div>
    </div>
    <div class="row">
        <div> 📅 <b>[Jun 2025]</b> REPA-E accepted at ICCV 2025!</div>
    </div>
    <div class="row">
        <div> 📅 <b>[Apr 2025]</b> Paper, code, and pretrained models available on <a href="https://github.com/End2End-Diffusion/REPA-E">GitHub</a> and <a href="https://huggingface.co/REPA-E">Hugging Face</a>.</div>
    </div>
    </section>

<!-- ## Updates -->
<h2 align="left" style="color:#ff000d">🆕 Model Releases: Hugging Face Compatible VAEs</h2>

> We are excited to release REPA-E E2E-VAE models as fully Hugging Face AutoencoderKL checkpoints, ready to use with diffusers out of the box. This release includes both our ImageNet-trained VAEs and our new text-to-image VAEs, all optimized for seamless integration into existing pipelines.

<h3 align="left">ImageNet VAEs (SD-VAE, IN-VAE, VA-VAE)</h3>

Our ImageNet-trained VAEs (E2E-SD-VAE, E2E-IN-VAE, E2E-VA-VAE) are now available as Hugging Face-compatible **AutoencoderKL** checkpoints. Previously, these models required loading through custom classes in our REPA-E repository. This new release provides direct compatibility with the `diffusers` API — no extra code or custom wrapper needed.

| Model | Link |
|---|---|
| **E2E-SDVAE-HF** | 🤗 [REPA-E/e2e-sdvae-hf](https://huggingface.co/REPA-E/e2e-sdvae-hf) |
| **E2E-INVAE-HF** | 🤗 [REPA-E/e2e-invae-hf](https://huggingface.co/REPA-E/e2e-invae-hf) |
| **E2E-VAVAE-HF** | 🤗 [REPA-E/e2e-vavae-hf](https://huggingface.co/REPA-E/e2e-vavae-hf) |

<h3 align="left">Text-to-Image VAEs</h3>

Building on the success of our ImageNet VAEs, we now release **REPA-E for T2I** — a family of end-to-end tuned VAEs specifically designed to supercharge text-to-image generation. These models consistently outperform standard VAEs across all T2I benchmarks (COCO-30K, DPG-Bench, GenAI-Bench, GenEval, MJHQ-30K), enable faster training convergence, without requiring additional representation alignment losses.

| Model | Link |
|---|---|
| **E2E-FLUX-VAE** | 🤗 [REPA-E/e2e-flux-vae](https://huggingface.co/REPA-E/e2e-flux-vae) |
| **E2E-SD-3.5-VAE** | 🤗 [REPA-E/e2e-sd3.5-vae](https://huggingface.co/REPA-E/e2e-sd3.5-vae) |
| **E2E-Qwen-Image-VAE** | 🤗 [REPA-E/e2e-qwenimage-vae](https://huggingface.co/REPA-E/e2e-qwenimage-vae) |

### ⚡️ Quickstart 
```python
from diffusers import AutoencoderKL

# Load end-to-end tuned VAE (ImageNet VAE example)
vae = AutoencoderKL.from_pretrained("REPA-E/e2e-vavae-hf").to("cuda")

# Or load a text-to-image VAE
vae = AutoencoderKL.from_pretrained("REPA-E/e2e-flux-vae").to("cuda")

# Use in your pipeline with vae.encode(...) / vae.decode(...)
```

## Getting Started
### 1. Environment Setup
To set up our environment, please run:

```bash
git clone https://github.com/REPA-E/REPA-E.git
cd REPA-E
conda env create -f environment.yml -y
conda activate repa-e
```

### 2. Prepare the training data
Download and extract the training split of the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index) dataset. Once it's ready, run the following command to preprocess the dataset:

```bash
python preprocessing.py --imagenet-path /PATH/TO/IMAGENET_TRAIN
```

Replace `/PATH/TO/IMAGENET_TRAIN` with the actual path to the extracted training images.

### 3. Train the REPA-E model

To train the REPA-E model, you first need to download the following pre-trained VAE checkpoints:
- [🤗 **SD-VAE (f8d4)**](https://huggingface.co/REPA-E/sdvae): Derived from the [Stability AI SD-VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse), originally trained on [Open Images](https://storage.googleapis.com/openimages/web/index.html) and fine-tuned on a subset of [LAION-2B](https://laion.ai/blog/laion-5b/).
- [🤗 **IN-VAE (f16d32)**](https://huggingface.co/REPA-E/invae): Trained from scratch on [ImageNet-1K](https://www.image-net.org/) using the [latent-diffusion](https://github.com/CompVis/latent-diffusion) codebase with our custom architecture.
- [🤗 **VA-VAE (f16d32)**](https://huggingface.co/REPA-E/vavae): Taken from [LightningDiT](https://github.com/hustvl/LightningDiT), this VAE is a visual tokenizer aligned with vision foundation models during reconstruction training. It is also trained on [ImageNet-1K](https://www.image-net.org/) for high-quality tokenization in high-dimensional latent spaces.

Recommended directory structure:
```
pretrained/
├── invae/
├── sdvae/
└── vavae/
```

Once you've downloaded the VAE checkpoint, you can launch REPA-E training with:
```bash
accelerate launch train_repae.py \
    --max-train-steps=400000 \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --data-dir="data" \
    --output-dir="exps" \
    --batch-size=256 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/2" \
    --checkpointing-steps=50000 \
    --loss-cfg-path="configs/l1_lpips_kl_gan.yaml" \
    --vae="f8d4" \
    --vae-ckpt="pretrained/sdvae/sdvae-f8d4.pt" \
    --disc-pretrained-ckpt="pretrained/sdvae/sdvae-f8d4-discriminator-ckpt.pt" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --vae-align-proj-coeff=1.5 \
    --bn-momentum=0.1 \
    --exp-name="sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k"
```
<details>
  <summary>Click to expand for configuration options</summary>

Then this script will automatically create the folder in `exps` to save logs and checkpoints. You can adjust the following options:

- `--output-dir`: Directory to save checkpoints and logs
- `--exp-name`: Experiment name (a subfolder will be created under `output-dir`)
- `--vae`: Choose between `[f8d4, f16d32]`
- `--vae-ckpt`: Path to a provided or custom VAE checkpoint
- `--disc-pretrained-ckpt`: Path to a provided or custom VAE discriminator checkpoint
- `--models`: Choose from `[SiT-B/2, SiT-L/2, SiT-XL/2, SiT-B/1, SiT-L/1, SiT-XL/1]`. The number indicates the patch size. Select a model compatible with your VAE architecture.
- `--enc-type`: `[dinov2-vit-b, dinov2-vit-l, dinov2-vit-g, dinov1-vit-b, mocov3-vit-b, mocov3-vit-l, clip-vit-L, jepa-vit-h, mae-vit-l]`
- `--encoder-depth`: Any integer from 1 up to the full depth of the selected encoder
- `--proj-coeff`: REPA-E projection coefficient for SiT alignment (float > 0)
- `--vae-align-proj-coeff`: REPA-E projection coefficient for VAE alignment (float > 0)
- `--bn-momentum`: Batchnorm layer momentum (float)

</details>

### 4. Use REPA-E Tuned VAE (E2E-VAE) for Accelerated Training and Better Generation
This section shows how to use the REPA-E fine-tuned VAE (E2E-VAE) in latent diffusion training. E2E-VAE acts as a drop-in replacement for the original VAE, enabling significantly accelerated generation performance. You can either download a pre-trained VAE or extract it from a REPA-E checkpoint.

**Step 1**: Obtain the fine-tuned VAE from REPA-E checkpoints:
- **Option 1**: Download pre-trained REPA-E VAEs directly from Hugging Face:
    - [🤗 **E2E-SDVAE**](https://huggingface.co/REPA-E/e2e-sdvae)
    - [🤗 **E2E-INVAE**](https://huggingface.co/REPA-E/e2e-invae)
    - [🤗 **E2E-VAVAE**](https://huggingface.co/REPA-E/e2e-vavae)
  
    Recommended directory structure:
    ```
    pretrained/
    ├── e2e-sdvae/
    ├── e2e-invae/
    └── e2e-vavae/
    ```
- **Option 2**: Extract the VAE from a full REPA-E checkpoint manually:
    ```bash
    python save_vae_weights.py \
        --repae-ckpt pretrained/sit-repae-vavae/checkpoints/0400000.pt \
        --vae-name e2e-vavae \
        --save-dir exps
    ```

**Step 2**: Cache latents to enable fast training:
```bash
accelerate launch --num_machines=1 --num_processes=8 cache_latents.py \
    --vae-arch="f16d32" \
    --vae-ckpt-path="pretrained/e2e-vavae/e2e-vavae-400k.pt" \
    --vae-latents-name="e2e-vavae" \
    --pproc-batch-size=128
```

**Step 3**: Train the SiT generation model using the cached latents:

```bash
accelerate launch train_ldm_only.py \
    --max-train-steps=4000000 \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --data-dir="data" \
    --batch-size=256 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/1" \
    --checkpointing-steps=50000 \
    --vae="f16d32" \
    --vae-ckpt="pretrained/e2e-vavae/e2e-vavae-400k.pt" \
    --vae-latents-name="e2e-vavae" \
    --learning-rate=1e-4 \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --output-dir="exps" \
    --exp-name="sit-xl-1-dinov2-b-enc8-ldm-only-e2e-vavae-0.5-4m"
```

For details on the available training options and argument descriptions, refer to [Section 3](#3-train-the-repa-e-model).

### 5. Generate samples and run evaluation
You can generate samples and save them as `.npz` files using the following script. Simply set the `--exp-path` and `--train-steps` corresponding to your trained model (REPA-E or Traditional LDM Training).

```bash
torchrun --nnodes=1 --nproc_per_node=8 generate.py \
    --num-fid-samples 50000 \
    --path-type linear \
    --mode sde \
    --num-steps 250 \
    --cfg-scale 1.0 \
    --guidance-high 1.0 \
    --guidance-low 0.0 \
    --exp-path pretrained/sit-ldm-e2e-vavae \
    --train-steps 4000000
```

<details>
  <summary>Click to expand for sampling options</summary>

You can adjust the following options for sampling:
- `--path-type linear`: Noise schedule type, choose from `[linear, cosine]`
- `--mode`: Sampling mode, `[ode, sde]`
- `--num-steps`: Number of denoising steps
- `--cfg-scale`: Guidance scale (float ≥ 1), setting it to 1 disables classifier-free guidance (CFG)
- `--guidance-high`: Upper guidance interval (float in [0, 1])
- `--guidance-low`: Lower guidance interval (float in [0, 1], must be < `--guidance-high`)
- `--exp-path`: Path to the experiment directory
- `--train-steps`: Training step of the checkpoint to evaluate

</details>

You can then use the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute image generation quality metrics, including gFID, sFID, Inception Score (IS), Precision, and Recall.

### Quantitative Results
Tables below report generation performance using gFID on 50k samples, with and without classifier-free guidance (CFG). We compare models trained end-to-end with **REPA-E** and models using a frozen REPA-E fine-tuned VAE (**E2E-VAE**). Lower is better. All linked checkpoints below are hosted on our [🤗 Hugging Face Hub](https://huggingface.co/REPA-E). To reproduce these results, download the respective checkpoints to the `pretrained` folder and run the evaluation script as detailed in [Section 5](#5-generate-samples-and-run-evaluation).

#### A. End-to-End Training (REPA-E)
| Tokenizer | Generation Model | Epochs | gFID-50k ↓ | gFID-50k (CFG) ↓ |
|:---------|:----------------|:-----:|:----:|:---:|
| [**SD-VAE<sup>*</sup>**](https://huggingface.co/REPA-E/sdvae) | [**SiT-XL/2**](https://huggingface.co/REPA-E/sit-repae-sdvae) | 80 | 4.07 | 1.67<sup>a</sup> |
| [**IN-VAE<sup>*</sup>**](https://huggingface.co/REPA-E/invae) | [**SiT-XL/1**](https://huggingface.co/REPA-E/sit-repae-invae) | 80 | 4.09 | 1.61<sup>b</sup> |
| [**VA-VAE<sup>*</sup>**](https://huggingface.co/REPA-E/vavae) | [**SiT-XL/1**](https://huggingface.co/REPA-E/sit-repae-vavae) | 80 | 4.05 | 1.73<sup>c</sup> |

\* The "Tokenizer" column refers to the initial VAE used for joint REPA-E training. The final (jointly optimized) VAE is bundled within the generation model checkpoint. 

<details>
  <summary>Click to expand for CFG parameters</summary>
  <ul>
    <li><strong>a</strong>: <code>--cfg-scale=2.2</code>, <code>--guidance-low=0.0</code>, <code>--guidance-high=0.65</code></li>
    <li><strong>b</strong>: <code>--cfg-scale=1.8</code>, <code>--guidance-low=0.0</code>, <code>--guidance-high=0.825</code></li>
    <li><strong>c</strong>: <code>--cfg-scale=1.9</code>, <code>--guidance-low=0.0</code>, <code>--guidance-high=0.825</code></li>
  </ul>
</details>

---

#### B. Traditional Latent Diffusion Model Training (Frozen VAE)
| Tokenizer | Generation Model | Method | Epochs | gFID-50k ↓ | gFID-50k (CFG) ↓ |
|:------|:---------|:----------------|:-----:|:----:|:---:|
| SD-VAE | SiT-XL/2 | SiT | 1400 | 8.30 | 2.06 |
| SD-VAE | SiT-XL/2 | REPA | 800 | 5.84 | 1.28 |
| VA-VAE | LightningDiT-XL/1 | LightningDiT | 800 | 2.05 | 1.25 |
| [**E2E-VAVAE (Ours)**](https://huggingface.co/REPA-E/e2e-vavae) | [**SiT-XL/1**](https://huggingface.co/REPA-E/sit-ldm-e2e-vavae) | REPA | 800 | **1.69** | **1.12**<sup>†</sup> |

In this setup, the VAE is kept frozen and only the generator is trained. Models using our E2E-VAE (fine-tuned via REPA-E) consistently outperform baselines such as SD-VAE and VA-VAE, achieving state-of-the-art performance when incorporating the REPA alignment objective.

**Note**: The results for the last three rows (REPA, LightningDiT, and E2E-VAE) are obtained using the class-balanced sampling protocol (50 images per class).

<details>
    <summary>Click to expand for CFG parameters</summary>
<ul>
    <li><strong>†</strong>: <code>--cfg-scale=2.4</code>, <code>--guidance-low=0.0</code>, <code>--guidance-high=0.73</code></li>
</ul>
</details>

## Acknowledgement
This codebase builds upon several excellent open-source projects, including:
- [1d-tokenizer](https://github.com/bytedance/1d-tokenizer)
- [edm2](https://github.com/NVlabs/edm2)
- [LightningDiT](https://github.com/hustvl/LightningDiT)
- [REPA](https://github.com/sihyun-yu/REPA)
- [Taming-Transformers](https://github.com/CompVis/taming-transformers)

We sincerely thank the authors for making their work publicly available.

## 📚 Citation
If you find our work useful, please consider citing:

```bibtex
@article{leng2025repae,
  title={REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers},
  author={Xingjian Leng and Jaskirat Singh and Yunzhong Hou and Zhenchang Xing and Saining Xie and Liang Zheng},
  year={2025},
  journal={arXiv preprint arXiv:2504.10483},
}
```
