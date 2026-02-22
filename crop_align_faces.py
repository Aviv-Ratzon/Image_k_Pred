#!/usr/bin/env python3
"""
Standalone face preprocessing: detect + align + crop faces to 64x64 using MTCNN (facenet-pytorch),
then write aligned images to: data/faces_aligned

- Input:  data/faces (recursively; jpg/png/webp)
- Output: data/faces_aligned (mirrors subfolders)
- Uses multi-GPU by spawning one process per GPU (default: all visible GPUs).
- If MTCNN fails on an image, you can either skip it (default) or fallback to center-crop.

Install:
  pip install facenet-pytorch pillow tqdm torchvision torch

Run:
  python preprocess_align_mtcnn.py \
    --in_dir data/faces \
    --out_dir data/faces_aligned \
    --image_size 64 \
    --margin 16 \
    --gpus 0,1,2,3,4,5,6,7 \
    --batch_size 128
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.multiprocessing as mp
from PIL import Image, ImageFile
from tqdm import tqdm

# Avoid PIL crash on truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images_recursive(root: Path) -> List[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            paths.append(p)
    paths.sort()
    return paths


def chunk_list(items: List[Path], num_chunks: int, chunk_idx: int) -> List[Path]:
    # deterministic sharding
    return items[chunk_idx::num_chunks]


def safe_open_rgb(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception:
        return None


def ensure_parent_dir(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    # simple fallback if desired
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.BILINEAR)
    return img


def worker(
    rank: int,
    world_size: int,
    gpu_id: int,
    in_dir: str,
    out_dir: str,
    image_size: int,
    margin: int,
    batch_size: int,
    skip_failed: bool,
    jpeg_quality: int,
) -> None:
    # Bind this process to a specific GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    from facenet_pytorch import MTCNN
    from torchvision.transforms.functional import to_pil_image

    in_root = Path(in_dir)
    out_root = Path(out_dir)

    all_paths = list_images_recursive(in_root)
    my_paths = chunk_list(all_paths, world_size, rank)

    mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        keep_all=False,
        post_process=False,  # we want raw aligned crop; we save as uint8 later
        device=device,
        select_largest=True,
    )

    processed = 0
    aligned_ok = 0
    failed = 0
    skipped = 0

    # Batch processing: MTCNN accepts a list of PIL Images
    pbar = tqdm(total=len(my_paths), desc=f"GPU {gpu_id} (rank {rank})", dynamic_ncols=True)

    i = 0
    while i < len(my_paths):
        batch_paths = my_paths[i : i + batch_size]
        i += batch_size

        pil_imgs: List[Optional[Image.Image]] = [safe_open_rgb(p) for p in batch_paths]

        # Keep indices for valid images only
        valid: List[Tuple[int, Image.Image]] = [(j, im) for j, im in enumerate(pil_imgs) if im is not None]
        if not valid:
            pbar.update(len(batch_paths))
            continue

        idxs = [j for j, _ in valid]
        imgs = [im for _, im in valid]

        # Forward MTCNN
        try:
            # returns: Tensor [B, 3, H, W] or list with Nones if some fail
            faces = mtcnn(imgs)
        except Exception:
            faces = [None] * len(imgs)

        # Normalize output to a list
        if torch.is_tensor(faces):
            face_list = [faces[k] for k in range(faces.shape[0])]
        else:
            face_list = list(faces)

        # Write outputs
        for local_k, face_tensor in enumerate(face_list):
            orig_batch_j = idxs[local_k]
            in_path = batch_paths[orig_batch_j]
            rel = in_path.relative_to(in_root)
            out_path = out_root / rel
            out_path = out_path.with_suffix(".jpg")  # normalize to jpg

            processed += 1

            if face_tensor is None:
                failed += 1
                if skip_failed:
                    skipped += 1
                    continue
                # fallback: center crop
                img = pil_imgs[orig_batch_j]
                if img is None:
                    skipped += 1
                    continue
                aligned = center_crop_resize(img, image_size)
                ensure_parent_dir(out_path)
                aligned.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
                aligned_ok += 1
                continue

            # face_tensor is float tensor (post_process=False). Convert to PIL safely.
            # Range is typically [0, 255] in float; clamp and cast.
            face_tensor = face_tensor.detach().float().clamp(0, 255) / 255.0
            aligned = to_pil_image(face_tensor.cpu())

            ensure_parent_dir(out_path)
            aligned.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
            aligned_ok += 1

        pbar.update(len(batch_paths))

        # Update progress postfix occasionally
        if processed % (batch_size * 10) == 0:
            pbar.set_postfix(ok=aligned_ok, failed=failed, skipped=skipped)

    pbar.close()

    print(
        f"[rank {rank} gpu {gpu_id}] done: processed={processed} ok={aligned_ok} failed={failed} skipped={skipped}",
        flush=True,
    )


def parse_gpus(gpus_arg: str) -> List[int]:
    gpus_arg = gpus_arg.strip()
    if gpus_arg.lower() in {"all", ""}:
        n = torch.cuda.device_count()
        return list(range(n))
    parts = [p.strip() for p in gpus_arg.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="data/faces", help="Input folder (recursive).")
    ap.add_argument("--out_dir", type=str, default="data/faces_aligned", help="Output folder.")
    ap.add_argument("--image_size", type=int, default=64, help="Output size (64 recommended).")
    ap.add_argument("--margin", type=int, default=16, help="MTCNN margin (pixels).")
    ap.add_argument("--batch_size", type=int, default=128, help="MTCNN batch size per GPU process.")
    ap.add_argument("--gpus", type=str, default="all", help="Comma list, e.g. 0,1,2 or 'all'.")
    ap.add_argument(
        "--skip_failed",
        action="store_true",
        help="If set, images where MTCNN fails are skipped (default behavior).",
    )
    ap.add_argument(
        "--fallback_center_crop",
        action="store_true",
        help="If set, when MTCNN fails we do a simple center-crop+resize instead of skipping.",
    )
    ap.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for saved outputs.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this multi-GPU script. Exiting.", file=sys.stderr)
        sys.exit(1)

    gpus = parse_gpus(args.gpus)
    if len(gpus) == 0:
        print("No GPUs selected/found.", file=sys.stderr)
        sys.exit(1)

    in_root = Path(args.in_dir)
    if not in_root.exists():
        print(f"Input directory does not exist: {args.in_dir}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Default behavior: skip failed.
    # If fallback_center_crop is set, we do NOT skip failed.
    skip_failed = True
    if args.fallback_center_crop:
        skip_failed = False
    elif args.skip_failed:
        skip_failed = True

    # Shard work across GPUs: one process per GPU
    world_size = len(gpus)
    print(
        f"Using {world_size} GPU processes: {gpus}\n"
        f"Input:  {args.in_dir}\n"
        f"Output: {args.out_dir}\n"
        f"image_size={args.image_size}, margin={args.margin}, batch_size={args.batch_size}\n"
        f"on_fail={'skip' if skip_failed else 'center-crop fallback'}",
        flush=True,
    )

    mp.set_start_method("spawn", force=True)
    procs = []
    for rank, gpu_id in enumerate(gpus):
        p = mp.Process(
            target=worker,
            args=(
                rank,
                world_size,
                gpu_id,
                args.in_dir,
                args.out_dir,
                args.image_size,
                args.margin,
                args.batch_size,
                skip_failed,
                args.jpeg_quality,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # quick summary
    print("All processes finished.", flush=True)


if __name__ == "__main__":
    main()