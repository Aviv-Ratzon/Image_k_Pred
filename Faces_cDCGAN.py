import os
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    data_root: str = "./data/faces_aligned"   # folder with images
    out_dir: str = "./runs/cdcgan_age"
    img_size: int = 128                # DCGAN typical: 64
    batch_size: int = 128
    num_workers: int = 8
    epochs: int = 500
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)

    z_dim: int = 512
    g_channels: int = 64
    d_channels: int = 64

    # conditioning
    use_age_groups: bool = True       # True: discrete groups; False: continuous age
    num_age_groups: int = 8           # used if use_age_groups=True
    # If continuous age: we normalize age to [0,1] by age/max_age_for_norm
    max_age_for_norm: float = 100.0

    # optional aux classifier (ACGAN-style) on discriminator
    use_aux_classifier: bool = False
    aux_loss_weight: float = 1.0

    # training tricks
    amp: bool = True
    seed: int = 42
    sample_every_epochs: int = 50
    fixed_n: int = 64                # number of fixed samples

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


import os
from typing import Optional

def ensure_face_data_downloaded(
    data_root: str,
    dataset_name: str = "ljnlonoljpiljm/utkface",
    split: str = "train",
    limit: Optional[int] = None,
    seed: int = 42,
) -> None:
    """
    Downloads a UTKFace-like dataset from Hugging Face and exports images to `data_root`
    with filenames that begin with age, e.g.:
        34_0_2_20170117200132563.jpg

    This makes it compatible with the earlier UTKFaceLikeDataset loader that parses age
    from filename prefix.

    Requires: pip install datasets pillow tqdm
    """
    os.makedirs(data_root, exist_ok=True)

    # If folder already has images, don't re-download/export
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    existing = [fn for fn in os.listdir(data_root) if os.path.splitext(fn.lower())[1] in exts]
    if len(existing) > 0:
        print(f"[data] Found {len(existing)} images in {data_root}. Skipping download.")
        return

    print("[data] Note: UTKFace is typically for non-commercial research purposes only; "
          "check the dataset license/terms before use.")
    # (license note based on UTKFace homepage)

    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError as e:
        raise RuntimeError(
            "Missing deps. Install with:\n"
            "  pip install datasets pillow tqdm\n"
        ) from e

    ds = load_dataset(dataset_name, split=split)

    # Optional shuffle+limit for quick tests
    if limit is not None:
        ds = ds.shuffle(seed=seed).select(range(int(limit)))

    # Export
    print(f"[data] Exporting {len(ds)} images to {data_root} ...")
    for ex in tqdm(ds, total=len(ds)):
        img = ex["image"]  # PIL image
        age = int(ex["age"]) if ex.get("age") is not None else 0

        # these fields may exist depending on dataset; safe defaults
        gender = ex.get("gender", 0)
        ethnicity = ex.get("ethnicity", 0)
        img_id = ex.get("id", None)

        # Normalize to UTKFace-like filename
        # Keep the critical part: filename starts with "{age}_"
        suffix = str(img_id) if img_id is not None else f"hf_{abs(hash(str(ex))) % (10**14)}"
        fn = f"{age}_{gender}_{ethnicity}_{suffix}.jpg"
        out_path = os.path.join(data_root, fn)

        # Save as jpeg for consistency (convert RGBA/P to RGB; JPEG does not support alpha)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out_path, format="JPEG", quality=95)

    print(f"[data] Done. Images available at: {data_root}")

# ----------------------------
# Dataset
# ----------------------------
class UTKFaceLikeDataset(Dataset):
    """
    Minimal dataset that reads images from a folder and extracts age from filename.

    Expected filename pattern like UTKFace:
        age_gender_race_date.jpg  e.g., 34_1_2_20170116174525125.jpg

    If your dataset differs, modify _parse_age().
    """
    def __init__(self, root: str, img_size: int, use_age_groups: bool, num_age_groups: int,
                 max_age_for_norm: float):
        self.root = root
        self.paths = []
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        for fn in os.listdir(root):
            if os.path.splitext(fn.lower())[1] in exts:
                self.paths.append(os.path.join(root, fn))
        self.paths.sort()
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in: {root}")

        self.use_age_groups = use_age_groups
        self.num_age_groups = num_age_groups
        self.max_age_for_norm = max_age_for_norm

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),  # [-1,1]
        ])
        

    def _parse_age(self, path: str) -> int:
        base = os.path.basename(path)
        # UTKFace pattern: age_...
        try:
            age_str = base.split("_")[0]
            age = int(age_str)
            return max(0, min(120, age))
        except Exception:
            raise ValueError(f"Could not parse age from filename: {base}")

    def _age_to_group(self, age: int) -> int:
        """
        Example binning into num_age_groups. You can customize.
        Here we split [0, max_age_for_norm] into equal-width bins.
        """
        max_age = self.max_age_for_norm
        a = min(max(age, 0), int(max_age))
        bin_w = max_age / self.num_age_groups
        g = int(a // bin_w)
        return min(self.num_age_groups - 1, max(0, g))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)

        age = self._parse_age(path)
        if self.use_age_groups:
            y = self._age_to_group(age)
            return x, torch.tensor(y, dtype=torch.long)
        else:
            # continuous scalar in [0,1]
            y = torch.tensor([min(age, self.max_age_for_norm) / self.max_age_for_norm],
                             dtype=torch.float32)
            return x, y


# ----------------------------
# Models
# ----------------------------
def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class CondEmbedding(nn.Module):
    """
    Provides a conditioning vector:
    - If age groups: learned nn.Embedding
    - If continuous age: MLP from scalar -> embed_dim
    """
    def __init__(self, use_age_groups: bool, num_age_groups: int, embed_dim: int):
        super().__init__()
        self.use_age_groups = use_age_groups
        self.embed_dim = embed_dim
        if use_age_groups:
            self.emb = nn.Embedding(num_age_groups, embed_dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

    def forward(self, y):
        if self.use_age_groups:
            return self.emb(y)  # [B, embed_dim]
        else:
            return self.mlp(y)  # [B, embed_dim]


class Generator(nn.Module):
    """
    DCGAN generator with conditional injection by concatenating z and cond embedding.
    Output: 3x64x64 in [-1,1] via Tanh.
    """
    def __init__(self, z_dim: int, cond_dim: int, base_ch: int = 64, out_size: int = 64):
        super().__init__()
        # assert out_size == 64, "This reference implementation is fixed to 64x64 for simplicity."
        in_dim = z_dim + cond_dim
        self.net = nn.Sequential(
            # in: (B, in_dim, 1, 1)
            nn.ConvTranspose2d(in_dim, base_ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, c):
        # z: [B,z_dim], c: [B,cond_dim]
        x = torch.cat([z, c], dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


class ProjectionDiscriminator(nn.Module):
    """
    DCGAN discriminator backbone + projection conditioning:
        score(x, y) = h(x) + <phi(y), f(x)>
    where f(x) is final feature vector and phi(y) is embedding.

    Optional: ACGAN-style aux classifier head.
    """
    def __init__(self, cond_dim: int, base_ch: int = 64, use_aux_classifier: bool = False,
                 num_age_groups: int = 8, use_age_groups: bool = True):
        super().__init__()
        self.use_aux_classifier = use_aux_classifier
        self.use_age_groups = use_age_groups
        self.num_age_groups = num_age_groups

        # 64x64 -> 4x4
        self.feat = nn.Sequential(
            nn.Conv2d(3, base_ch, 4, 2, 1, bias=False),     # 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),  # 16
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False),  # 8
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False),  # 4
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = nn.Conv2d(base_ch * 8, base_ch * 8, 4, 1, 0, bias=False)  # 1x1
        self.out = nn.Linear(base_ch * 8, 1)

        # projection embedding maps condition -> same dimension as features
        self.proj = nn.Linear(cond_dim, base_ch * 8, bias=False)

        # optional aux head (better if condition is discrete)
        if self.use_aux_classifier:
            if not self.use_age_groups:
                raise ValueError("Aux classifier is intended for discrete age groups.")
            self.aux = nn.Linear(base_ch * 8, num_age_groups)

    def forward(self, x, c):
        h = self.feat(x)
        h = self.final_conv(h).squeeze(-1).squeeze(-1)  # [B, feat_dim]
        base_logit = self.out(h).squeeze(1)             # [B]

        # projection term
        pc = self.proj(c)                                # [B, feat_dim]
        proj_logit = torch.sum(h * pc, dim=1)            # [B]

        logit = base_logit + proj_logit

        if self.use_aux_classifier:
            aux_logits = self.aux(h)
            return logit, aux_logits
        return logit


# ----------------------------
# Losses (DCGAN-style BCE with logits)
# ----------------------------
def d_loss_fn(real_logits, fake_logits):
    # real -> 1, fake -> 0
    loss_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return loss_real + loss_fake

def g_loss_fn(fake_logits):
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


# ----------------------------
# Training
# ----------------------------
@torch.no_grad()
def sample_grid(G, cond_emb, cfg: TrainConfig, fixed_z, fixed_y, epoch: int):
    G.eval()
    c = cond_emb(fixed_y)
    fake = G(fixed_z, c)
    grid = make_grid(fake, nrow=int(math.sqrt(fake.size(0))), normalize=True, value_range=(-1, 1))
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_image(grid, os.path.join(cfg.out_dir, f"samples_epoch_{epoch:04d}.png"))
    G.train()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = TrainConfig()

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device(cfg.device)

    ensure_face_data_downloaded(
        data_root=cfg.data_root,
        dataset_name="ljnlonoljpiljm/utkface",
        split="train",
        limit=None,   # set e.g. 2000 for a fast smoke test
    )

    ds = UTKFaceLikeDataset(
        root=cfg.data_root,
        img_size=cfg.img_size,
        use_age_groups=cfg.use_age_groups,
        num_age_groups=cfg.num_age_groups,
        max_age_for_norm=cfg.max_age_for_norm,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    cond_dim = 64
    cond_emb = CondEmbedding(cfg.use_age_groups, cfg.num_age_groups, cond_dim).to(device)

    G = Generator(cfg.z_dim, cond_dim, base_ch=cfg.g_channels, out_size=cfg.img_size).to(device)
    D = ProjectionDiscriminator(
        cond_dim=cond_dim,
        base_ch=cfg.d_channels,
        use_aux_classifier=cfg.use_aux_classifier,
        num_age_groups=cfg.num_age_groups,
        use_age_groups=cfg.use_age_groups,
    ).to(device)

    G.apply(weights_init_dcgan)
    D.apply(weights_init_dcgan)

    optG = torch.optim.Adam(list(G.parameters()) + list(cond_emb.parameters()), lr=cfg.lr, betas=cfg.betas)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # fixed samples
    fixed_z = torch.randn(cfg.fixed_n, cfg.z_dim, device=device)
    if cfg.use_age_groups:
        # balanced grid over age groups
        ys = torch.arange(cfg.num_age_groups, device=device).repeat(math.ceil(cfg.fixed_n / cfg.num_age_groups))
        fixed_y = ys[:cfg.fixed_n]
    else:
        # continuous ages evenly spaced
        fixed_y = torch.linspace(0, 1, steps=cfg.fixed_n, device=device).unsqueeze(1)

    for epoch in range(1, cfg.epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}")
        for real, y in pbar:
            real = real.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---------------------
            # Train D
            # ---------------------
            z = torch.randn(real.size(0), cfg.z_dim, device=device)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                c = cond_emb(y)
                fake = G(z, c).detach()

                if cfg.use_aux_classifier:
                    real_logits, real_aux = D(real, c)
                    fake_logits, _ = D(fake, c)
                    lossD = d_loss_fn(real_logits, fake_logits)
                    # aux classification on REAL (common choice)
                    loss_aux = F.cross_entropy(real_aux, y)
                    lossD = lossD + cfg.aux_loss_weight * loss_aux
                else:
                    real_logits = D(real, c)
                    fake_logits = D(fake, c)
                    lossD = d_loss_fn(real_logits, fake_logits)

            optD.zero_grad(set_to_none=True)
            scaler.scale(lossD).backward()
            scaler.step(optD)

            # ---------------------
            # Train G (+ cond embedding)
            # ---------------------
            z = torch.randn(real.size(0), cfg.z_dim, device=device)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                c = cond_emb(y)
                fake = G(z, c)

                if cfg.use_aux_classifier:
                    fake_logits, fake_aux = D(fake, c)
                    lossG = g_loss_fn(fake_logits)
                    # encourage generated samples to match condition
                    lossG = lossG + cfg.aux_loss_weight * F.cross_entropy(fake_aux, y)
                else:
                    fake_logits = D(fake, c)
                    lossG = g_loss_fn(fake_logits)

            optG.zero_grad(set_to_none=True)
            scaler.scale(lossG).backward()
            scaler.step(optG)
            scaler.update()

            pbar.set_postfix(lossD=float(lossD.detach().cpu()), lossG=float(lossG.detach().cpu()))

        if epoch % cfg.sample_every_epochs == 0:
            sample_grid(G, cond_emb, cfg, fixed_z, fixed_y, epoch)

        # save checkpoints
        torch.save({
            "G": G.state_dict(),
            "D": D.state_dict(),
            "cond_emb": cond_emb.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "cfg": cfg.__dict__,
            "epoch": epoch
        }, os.path.join(cfg.out_dir, "latest.pt"))

    print("Done.")


if __name__ == "__main__":
    main()