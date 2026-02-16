"""
cDCGAN (conditional Deep Convolutional GAN) for MNIST
Conditioned on (source image, action) from MNISTActionDataset.
Uses a ConditionEncoder (CNN + FCN) to encode the conditioning inputs.
"""

import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# MNISTActionDataset
# ---------------------------------------------------------------------------
class MNISTActionDataset(torch.utils.data.Dataset):
    """Dataset for MNIST with action-based transformations."""

    def __init__(self, A=2, cyclic=False, transform=None):
        self.A = A
        self.cyclic = cyclic
        self.transform = transform
        self.action_dim = 2 * A + 1
        self.onehot = lambda x: torch.eye(self.action_dim)[x + A].float()

        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        self.data_idx_by_class = {i: [] for i in range(10)}
        self.images = []
        self.labels = []
        self.group_idx = []
        for idx, (image, label) in enumerate(self.dataset):
            self.group_idx.append(len(self.data_idx_by_class[label]))
            self.data_idx_by_class[label].append(idx)
            self.images.append(image)
            self.labels.append(label)
        print(f"MNISTActionDataset: A={A}, cyclic={cyclic}, samples={len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.cyclic:
            action = random.randint(-self.A, self.A)
            target_label = (label + action) % 10
        else:
            action = random.randint(
                max(-label, -self.A), min(9 - label, self.A)
            )
            target_label = label + action
        action_obs = self.onehot(action)
        target_image_idx = self.data_idx_by_class[target_label][
            self.group_idx[idx] % len(self.data_idx_by_class[target_label])
        ]
        target_image = self.images[target_image_idx]
        return image, label, target_image, target_label, action, action_obs


# ---------------------------------------------------------------------------
# ConditionEncoder: (image, action_obs) -> cond_embedding
# Uses CNN for image + FCN for action, then FCN to produce embedding.
# ---------------------------------------------------------------------------
class ConditionEncoder(nn.Module):
    def __init__(self, action_dim, cond_dim=128, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        # CNN for image (1, 28, 28)
        self.img_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 28 -> 14
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),  # 14 -> 7
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 7 -> 4
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 1, 0),  # 4 -> 2
            nn.LeakyReLU(0.2),
        )
        # Conv output: 256 * 1 * 1
        self.img_fc_out = 256
        # FCN for action
        self.action_fc = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        # Combined: img features + action features -> cond
        self.combined_fc = nn.Sequential(
            nn.Linear(self.img_fc_out + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, cond_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, image, action_obs):
        # image: [B, 1, 28, 28] or [B, 784]
        # action_obs: [B, action_dim]
        if image.dim() == 2:
            image = image.view(-1, 1, 28, 28)
        img_feat = self.img_conv(image)
        img_feat = img_feat.view(img_feat.size(0), -1)
        action_feat = self.action_fc(action_obs)
        combined = torch.cat([img_feat, action_feat], dim=1)
        return self.combined_fc(combined)


# ---------------------------------------------------------------------------
# Generator: (noise, cond) -> fake image
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, cond_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size = 7
        self.fc = nn.Linear(latent_dim + cond_dim, 256 * self.init_size ** 2)

        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 7 -> 14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 14 -> 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, cond, return_hidden=False):
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, self.init_size, self.init_size)
        h_decoder = self.conv_up(x)  # [B, 64, 28, 28] - penultimate before output conv
        x_fake = self.conv_out(h_decoder)
        if return_hidden:
            h_decoder_flat = h_decoder.view(h_decoder.size(0), -1)
            return x_fake, h_decoder_flat
        return x_fake


# ---------------------------------------------------------------------------
# Discriminator: (image, cond) -> real/fake logit
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, cond_dim=128, cond_spatial_channels=16):
        super().__init__()
        self.cond_dim = cond_dim
        self.cond_spatial_channels = cond_spatial_channels
        self.cond_to_spatial = nn.Linear(cond_dim, cond_spatial_channels)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1 + cond_spatial_channels, 64, 4, 2, 1),  # image + projected cond
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        # Conv output is 512*1*1; concat cond for final fc
        self.fc = nn.Linear(512 + cond_dim, 1)

    def forward(self, img, cond):
        # Project cond to spatial channels and broadcast
        c_spatial = self.cond_to_spatial(cond)  # [B, cond_spatial_channels]
        c_spatial = c_spatial.view(-1, self.cond_spatial_channels, 1, 1).expand(
            -1, -1, img.size(2), img.size(3)
        )
        x = torch.cat([img, c_spatial], dim=1)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, cond], dim=1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_cdcgan(
    G, D, cond_encoder,
    train_loader,
    criterion,
    opt_G, opt_D,
    latent_dim,
    device,
    epochs=30,
    log_interval=100,
    sample_dir=None,
):
    G.train()
    D.train()
    cond_encoder.train()
    loss_G_list, loss_D_list = [], []

    for epoch in range(1, epochs + 1):
        loss_G_epoch, loss_D_epoch = 0.0, 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            (source_imgs, labels, target_imgs, target_labels, actions, action_obs) = [
                x.to(device) for x in batch
            ]
            B = source_imgs.size(0)
            real_valid = torch.ones(B, 1, device=device)
            fake_valid = torch.zeros(B, 1, device=device)

            # ----- Train Discriminator -----
            opt_D.zero_grad()
            cond = cond_encoder(source_imgs, action_obs)
            z = torch.randn(B, latent_dim, device=device)
            fake_imgs = G(z, cond)

            D_real = D(target_imgs, cond)
            D_fake = D(fake_imgs.detach(), cond)

            loss_D_real = criterion(D_real, real_valid)
            loss_D_fake = criterion(D_fake, fake_valid)
            loss_D = (loss_D_real + loss_D_fake) / 2

            loss_D.backward()
            opt_D.step()

            # ----- Train Generator ----- (recompute cond for fresh computation graph)
            opt_G.zero_grad()
            cond = cond_encoder(source_imgs, action_obs)
            z = torch.randn(B, latent_dim, device=device)
            fake_imgs = G(z, cond)
            D_fake = D(fake_imgs, cond)
            loss_G = criterion(D_fake, real_valid)

            loss_G.backward()
            opt_G.step()

            loss_G_epoch += loss_G.item()
            loss_D_epoch += loss_D.item()
            n_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                print(f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss_D: {loss_D_epoch / n_batches:.4f} "
                      f"Loss_G: {loss_G_epoch / n_batches:.4f}")

        loss_G_list.append(loss_G_epoch / n_batches)
        loss_D_list.append(loss_D_epoch / n_batches)
        print(f"Epoch {epoch} | Loss_D: {loss_D_list[-1]:.4f} | Loss_G: {loss_G_list[-1]:.4f}")

        if sample_dir and (epoch % 5 == 0 or epoch == 1):
            save_samples_grid(G, cond_encoder, train_loader, latent_dim, device,
                             f"{sample_dir}/epoch_{epoch:03d}.png", n_samples=20)

    return loss_G_list, loss_D_list


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_loss(loss_G_list, loss_D_list, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_D_list, label="Discriminator", color="C0")
    ax.plot(loss_G_list, label="Generator", color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_samples_grid(G, cond_encoder, train_loader, latent_dim, device, save_path, n_samples=20):
    """Save grid: source | target | generated for random batch conditions."""
    G.eval()
    cond_encoder.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        (source_imgs, labels, target_imgs, target_labels, actions, action_obs) = [
            x.to(device) for x in batch
        ]
        source_imgs = source_imgs[:n_samples]
        target_imgs = target_imgs[:n_samples]
        action_obs = action_obs[:n_samples]
        cond = cond_encoder(source_imgs, action_obs)
        z = torch.randn(n_samples, latent_dim, device=device)
        fake_imgs = G(z, cond)

        rows = []
        for i in range(min(10, n_samples)):
            row = torch.cat([source_imgs[i:i+1], target_imgs[i:i+1], fake_imgs[i:i+1]], dim=0)
            rows.append(row)
        grid = torch.cat(rows, dim=0)
        grid = make_grid(grid, nrow=3, normalize=True, padding=2)
        save_image(grid, save_path)
    G.train()
    cond_encoder.train()


def plot_comparison(G, cond_encoder, train_loader, latent_dim, device, save_path, n_rows=5):
    """Plot source | target | generated with labels."""
    G.eval()
    cond_encoder.eval()
    fig, axs = plt.subplots(n_rows, 3, figsize=(6, 2 * n_rows))
    with torch.no_grad():
        batch = next(iter(train_loader))
        (source_imgs, labels, target_imgs, target_labels, actions, action_obs) = [
            x.to(device) for x in batch
        ]
        n = min(n_rows, source_imgs.size(0))
        cond = cond_encoder(source_imgs[:n], action_obs[:n])
        z = torch.randn(n, latent_dim, device=device)
        fake_imgs = G(z, cond)

        for i in range(n):
            axs[i, 0].imshow(source_imgs[i, 0].cpu().numpy(), cmap="gray")
            axs[i, 0].set_ylabel(f"s:{labels[i].item()}→{target_labels[i].item()} a:{actions[i].item()}")
            axs[i, 0].axis("off")
            axs[i, 1].imshow(target_imgs[i, 0].cpu().numpy(), cmap="gray")
            axs[i, 1].set_title("Target")
            axs[i, 1].axis("off")
            axs[i, 2].imshow(fake_imgs[i, 0].cpu().numpy(), cmap="gray")
            axs[i, 2].set_title("Generated")
            axs[i, 2].axis("off")
    axs[0, 0].set_title("Source")
    plt.suptitle("cDCGAN: source image + action → target")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    G.train()
    cond_encoder.train()


def plot_interpolation(G, cond_encoder, train_loader, latent_dim, device, save_path, n_steps=10):
    """Interpolate in latent space for fixed (source, action) condition."""
    G.eval()
    cond_encoder.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        source_imgs = batch[0][:1].to(device)
        action_obs = batch[5][:1].to(device)
        cond = cond_encoder(source_imgs, action_obs).expand(n_steps, -1)
        z1 = torch.randn(1, latent_dim, device=device)
        z2 = torch.randn(1, latent_dim, device=device)
        alphas = torch.linspace(0, 1, n_steps).to(device)
        imgs = []
        for a in alphas:
            z = ((1 - a) * z1 + a * z2).expand(1, -1)
            img = G(z, cond[:1])
            imgs.append(img)
        grid = torch.cat(imgs, dim=0)
        grid = make_grid(grid, nrow=n_steps, normalize=True, padding=2)
        save_image(grid, save_path)
    G.train()
    cond_encoder.train()


def plot_pca_encoder_decoder(G, cond_encoder, train_loader, latent_dim, device, save_dir):
    """
    Plot PCA of ConditionEncoder output and Generator's decoder hidden states,
    colored by target labels (and action), similar to main_GAN.py.
    """
    G.eval()
    cond_encoder.eval()
    hidden_encoder_l = []
    hidden_decoder_l = []
    target_label_l = []
    action_l = []

    with torch.no_grad():
        for batch in train_loader:
            (source_imgs, labels, target_imgs, target_labels, actions, action_obs) = [
                x.to(device) for x in batch
            ]
            cond = cond_encoder(source_imgs, action_obs)
            z = torch.randn(source_imgs.size(0), latent_dim, device=device)
            _, h_decoder = G(z, cond, return_hidden=True)

            hidden_encoder_l.append(cond.detach().cpu().numpy())
            hidden_decoder_l.append(h_decoder.detach().cpu().numpy())
            target_label_l.append(target_labels.detach().cpu().numpy())
            action_l.append(actions.detach().cpu().numpy())

    os.makedirs(f"{save_dir}/pca", exist_ok=True)
    os.makedirs(f"{save_dir}/pca_centers", exist_ok=True)

    for var_list, var_name in zip(
        [hidden_encoder_l, hidden_decoder_l],
        ["hidden_encoder", "hidden_decoder"],
    ):
        var_np = np.concatenate(var_list, axis=0)
        label_np = np.concatenate(target_label_l, axis=0)
        action_np = np.concatenate(action_l, axis=0)

        pca = PCA(n_components=4)
        var_pca = pca.fit_transform(var_np)
        explained_variance_ratio = pca.explained_variance_ratio_

        # Full PCA: colored by target label and action
        fig, axs_all = plt.subplots(2, 2, figsize=(10, 10))
        for pcs_i in range(2):
            axs = axs_all[pcs_i]
            for ax, color_var, color_label in zip(
                axs, [label_np, action_np], ["target label", "action"]
            ):
                scatter = ax.scatter(
                    var_pca[:, pcs_i * 2],
                    var_pca[:, pcs_i * 2 + 1],
                    c=color_var,
                    cmap="coolwarm",
                    alpha=0.7,
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_label)
                ax.set_xlabel(
                    f"Component {2*pcs_i+1} ({100*explained_variance_ratio[2*pcs_i]:.2f}%)"
                )
                ax.set_ylabel(
                    f"Component {2*pcs_i+2} ({100*explained_variance_ratio[2*pcs_i+1]:.2f}%)"
                )
                ax.set_title(f"PCA of {var_name} colored by {color_label}")
        plt.tight_layout()
        fig.savefig(f"{save_dir}/pca/pca_{var_name}.png")
        plt.close()

        # PCA centers (mean per target label, mean per action)
        fig, axs_all = plt.subplots(2, 2, figsize=(10, 10))
        for pcs_i in range(2):
            axs = axs_all[pcs_i]
            for ax, color_var, color_label in zip(
                axs, [label_np, action_np], ["target label", "action"]
            ):
                var_pca_centers = np.array(
                    [var_pca[color_var == i].mean(axis=0) for i in np.unique(color_var)]
                )
                scatter = ax.scatter(
                    var_pca_centers[:, pcs_i * 2],
                    var_pca_centers[:, pcs_i * 2 + 1],
                    c=np.unique(color_var),
                    cmap="coolwarm",
                    alpha=0.7,
                    s=100,
                    edgecolors="black",
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_label)
                ax.set_xlabel(
                    f"Component {2*pcs_i+1} ({100*explained_variance_ratio[2*pcs_i]:.2f}%)"
                )
                ax.set_ylabel(
                    f"Component {2*pcs_i+2} ({100*explained_variance_ratio[2*pcs_i+1]:.2f}%)"
                )
                ax.set_title(f"PCA of {var_name} centers colored by {color_label}")
                ax.axis("equal")
        plt.tight_layout()
        fig.savefig(f"{save_dir}/pca_centers/pca_{var_name}_centers.png")
        plt.close()

    G.train()
    cond_encoder.train()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def get_config(
    latent_dim=100,
    batch_size=128,
    epochs=30,
    lr=2e-4,
    A=5,
    cond_dim=128,
    cond_hidden=256,
):
    """Return config dict for reproducibility."""
    return {
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "A": A,
        "cond_dim": cond_dim,
        "cond_hidden": cond_hidden,
        "action_dim": 2 * A + 1,
    }


def save_checkpoint(G, D, cond_encoder, config, loss_G_list, loss_D_list, save_path):
    """Save model checkpoints and config to a file."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "G_state_dict": G.state_dict(),
            "D_state_dict": D.state_dict(),
            "cond_encoder_state_dict": cond_encoder.state_dict(),
            "config": config,
            "loss_G": loss_G_list,
            "loss_D": loss_D_list,
        },
        save_path,
    )
    config_path = save_path.replace(".pt", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path):
    """Load checkpoint and return models, config, loss history."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ckpt["config"]
    action_dim = config["action_dim"]
    cond_dim = config["cond_dim"]
    latent_dim = config["latent_dim"]
    cond_hidden = config.get("cond_hidden", 256)

    cond_encoder = ConditionEncoder(
        action_dim=action_dim, cond_dim=cond_dim, hidden_dim=cond_hidden
    ).to(device)
    G = Generator(latent_dim=latent_dim, cond_dim=cond_dim).to(device)
    D = Discriminator(cond_dim=cond_dim).to(device)

    cond_encoder.load_state_dict(ckpt["cond_encoder_state_dict"])
    G.load_state_dict(ckpt["G_state_dict"])
    D.load_state_dict(ckpt["D_state_dict"])

    loss_G_list = ckpt.get("loss_G", [])
    loss_D_list = ckpt.get("loss_D", [])

    return G, D, cond_encoder, config, loss_G_list, loss_D_list


def visualize_only(checkpoint_path, out_dir=None):
    """Load checkpoint and run all visualizations without training."""
    print(f"Loading checkpoint from {checkpoint_path}")
    G, D, cond_encoder, config, loss_G_list, loss_D_list = load_checkpoint(checkpoint_path)

    latent_dim = config["latent_dim"]
    batch_size = config["batch_size"]
    A = config["A"]

    if out_dir is None:
        out_dir = "figures_cDCGAN"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = MNISTActionDataset(A=A, cyclic=False, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Saving visualizations...")
    if loss_G_list and loss_D_list:
        plot_loss(loss_G_list, loss_D_list, f"{out_dir}/loss.png")
    save_samples_grid(G, cond_encoder, train_loader, latent_dim, device,
                     f"{out_dir}/samples_grid.png", n_samples=30)
    plot_comparison(G, cond_encoder, train_loader, latent_dim, device, f"{out_dir}/comparison.png")
    plot_interpolation(G, cond_encoder, train_loader, latent_dim, device, f"{out_dir}/interpolation.png")
    save_samples_grid(G, cond_encoder, train_loader, latent_dim, device,
                     f"{out_dir}/samples/final.png", n_samples=20)
    print("Plotting PCA of encoder and decoder...")
    plot_pca_encoder_decoder(G, cond_encoder, train_loader, latent_dim, device, out_dir)

    print(f"Done. Outputs saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train_and_vis(save_checkpoint_path=None):
    latent_dim = 100
    batch_size = 128
    epochs = 30
    lr = 2e-4
    A = 5  # action range [-A, A]
    cond_dim = 128
    cond_hidden = 256
    config = get_config(
        latent_dim=latent_dim,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        A=A,
        cond_dim=cond_dim,
        cond_hidden=cond_hidden,
    )

    out_dir = "figures_cDCGAN"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = MNISTActionDataset(A=A, cyclic=False, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    action_dim = 2 * A + 1
    cond_encoder = ConditionEncoder(action_dim=action_dim, cond_dim=cond_dim, hidden_dim=cond_hidden).to(device)
    G = Generator(latent_dim=latent_dim, cond_dim=cond_dim).to(device)
    D = Discriminator(cond_dim=cond_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(
        list(G.parameters()) + list(cond_encoder.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    opt_D = optim.Adam(
        list(D.parameters()) + list(cond_encoder.parameters()), lr=lr, betas=(0.5, 0.999)
    )

    print("Training cDCGAN on MNISTActionDataset...")
    loss_G_list, loss_D_list = train_cdcgan(
        G, D, cond_encoder, train_loader, criterion, opt_G, opt_D,
        latent_dim=latent_dim, device=device, epochs=epochs,
        sample_dir=f"{out_dir}/samples",
    )

    print("Saving visualizations...")
    plot_loss(loss_G_list, loss_D_list, f"{out_dir}/loss.png")
    save_samples_grid(G, cond_encoder, train_loader, latent_dim, device,
                     f"{out_dir}/samples_grid.png", n_samples=30)
    plot_comparison(G, cond_encoder, train_loader, latent_dim, device, f"{out_dir}/comparison.png")
    plot_interpolation(G, cond_encoder, train_loader, latent_dim, device, f"{out_dir}/interpolation.png")
    save_samples_grid(G, cond_encoder, train_loader, latent_dim, device,
                     f"{out_dir}/samples/final.png", n_samples=20)
    print("Plotting PCA of encoder and decoder...")
    plot_pca_encoder_decoder(G, cond_encoder, train_loader, latent_dim, device, out_dir)

    if save_checkpoint_path:
        save_checkpoint(G, D, cond_encoder, config, loss_G_list, loss_D_list, save_checkpoint_path)

    print(f"Done. Outputs saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cDCGAN: train or visualize from checkpoint")
    parser.add_argument(
        "--visualize-only",
        type=str,
        default=None,
        metavar="PATH",
        help="Load checkpoint and run visualizations only (no training)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/cdcgan.pt",
        metavar="PATH",
        help="Path to save checkpoint after training (default: checkpoints/cdcgan.pt)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (used with --visualize-only)",
    )
    args = parser.parse_args()

    if args.visualize_only:
        visualize_only(args.visualize_only, out_dir=args.out_dir)
    else:
        train_and_vis(save_checkpoint_path=args.checkpoint)
