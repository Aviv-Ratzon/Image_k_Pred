"""
cDCGAN (conditional Deep Convolutional GAN) for MNIST
Conditioned on (source image, action) from MNISTActionDataset.
Uses a ConditionEncoder (CNN + FCN) to encode the conditioning inputs.
"""

import itertools
import multiprocessing as mp
import random
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


N_DIGITS = 10

# ---------------------------------------------------------------------------
# Pretrained MNIST Classifier
# ---------------------------------------------------------------------------
class MNISTClassifier(nn.Module):
    """Simple CNN classifier for 28x28 grayscale MNIST. Expects input normalized to [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_pretrained_mnist_classifier(device, checkpoint_path="checkpoints/mnist_classifier_cdcgan.pt"):
    """Load pretrained MNIST classifier, training it first if checkpoint does not exist."""
    d = os.path.dirname(checkpoint_path)
    if d:
        os.makedirs(d, exist_ok=True)
    classifier = MNISTClassifier().to(device)
    if os.path.isfile(checkpoint_path):
        classifier.load_state_dict(torch.load(checkpoint_path, map_location=device))
        classifier.eval()
        print(f"Loaded pretrained classifier from {checkpoint_path}")
        return classifier
    # Train classifier
    print(f"Training MNIST classifier (saving to {checkpoint_path})...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    opt = optim.Adam(classifier.parameters(), lr=1e-3)
    classifier.train()
    for epoch in range(10):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            opt.zero_grad()
            logits = classifier(batch_x)
            loss = nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            opt.step()
    classifier.eval()
    torch.save(classifier.state_dict(), checkpoint_path)
    print(f"Saved classifier to {checkpoint_path}")
    return classifier


# ---------------------------------------------------------------------------
# MNISTActionDataset
# ---------------------------------------------------------------------------
class MNISTActionDataset(torch.utils.data.Dataset):
    """Dataset for MNIST with action-based transformations."""

    def __init__(self, A=2, cyclic=False, transform=None, exclude_pairs={}):
        self.A = A
        self.cyclic = cyclic
        self.transform = transform
        self.action_dim = 2 * (N_DIGITS-1) + 1
        self.only_zero_action = False
        self.onehot = lambda x: torch.eye(self.action_dim)[x + A].float()
        self.exclude_pairs = {f'{i}':j for i, j in exclude_pairs.items()}

        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        self.data_idx_by_class = {i: [] for i in range(10)}
        self.images = []
        self.labels = []
        for idx, (image, label) in enumerate(self.dataset):
            self.data_idx_by_class[label].append(idx)
            self.images.append(image)
            self.labels.append(label)
        self._class_size = [len(self.data_idx_by_class[c]) for c in range(10)]
        print(f"MNISTActionDataset: A={A}, cyclic={cyclic}, samples={len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        action_list = np.arange(-self.A, self.A + 1)
        if str(label) in self.exclude_pairs:
            action_list = action_list[action_list != self.exclude_pairs[str(label)]]
        if self.only_zero_action:
            action = 0
            target_label = label
        elif self.cyclic:
            action = random.choice(action_list)
            target_label = (label + action) % 10
        else:
            action_list = action_list[(action_list >= max(-label, -self.A)) & (action_list <= min(9 - label, self.A))]
            action = random.choice(action_list)
            target_label = label + action
        action_obs = self.onehot(action)
        n = self._class_size[target_label]
        target_image_idx = self.data_idx_by_class[target_label][random.randint(0, n - 1)]
        target_image = self.images[target_image_idx]
        return image, label, target_image, target_label, action, action_obs

    def generate_excluded_pair(self):
        label = random.choice(list(self.exclude_pairs.keys()))
        action = self.exclude_pairs[label]
        label = int(label)
        action_obs = self.onehot(action)

        image = self.images[np.random.choice(self.data_idx_by_class[label])]
        target_label = label + action
        target_image_idx = np.random.choice(self.data_idx_by_class[target_label])
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
        self.fc = nn.Sequential(nn.Linear(latent_dim + cond_dim, 1024), 
                                nn.LeakyReLU(0.2),
                                nn.Linear(1024, 1024),
                                nn.LeakyReLU(0.2),
                                nn.Linear(1024, 256 * self.init_size ** 2))
                                
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 7 -> 14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 14 -> 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, self.init_size, self.init_size)
        return self.conv_blocks(x)


# ---------------------------------------------------------------------------
# Discriminator: (image, cond) -> real/fake logit
# use_spectral_norm: applies spectral normalization to constrain Lipschitz constant (reduces mode collapse)
# ---------------------------------------------------------------------------
def _make_conv_layer(in_ch, out_ch, k, s, p, spectral_norm=False):
    layer = nn.Conv2d(in_ch, out_ch, k, s, p)
    if spectral_norm:
        layer = nn.utils.spectral_norm(layer)
    return layer


def _make_linear_layer(in_feat, out_feat, spectral_norm=False):
    layer = nn.Linear(in_feat, out_feat)
    if spectral_norm:
        layer = nn.utils.spectral_norm(layer)
    return layer


class Discriminator(nn.Module):
    def __init__(self, cond_dim=128, cond_spatial_channels=16, use_spectral_norm=True):
        super().__init__()
        self.cond_dim = cond_dim
        self.cond_spatial_channels = cond_spatial_channels
        sn = use_spectral_norm
        self.cond_to_spatial = _make_linear_layer(cond_dim, cond_spatial_channels, spectral_norm=sn)
        self.conv_blocks = nn.Sequential(
            _make_conv_layer(1 + cond_spatial_channels, 64, 4, 2, 1, spectral_norm=sn),
            nn.LeakyReLU(0.2),
            _make_conv_layer(64, 128, 4, 2, 1, spectral_norm=sn),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            _make_conv_layer(128, 256, 4, 2, 1, spectral_norm=sn),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            _make_conv_layer(256, 512, 3, 1, 0, spectral_norm=sn),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.fc = _make_linear_layer(512 + cond_dim, 1, spectral_norm=sn)

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
# Anti-mode-collapse options:
#   label_smooth_real, label_smooth_fake: soften targets (e.g., 0.9, 0.0) to prevent overconfident D
#   instance_noise_std: add Gaussian noise to D inputs (helps early exploration, hinders mode collapse)
#   instance_noise_decay: per-epoch decay factor (0.99 = slow, 0.9 = faster decay toward zero)
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
    label_smooth_real=0.9,
    label_smooth_fake=0.0,
    instance_noise_std=0.1,
    instance_noise_decay=0.99,
):
    G.train()
    D.train()
    cond_encoder.train()
    loss_G_list, loss_D_list = [], []

    for epoch in range(1, epochs + 1):
        loss_G_epoch, loss_D_epoch = 0.0, 0.0
        n_batches = 0
        # Decay instance noise over epochs (higher early, lower late)
        noise_std = instance_noise_std * (instance_noise_decay ** (epoch - 1)) if instance_noise_std > 0 else 0.0

        for batch_idx, batch in enumerate(train_loader):
            (source_imgs, labels, target_imgs, target_labels, actions, action_obs) = [
                x.to(device) for x in batch
            ]
            B = source_imgs.size(0)
            real_valid = torch.full((B, 1), label_smooth_real, device=device)
            fake_valid = torch.full((B, 1), label_smooth_fake, device=device)

            # ----- Train Discriminator -----
            opt_D.zero_grad()
            cond = cond_encoder(source_imgs, action_obs)
            z = torch.randn(B, latent_dim, device=device)
            fake_imgs = G(z, cond)

            # Instance noise: add to inputs before D (helps prevent mode collapse)
            real_in = target_imgs + noise_std * torch.randn_like(target_imgs, device=device) if noise_std > 0 else target_imgs
            fake_in = fake_imgs.detach() + noise_std * torch.randn_like(fake_imgs, device=device) if noise_std > 0 else fake_imgs.detach()

            D_real = D(real_in, cond)
            D_fake = D(fake_in, cond)

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
            fake_in_g = fake_imgs + noise_std * torch.randn_like(fake_imgs, device=device) if noise_std > 0 else fake_imgs
            D_fake = D(fake_in_g, cond)
            loss_G = criterion(D_fake, real_valid)

            loss_G.backward()
            opt_G.step()

            loss_G_epoch += loss_G.item()
            loss_D_epoch += loss_D.item()
            n_batches += 1

            # if (batch_idx + 1) % log_interval == 0:
            #     print(f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
            #           f"Loss_D: {loss_D_epoch / n_batches:.4f} "
            #           f"Loss_G: {loss_G_epoch / n_batches:.4f}")

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
    fig, axs = plt.subplots(n_rows, 7, figsize=(2*7, 2 * n_rows))
    with torch.no_grad():
        batch = next(iter(train_loader))
        (source_imgs, labels, target_imgs, target_labels, actions, action_obs) = [
            x.to(device) for x in batch
        ]
        n = min(n_rows, source_imgs.size(0)//2)
        cond = cond_encoder(source_imgs, action_obs)
        z = torch.randn(source_imgs.shape[0], latent_dim, device=device)
        fake_imgs = G(z, cond)
        for j in range(2):
            for i in range(n):
                ind = i + j*n
                axs[i, j*4+0].imshow(source_imgs[ind, 0].cpu().numpy(), cmap="gray")
                axs[i, j*4+0].set_ylabel(f"s:{labels[ind].item()}→{target_labels[ind].item()} a:{actions[ind].item()}")
                axs[i, j*4+0].axis("off")
                axs[i, j*4+1].imshow(target_imgs[ind, 0].cpu().numpy(), cmap="gray")
                axs[i, j*4+1].set_title("Target")
                axs[i, j*4+1].axis("off")
                axs[i, j*4+2].imshow(fake_imgs[ind, 0].cpu().numpy(), cmap="gray")
                axs[i, j*4+2].set_title("Generated")
                axs[i, j*4+2].axis("off")
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


def plot_cond_pca(cond_encoder, train_loader, device, save_path):
    cond_encoder.eval()
    encoded_conds = []
    labels_l = []
    actions_l = []
    with torch.no_grad():
        for batch in train_loader:
            source_imgs = batch[0].to(device)
            actions = batch[4].to(device)
            action_obs = batch[5].to(device)
            target_labels = batch[3].to(device)
            cond = cond_encoder(source_imgs, action_obs)
            cond = cond.view(cond.size(0), -1)
            encoded_conds.append(cond.detach().cpu().numpy())
            labels_l.append(target_labels.detach().cpu().numpy())
            actions_l.append(actions.detach().cpu().numpy())
    pca = PCA(n_components=2)
    pca.fit(np.concatenate(encoded_conds, axis=0))
    cond_pca = pca.transform(np.concatenate(encoded_conds, axis=0))
    fig, axs = plt.subplots(1,2, figsize=(10, 5))
    im = axs[0].scatter(cond_pca[:, 0], cond_pca[:, 1], c=np.concatenate(labels_l, axis=0), cmap='coolwarm', alpha=0.7)
    cbar = plt.colorbar(im, ax=axs[0])
    cbar.set_label('Target Label')
    axs[0].set_xlabel(f'PCA Component 1 ({100*pca.explained_variance_ratio_[0]:.2f}%)')
    axs[0].set_ylabel(f'PCA Component 2 ({100*pca.explained_variance_ratio_[1]:.2f}%)')
    axs[0].set_title('PCA of Encoded Conditions Colored by Target Label')
    plt.tight_layout()
    im = axs[1].scatter(cond_pca[:, 0], cond_pca[:, 1], c=np.concatenate(actions_l, axis=0), cmap='coolwarm', alpha=0.7)
    cbar = plt.colorbar(im, ax=axs[1])
    cbar.set_label('Action')
    axs[1].set_xlabel(f'PCA Component 1 ({100*pca.explained_variance_ratio_[0]:.2f}%)')
    axs[1].set_ylabel(f'PCA Component 2 ({100*pca.explained_variance_ratio_[1]:.2f}%)')
    axs[1].set_title('PCA of Encoded Conditions Colored by Action')
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close()
    cond_encoder.train()


def save_samples_grid_pca(G, cond_encoder, train_loader, latent_dim, device, save_path, n_samples=40):
    cond_encoder.eval()
    encoded_conds = []
    labels_l = []
    actions_l = []
    with torch.no_grad():
        for batch in train_loader:
            source_imgs = batch[0].to(device)
            actions = batch[4].to(device)
            action_obs = batch[5].to(device)
            target_labels = batch[3].to(device)
            cond = cond_encoder(source_imgs, action_obs)
            cond = cond.view(cond.size(0), -1)
            encoded_conds.append(cond.detach().cpu().numpy())
            labels_l.append(target_labels.detach().cpu().numpy())
            actions_l.append(actions.detach().cpu().numpy())
    pca = PCA(n_components=2)
    pca.fit(np.concatenate(encoded_conds, axis=0))
    cond_pca = pca.transform(np.concatenate(encoded_conds, axis=0))

    # Generate n_samples points along the first PC axis of cond_pca,
    # ranging from the minimum to the maximum projection of the original conditions
    # Use the mean of the other PCs for each point

    # Find min and max along PC1
    pc1_vals = cond_pca[:, 0]
    pc2_mean = np.mean(cond_pca[:, 1])

    pc1_min = np.min(pc1_vals)
    pc1_max = np.max(pc1_vals)
    pc1_points = np.linspace(pc1_min, pc1_max, n_samples)
    pca_points = np.stack([pc1_points, np.full(n_samples, pc2_mean)], axis=1)

    # Map these points back to the encoded condition space
    pca_inv = pca.inverse_transform(pca_points)  # shape (n_samples, cond_dim)

    """Save grid: source | target | generated for random batch conditions."""
    G.eval()
    cond = torch.tensor(pca_inv, dtype=torch.float32, device=device)
    z = torch.randn(n_samples, latent_dim, device=device)
    fake_imgs = G(z, cond)

    grid = make_grid(fake_imgs, nrow=10, normalize=True, padding=2)
    save_image(grid, save_path)
    G.train()
    cond_encoder.train()

def test_model_excluded_pairs(G, cond_encoder, dataset, classifier, latent_dim, device, save_path, n_samples=10):
    cond_encoder.eval()
    G.eval()
    draw_samples = 300

    labels_l = []
    actions_l = []
    target_imgs = []
    source_imgs = []
    action_obs = []
    target_labels_l = []
    for i in range(draw_samples):
        image, label, target_image, target_label, action, action_obs_i = dataset.generate_excluded_pair()
        image = image.to(device)
        target_image = target_image.to(device)
        action_obs_i = action_obs_i.to(device)
        action_obs.append(action_obs_i)
        source_imgs.append(image)
        target_imgs.append(target_image)
        target_labels_l.append(target_label)
    target_labels = torch.tensor(target_labels_l, dtype=torch.long, device=device)
    source_imgs = torch.stack(source_imgs, dim=0)
    target_imgs = torch.stack(target_imgs, dim=0)
    action_obs = torch.stack(action_obs, dim=0)
    with torch.no_grad():
        cond = cond_encoder(source_imgs, action_obs)
        cond = cond.view(cond.size(0), -1)
    z = torch.randn(draw_samples, latent_dim, device=device)
    fake_imgs = G(z, cond)
    
    classifier.eval()
    pred = classifier(fake_imgs).argmax(dim=1)
    correct = (pred == target_labels).sum().item()
    total = target_labels.size(0)
    accuracy = correct / total

    rows = []
    for i in range(min(10, n_samples)):
        row = torch.cat([source_imgs[i:i+1], target_imgs[i:i+1], fake_imgs[i:i+1]], dim=0)
        rows.append(row)
    grid = torch.cat(rows, dim=0)
    grid = make_grid(grid, nrow=3, normalize=True, padding=2)
    save_image(grid, save_path)
    G.train()
    cond_encoder.train()
    
    return accuracy
    
    
    
def save_trial_statistics(G, cond_encoder, train_loader, latent_dim, device, save_path, classifier=None, excluded_pairs_accuracy=None):
    """Compute and save Participation Ratio, R^2(target label vs first n PCs), and classifier accuracy on generated images."""
    cond_encoder.eval()
    G.eval()
    encoded_conds = []
    target_labels_l = []
    with torch.no_grad():
        for batch in train_loader:
            source_imgs = batch[0].to(device)
            action_obs = batch[5].to(device)
            target_labels = batch[3]
            cond = cond_encoder(source_imgs, action_obs)
            cond = cond.view(cond.size(0), -1)
            encoded_conds.append(cond.detach().cpu().numpy())
            target_labels_l.append(target_labels.numpy())
    cond_encoder.train()
    G.train()

    X = np.concatenate(encoded_conds, axis=0)
    y = np.concatenate(target_labels_l, axis=0).astype(np.float64).reshape(-1, 1)

    # Participation Ratio: PR = (sum λ_i)^2 / sum(λ_i^2) from covariance eigenvalues
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 0)  # numerical safety
    participation_ratio = (np.sum(eigenvalues) ** 2) / (np.sum(eigenvalues ** 2) + 1e-10)

    # R^2 between first n PCs and target label, for n = 1..30
    n_pcs_max = min(30, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_pcs_max)
    X_pca = pca.fit_transform(X_centered)
    n_l = [n for n in range(1, n_pcs_max + 1)]
    r2_per_n = []
    for n in n_l:
        X_n = X_pca[:, :n]
        reg = LinearRegression().fit(X_n, y.ravel())
        r2 = reg.score(X_n, y.ravel())
        r2_per_n.append((n, r2))

    # Classifier accuracy on generated images
    classifier_accuracy = None
    if classifier is not None:
        classifier.eval()
        cond_encoder.eval()
        G.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in train_loader:
                source_imgs = batch[0].to(device)
                action_obs = batch[5].to(device)
                target_labels = batch[3].to(device)
                cond = cond_encoder(source_imgs, action_obs)
                z = torch.randn(source_imgs.size(0), latent_dim, device=device)
                fake_imgs = G(z, cond)
                pred = classifier(fake_imgs).argmax(dim=1)
                correct += (pred == target_labels).sum().item()
                total += target_labels.size(0)
        classifier_accuracy = correct / total if total > 0 else 0.0
        classifier.train()
        cond_encoder.train()
        G.train()

    import pickle
    stats = {
        "participation_ratio": participation_ratio,
        "r2_per_n": r2_per_n,
        "n_l": n_l,
        "classifier_accuracy": classifier_accuracy,
    }
    if excluded_pairs_accuracy is not None:
        stats["excluded_pairs_accuracy"] = excluded_pairs_accuracy
    with open(save_path, "wb") as f:
        pickle.dump(stats, f)
    acc_str = f" (classifier acc: {classifier_accuracy:.4f})" if classifier_accuracy is not None else ""
    excluded_pairs_acc_str = f" (excluded pairs acc: {excluded_pairs_accuracy:.4f})" if excluded_pairs_accuracy is not None else ""
    print(f"Saved trial statistics to {save_path}{acc_str}")


def plot_trial_statistics(base_folder):
    """Load trial_statistics.pkl from each A_* / seed_* folder and plot R^2 vs n (line) and PR (boxplot)."""
    import pickle
    import glob

    # Collect data: A_value -> list of {participation_ratio, r2_per_n}
    data_by_A = {}
    for A_dir in sorted(glob.glob(os.path.join(base_folder, "A_*"))):
        if not os.path.isdir(A_dir):
            continue
        A_name = os.path.basename(A_dir)  # e.g. A_5
        A_val = A_name.split("_")[1]  # e.g. "5"
        data_by_A[A_val] = []
        for seed_dir in sorted(glob.glob(os.path.join(A_dir, "seed_*"))):
            if not os.path.isdir(seed_dir):
                continue
            pkl_path = os.path.join(seed_dir, "trial_statistics.pkl")
            if not os.path.isfile(pkl_path):
                continue
            with open(pkl_path, "rb") as f:
                stats = pickle.load(f)
            print(stats["classifier_accuracy"])
            if stats["classifier_accuracy"] < 0.3:
                continue
            stats['n_l'] = stats['n_l'][:15]
            stats['r2_per_n'] = stats['r2_per_n'][:15]
            if "excluded_pairs_accuracy" in stats:
                stats['excluded_pairs_accuracy'] = stats['excluded_pairs_accuracy']
            data_by_A[A_val].append(stats)

    if not data_by_A:
        print(f"No trial_statistics.pkl found in {base_folder}")
        return

    A_vals_sorted = sorted(data_by_A.keys(), key=lambda x: int(x))
    viridis = plt.colormaps["viridis"]
    colors = [viridis(i / max(len(A_vals_sorted) - 1, 1)) for i in range(len(A_vals_sorted))]

    fig, (ax_line, ax_box, ax_box2) = plt.subplots(1, 3, figsize=(18, 5))

    # Line plot: R^2 vs n, one line per A (mean across seeds), Viridis gradient
    for idx, A_val in enumerate(A_vals_sorted):
        entries = data_by_A[A_val]
        min_len = min(len(e["r2_per_n"]) for e in entries)
        n_vals = np.array([entries[0]["r2_per_n"][i][0] for i in range(min_len)])
        r2_matrix = np.array([[r2 for _, r2 in e["r2_per_n"][:min_len]] for e in entries])
        r2_mean = r2_matrix.mean(axis=0)
        r2_std = r2_matrix.std(axis=0)
        c = colors[idx]
        ax_line.plot(n_vals, r2_mean, label=f"A={A_val}", marker="o", markersize=3, color=c)
        ax_line.fill_between(n_vals, r2_mean - r2_std, r2_mean + r2_std, alpha=0.2, color=c)
    ax_line.set_xlabel("n (number of PCs)")
    ax_line.set_ylabel("R² (target label)")
    ax_line.set_title("R² vs first n PCs")
    ax_line.legend()
    ax_line.grid(True, alpha=0.3)

    # Boxplot: Participation Ratio per A, Viridis gradient
    pr_by_A = [data_by_A[A_val] for A_val in A_vals_sorted]
    pr_values = [[e["participation_ratio"] for e in entries] for entries in pr_by_A]
    A_labels = [f"A={A_val}" for A_val in A_vals_sorted]
    bp = ax_box.boxplot(pr_values, labels=A_labels, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        # Add number of samples as text above the box
        n_samples = len(pr_values[i])
        # Find the y position (top whisker)
        whisker_y = bp['whiskers'][2 * i + 1].get_ydata()[1]
        ax_box.text(i + 1, whisker_y + 0.05, f"n={n_samples}", ha='center', va='bottom', fontsize=9)
    ax_box.set_ylabel("Participation Ratio")
    ax_box.set_title("Participation Ratio by A")

    # Boxplot: Excluded Pairs Accuracy per A, Viridis gradient
    excluded_pairs_acc_by_A = [data_by_A[A_val] for A_val in A_vals_sorted]
    excluded_pairs_acc_values = []
    for entries in excluded_pairs_acc_by_A:
        vals_l = []
        for e in entries:
            if "excluded_pairs_accuracy" in e:
                vals_l.append(e["excluded_pairs_accuracy"])
        excluded_pairs_acc_values.append(vals_l)
    A_labels = [f"A={A_val}" for A_val in A_vals_sorted]
    bp = ax_box2.boxplot(excluded_pairs_acc_values, labels=A_labels, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
    ax_box2.set_ylabel("Excluded Pairs Accuracy")
    ax_box2.set_title("Excluded Pairs Accuracy by A")

    plt.tight_layout()
    out_path = os.path.join(base_folder, "trial_statistics_summary.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")



    fig, (ax_box1, ax_box2) = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot: Maximal R² per run by A, Viridis gradient
    max_r2_by_A = [data_by_A[A_val] for A_val in A_vals_sorted]
    max_r2_values = [
        [max([r2 for _, r2 in e["r2_per_n"]]) for e in entries] 
        for entries in max_r2_by_A
    ]
    A_labels = [f"A={A_val}" for A_val in A_vals_sorted]
    bp = ax_box1.boxplot(max_r2_values, labels=A_labels, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
    ax_box1.set_ylabel("Maximal R² (target label)")
    ax_box1.set_title("Maximal R² by A")

    # Boxplot: Participation Ratio per A, Viridis gradient
    acc_by_A = [data_by_A[A_val] for A_val in A_vals_sorted]
    acc_values = [[e["classifier_accuracy"] for e in entries] for entries in acc_by_A]
    A_labels = [f"A={A_val}" for A_val in A_vals_sorted]
    bp = ax_box2.boxplot(acc_values, labels=A_labels, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
    ax_box2.set_ylabel("Accuracy")
    ax_box2.set_title("Accuracy by A")

    plt.tight_layout()
    out_path = os.path.join(base_folder, "trial_statistics_summary_2.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_single_task(args):
    """Worker: train and visualize for one (A, seed) on assigned GPU. Runs in subprocess."""
    task_idx, A, seed = args
    num_gpus = 8
    gpu_id = task_idx % num_gpus
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    exclude_pairs = {
        '4': 1,
    }

    latent_dim = 100
    batch_size = 128
    epochs = 30
    lr = 2e-4
    cond_dim = 128
    cond_hidden = 256

    torch.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))

    out_dir = f"figures_cDCGAN/A_{A}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = MNISTActionDataset(A=A, cyclic=False, transform=transform, exclude_pairs=exclude_pairs)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    action_dim = 2 * (N_DIGITS-1) + 1
    cond_encoder = ConditionEncoder(action_dim=action_dim, cond_dim=cond_dim, hidden_dim=cond_hidden).to(device)
    G = Generator(latent_dim=latent_dim, cond_dim=cond_dim).to(device)
    D = Discriminator(cond_dim=cond_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.AdamW(
        list(G.parameters()) + list(cond_encoder.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    opt_D = optim.AdamW(
        list(D.parameters()) + list(cond_encoder.parameters()), lr=lr, betas=(0.5, 0.999)
    )

    # Anti-mode-collapse: spectral norm (in D), label smoothing, instance noise
    print(f"[GPU {gpu_id}] A={A} seed={seed}: Training...")
    loss_G_list, loss_D_list = train_cdcgan(
        G, D, cond_encoder, train_loader, criterion, opt_G, opt_D,
        latent_dim=latent_dim, device=device, epochs=epochs,
        sample_dir=f"{out_dir}/samples",
        label_smooth_real=0.9,
        label_smooth_fake=0.0,
        instance_noise_std=0.1,
        instance_noise_decay=0.99,
    )

    print(f"[GPU {gpu_id}] A={A} seed={seed}: Saving visualizations...")
    plot_loss(loss_G_list, loss_D_list, f"{out_dir}/loss.png")
    save_samples_grid(G, cond_encoder, train_loader, latent_dim, device, f"{out_dir}/samples_grid.png", n_samples=30)
    plot_comparison(G, cond_encoder, train_loader, latent_dim, device, f"{out_dir}/comparison.png")
    plot_interpolation(G, cond_encoder, train_loader, latent_dim, device, f"{out_dir}/interpolation.png")
    save_samples_grid(G, cond_encoder, train_loader, latent_dim, device, f"{out_dir}/samples/final.png", n_samples=20)
    plot_cond_pca(cond_encoder, train_loader, device, f"{out_dir}/cond_pca.png")
    save_samples_grid_pca(G, cond_encoder, train_loader, latent_dim, device, f"{out_dir}/samples_grid_pca.png", n_samples=40)
    classifier = get_pretrained_mnist_classifier(device)
    dataset.only_zero_action = True
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    excluded_pairs_accuracy = test_model_excluded_pairs(G, cond_encoder, dataset, classifier, latent_dim, device, f"{out_dir}/samples_excluded_pairs.png", n_samples=10)
    save_trial_statistics(G, cond_encoder, test_loader, latent_dim, device, f"{out_dir}/trial_statistics.pkl", classifier=classifier, excluded_pairs_accuracy=excluded_pairs_accuracy)

    print(f"[GPU {gpu_id}] A={A} seed={seed}: Done.")
    return out_dir


def main():
    num_processes = 16
    num_gpus = 8
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set
    tasks = list(itertools.product(np.arange(10), np.arange(5,120)))
    # Assign GPU: task_idx % 8 -> 2 processes per GPU
    task_args = [(i, A, seed) for i, (A, seed) in enumerate(tasks)]

    print(f"Running {len(tasks)} tasks with {num_processes} processes across {num_gpus} GPUs")
    with mp.Pool(num_processes) as pool:
        pool.map(_run_single_task, task_args)
    print("All tasks completed.")
    plot_trial_statistics("figures_cDCGAN")


if __name__ == "__main__":
    # main()
    plot_trial_statistics("figures_cDCGAN")
    # _run_single_task((0, 0, 0))
