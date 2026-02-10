
import torch
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import random

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch import optim
from torchvision.utils import make_grid, save_image

import os

os.makedirs("figures", exist_ok=True)
import shutil
for subdir in ["samples", "pca"]:
    if os.path.exists(f"figures/{subdir}"):
        shutil.rmtree(f"figures/{subdir}")
    os.makedirs(f"figures/{subdir}", exist_ok=True)
os.makedirs("data", exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNISTActionDataset(Dataset):
    """
    Dataset for MNIST with action-based transformations.
    
    Args:
        N: Number of samples per digit class
        A: Action range [-A, A]
        cyclic: Whether to use cyclic addition (mod 10)
        transform: Image transformations
    """
    
    def __init__(self, A=2, cyclic=False, transform=None, image_actions=False):
        self.A = A
        self.cyclic = cyclic
        self.transform = transform
        self.image_actions = image_actions
        
        # Load MNIST dataset
        self.dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        
        self.data_idx_by_class = {i: [] for i in range(10)}
        self.images = []
        self.labels = []
        # Group data by class
        for idx, (image, label) in enumerate(self.dataset):
            self.data_idx_by_class[label].append(idx)
            self.images.append(image)
            self.labels.append(label)
        
        for i in range(10):
            print(f'There are {len(self.data_idx_by_class[i])} images for class {i}')
        
        print(f"Created dataset with {len(self.images)} samples")
        print(f"Action range: [-{A}, {A}], Cyclic: {cyclic}, Image actions: {image_actions}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].flatten()
        label = self.labels[idx]
        if self.cyclic:
            action = random.randint(-self.A, self.A)
            target_label = (label + action) % 10
        else:
            action = random.randint(max(-label, -self.A), min(9 - label, self.A))
            target_label = label + action
        
        target_image_idx = random.choice(self.data_idx_by_class[target_label])
        target_image = self.images[target_image_idx].flatten()
        return image, label, target_image, target_label, action


class Generator(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(out_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(True),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(1024+1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            # nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(True),

            nn.Linear(1024, out_dim),
            nn.Tanh()
        )

    def forward(self, image, action):
        z = self.encoder(image)
        z = self.action_encoder(torch.cat([z, action], dim=1))
        out = self.decoder(z)
        return out, z

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3),

            nn.Linear(256, 11)  # logits (no sigmoid here)
        )

    def forward(self, x):
        return self.net(x)



class ConditionalVAE(nn.Module):
    """
    Conditional VAE for predicting a target image x_next given a condition c.

    We use:
      - x = target image (flattened MNIST, normalized to [-1, 1])
      - c = concat(source_image, action)

    Encoder approximates q(z | x, c); decoder models p(x | z, c).
    """

    def __init__(self, image_dim: int, latent_dim: int = 32, hidden_dim: int = 1024):
        super().__init__()
        self.image_dim = image_dim
        self.latent_dim = latent_dim

        # Condition is (source_image + action)
        self.cond_dim = image_dim + 1

        enc_in = image_dim + self.cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        dec_in = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh(),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, cond], dim=1))
        return self.fc_mu(h), self.fc_logvar(h), h

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, h = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z, h


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Increase A for more interesting "action -> target digit" transitions.
    # Set A=0 to make target always match the source digit.
    A = 2
    dataset = MNISTActionDataset(A=A, transform=transform)
    image, label, target_image, target_label, action = dataset[0]

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    axs[0].imshow(image.reshape(28, 28), cmap='gray')
    fig.suptitle(f'Label: {label}, Target Label: {target_label}, Action: {action}')
    im = axs[1].imshow(target_image.reshape(28, 28), cmap='gray')
    fig.colorbar(im, ax=axs[1])
    plt.show()

    model = ConditionalVAE(image_dim=image.shape[0], latent_dim=32, hidden_dim=1024).to(device)

    # ----------------------------
    # Loss + Optimizers
    # ----------------------------
    lr = 0.0002
    epochs = 20
    batch_size = 128
    beta_kl = 0.05  # KL weight (beta-VAE); tune up/down to trade off sharpness vs structure

    recon_criterion = nn.L1Loss()

    opt = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_total_l = []
    loss_recon_l = []
    loss_kl_l = []
    hidden_l = []
    target_label_l = []
    for epoch in range(1, epochs + 1):
        loss_total_epoch = 0
        loss_recon_epoch = 0
        loss_kl_epoch = 0
        for real_imgs, labels, target_imgs, target_labels, actions in train_loader:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            target_imgs = target_imgs.to(device)
            target_labels = target_labels.to(device)
            actions = actions.to(device).float()

            # Condition: (source image, action)
            cond = torch.cat([real_imgs, actions.unsqueeze(1)], dim=1)

            # ------------------------
            # Train cVAE
            # ------------------------
            recon, mu, logvar, z, h = model(target_imgs, cond)

            recon_loss = recon_criterion(recon, target_imgs)
            # KL(q(z|x,c) || N(0,1)) averaged over batch
            kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

            total_loss = recon_loss + beta_kl * kl_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            loss_total_epoch += total_loss.item()
            loss_recon_epoch += recon_loss.item()
            loss_kl_epoch += kl_loss.item()

            # Use mu as a stable "hidden state" for PCA visualization
            hidden_l.append(h.detach().cpu().numpy())
            target_label_l.append(target_labels.detach().cpu().numpy())

        loss_total_l.append(loss_total_epoch / len(train_loader))
        loss_recon_l.append(loss_recon_epoch / len(train_loader))
        loss_kl_l.append(loss_kl_epoch / len(train_loader))

        # Save sample grid each epoch
        with torch.no_grad():
            # Prior samples for conditional generation (not just reconstruction)
            z_prior = torch.randn(real_imgs.shape[0], model.latent_dim, device=device)
            gen = model.decode(z_prior)

            # Reconstructions (posterior) for reference
            recon_vis, _, _, _, _ = model(target_imgs, cond)

            samples = gen.view(-1, 1, 28, 28)
            recons = recon_vis.view(-1, 1, 28, 28)

        n = 5
        fig, axs_all = plt.subplots(n, 3, figsize=(10, 3*n))
        for i in range(n):
            axs = axs_all[i]
            axs[0].imshow(real_imgs[i].reshape(28, 28).cpu().numpy(), cmap='gray')
            axs[0].set_ylabel(f's: {labels[i].item()}, s_next: {target_labels[i].item()}, a: {actions[i].item()}')
            axs[0].set_title("source")

            axs[1].imshow(target_imgs[i].reshape(28, 28).cpu().numpy(), cmap='gray')
            axs[1].set_title("target")

            axs[2].imshow(samples[i].reshape(28, 28).cpu().numpy(), cmap='gray')
            axs[2].set_title("generated (prior)")
            axs[0].axis('off')
            axs[1].axis('off')
            axs[2].axis('off')
        fig.tight_layout()
        plt.savefig(f"figures/samples/samples_epoch_{epoch:03d}.png")
        plt.close()

        # INSERT_YOUR_CODE
        # PCA plot of hidden_l colored by target_label_l
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.decomposition import PCA

        # Transform lists to numpy arrays for plotting
        hidden_np = np.concatenate(hidden_l, axis=0)
        label_np = np.concatenate(target_label_l, axis=0)

        # Run PCA
        pca = PCA(n_components=2)
        hidden_pca = pca.fit_transform(hidden_np)

        plt.figure(figsize=(7, 5))
        scatter = plt.scatter(hidden_pca[:, 0], hidden_pca[:, 1], c=label_np, cmap='coolwarm', alpha=0.7)
        cbar = plt.colorbar(scatter, ticks=np.unique(label_np))
        cbar.set_label('Target label')
        plt.xlabel(f'Component 1 ({100*pca.explained_variance_ratio_[0]:.2f}%)')
        plt.ylabel(f'Component 2 ({100*pca.explained_variance_ratio_[1]:.2f}%)')
        plt.title("PCA of Hidden States colored by Target Label")
        plt.tight_layout()
        plt.savefig(f"figures/pca/pca_hidden_epoch_{epoch:03d}.png")
        plt.close()


        print(
            f"Epoch {epoch:03d} | loss_total: {loss_total_l[-1]:.4f} | "
            f"recon: {loss_recon_l[-1]:.4f} | kl: {loss_kl_l[-1]:.4f}"
        )


    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(loss_total_l, label='loss_total')
    ax.plot(loss_recon_l, label='loss_recon')
    ax.plot(loss_kl_l, label='loss_kl')
    plt.yscale('log')
    ax.legend()
    plt.savefig(f"figures/loss.png")
    plt.show()

    n = 5
    fig, axs_all = plt.subplots(n, 3, figsize=(10, 3*n))
    for i in range(n):
        axs = axs_all[i]
        axs[0].imshow(real_imgs[i].reshape(28, 28).cpu().numpy(), cmap='gray')
        axs[0].set_ylabel(f's: {labels[i].item()}, s_next: {target_labels[i].item()}, a: {actions[i].item()}')

        axs[1].imshow(target_imgs[i].reshape(28, 28).cpu().numpy(), cmap='gray')
        axs[1].set_title("target")

        axs[2].imshow(samples[i].reshape(28, 28).cpu().numpy(), cmap='gray')
        axs[2].set_title("generated (prior)")

        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
    fig.tight_layout()
    plt.savefig(f"figures/samples.png")
    plt.show()
