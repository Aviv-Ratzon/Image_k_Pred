
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



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = MNISTActionDataset(A=0, transform=transform)
    image, label, target_image, target_label, action = dataset[0]

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    axs[0].imshow(image.reshape(28, 28), cmap='gray')
    fig.suptitle(f'Label: {label}, Target Label: {target_label}, Action: {action}')
    im = axs[1].imshow(target_image.reshape(28, 28), cmap='gray')
    fig.colorbar(im, ax=axs[1])
    plt.show()

    G = Generator(out_dim=image.shape[0]).to(device)
    D = Discriminator(in_dim=image.shape[0]).to(device)

    # ----------------------------
    # Loss + Optimizers
    # ----------------------------
    lr = 0.0002
    epochs = 20
    batch_size = 128
    lambda_pixel = 0.1 # Hyperparameter to balance GAN vs Reconstruction

    criterion = nn.CrossEntropyLoss()
    L1_criterion = nn.L1Loss()
    
    opt_G = optim.Adam(G.parameters(), lr=lr)
    opt_D = optim.Adam(D.parameters(), lr=lr/2)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_D_l = []
    loss_G_l = []
    hidden_l = []
    target_label_l = []
    for epoch in range(1, epochs + 1):
        loss_D_epoch = 0
        loss_G_epoch = 0
        for real_imgs, labels, target_imgs, target_labels, actions in train_loader:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            target_imgs = target_imgs.to(device)
            target_labels = target_labels.to(device)
            actions = actions.to(device)

            # ------------------------
            # Train Discriminator
            # ------------------------
            fake_imgs, _ = G(real_imgs, actions.unsqueeze(1))  # detach so G isn't updated here

            real_labels = target_labels
            fake_labels = torch.zeros(len(real_labels), device=device).long() + 10

            D_real_logits = D(target_imgs)
            D_fake_logits = D(fake_imgs.detach())

            loss_D_real = criterion(D_real_logits, target_labels)
            loss_D_fake = criterion(D_fake_logits, fake_labels)
            loss_D = (loss_D_real + loss_D_fake) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ------------------------
            # Train Generator
            # ------------------------
            gen_imgs, hidden = G(real_imgs, actions.unsqueeze(1))
            # Want D(gen_imgs) -> "real" (1)
            D_gen_logits = D(gen_imgs)
            loss_G = criterion(D_gen_logits, real_labels)

            # Inside the Generator training block
            loss_pixel = L1_criterion(gen_imgs, target_imgs) # Force it to match the ground truth target

            # Combine losses
            total_loss_G = loss_G + (lambda_pixel * loss_pixel)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loss_D_epoch += loss_D.item()
            loss_G_epoch += loss_G.item()
            hidden_l.append(hidden.detach().cpu().numpy())
            target_label_l.append(target_labels.detach().cpu().numpy())

        loss_D_l.append(loss_D_epoch / len(train_loader))
        loss_G_l.append(loss_G_epoch / len(train_loader))

        # Save sample grid each epoch
        with torch.no_grad():
            samples = G(real_imgs, actions.unsqueeze(1))[0].view(-1, 1, 28, 28)
            # grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
            # save_image(grid, f"figures/samples/mnist_gan_epoch_{epoch:03d}.png")

        n = 5
        fig, axs_all = plt.subplots(n, 2, figsize=(7, 3*n))
        for i in range(n):
            axs = axs_all[i]
            axs[0].imshow(real_imgs[i].reshape(28, 28).cpu().numpy(), cmap='gray')
            axs[0].set_ylabel(f's: {labels[i].item()}, s_next: {target_labels[i].item()}, a: {actions[i].item()}')
            im = axs[1].imshow(samples[i].reshape(28, 28).cpu().numpy(), cmap='gray')
            axs[1].set_title(f'a: {actions[i].item()}, s: {target_labels[i].item()}')
            axs[0].axis('off')
            axs[1].axis('off')
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


        print(f"Epoch {epoch:03d} | loss_D: {loss_D.item():.4f} | loss_G: {loss_G.item():.4f}")


    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(loss_D_l, label='loss_D')
    ax.plot(loss_G_l, label='loss_G')
    plt.yscale('log')
    ax.legend()
    plt.savefig(f"figures/loss.png")
    plt.show()

    n = 5
    fig, axs_all = plt.subplots(n, 2, figsize=(7, 3*n))
    for i in range(n):
        axs = axs_all[i]
        axs[0].imshow(real_imgs[i].reshape(28, 28).cpu().numpy(), cmap='gray')
        axs[0].set_ylabel(f's: {labels[i].item()}, s_next: {target_labels[i].item()}, a: {actions[i].item()}')
        im = axs[1].imshow(samples[i].reshape(28, 28).cpu().numpy(), cmap='gray')
        axs[1].set_title(f'a: {actions[i].item()}, s: {target_labels[i].item()}')
        
        axs[0].axis('off')
        axs[1].axis('off')
    fig.tight_layout()
    plt.savefig(f"figures/samples.png")
    plt.show()
