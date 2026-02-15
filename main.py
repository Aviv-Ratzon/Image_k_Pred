
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch import optim
from torchvision.utils import make_grid, save_image
from sklearn.decomposition import PCA

import os


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
        self.action_dim = 2*A+1
        self.onehot = lambda x: torch.eye(self.action_dim)[x+A].float()
        self.single_output_sample = False
        
        # Load MNIST dataset
        self.dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        # subsample_indices = list(range(1000))
        # self.dataset = torch.utils.data.Subset(self.dataset, subsample_indices)
        
        self.data_idx_by_class = {i: [] for i in range(10)}
        self.images = []
        self.labels = []
        self.group_idx = []
        # Group data by class
        for idx, (image, label) in enumerate(self.dataset):
            self.group_idx.append(len(self.data_idx_by_class[label]))
            self.data_idx_by_class[label].append(idx)
            self.images.append(image)
            self.labels.append(label)
        
        # for i in range(10):
        #     print(f'There are {len(self.data_idx_by_class[i])} images for class {i}')
        
        # print(f"Created dataset with {len(self.images)} samples")
        print(f"Action range: [-{A}, {A}], Cyclic: {cyclic}, Image actions: {image_actions}, samples: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].flatten()
        label = self.labels[idx]
        # image_idx = self.data_idx_by_class[label][0]
        # image = self.images[image_idx].flatten()
        if self.cyclic:
            action = random.randint(-self.A, self.A)
            target_label = (label + action) % 10
        else:
            action = random.randint(max(-label, -self.A), min(9 - label, self.A))
            target_label = label + action
        
        action_obs = self.onehot(action)  # shape (2A+1,)
        if self.single_output_sample:
            target_image_idx = self.data_idx_by_class[target_label][0]
        else:
            target_image_idx = self.data_idx_by_class[target_label][self.group_idx[idx] % len(self.data_idx_by_class[target_label])]
        target_image = self.images[target_image_idx].flatten()
        return image, label, target_image, target_label, action, action_obs


class Encoder(nn.Module):
    """
    Encoder E(x, a) that maps input image and action to latent vector z.
    
    Args:
        input_channels: Number of input channels (1 for MNIST)
        action_dim: Dimension of action vector (2*A+1)
        latent_dim: Dimension of latent vector z
    """
    
    def __init__(self, input_channels=1, action_dim=5, latent_dim=64, n_layers=2, hidden_dim=512, image_actions=False):
        super(Encoder, self).__init__()
        self.image_actions = image_actions
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Unflatten(1, (1, 28, 28)),
            nn.Conv2d(input_channels, 32, 4, 2, 1),  # 28x28 -> 14x14
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 -> 7x7
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 7x7 -> 3x3
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, 1, 0),  # 3x3 -> 1x1
            nn.ReLU(),
        )
        
        # Action encoder
        if image_actions:
            self.action_encoder = nn.Sequential(
                nn.Unflatten(1, (1, 28, 28)),
                nn.Conv2d(input_channels, 32, 4, 2, 1),  # 28x28 -> 14x14
                nn.ReLU(),
                
                nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 -> 7x7
                nn.ReLU(),
                
                nn.Conv2d(64, 128, 4, 2, 1),  # 7x7 -> 3x3
                nn.ReLU(),
                
                nn.Conv2d(128, 256, 3, 1, 0),  # 3x3 -> 1x1
                nn.ReLU(),
            )
        else:
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
        
        # Combined encoder
        action_encoder_dim = 256 if image_actions else 64
        image_encoder_dim = 256
        combined_layers = []
        input_dim = image_encoder_dim + action_encoder_dim
        for i in range(n_layers - 1):
            combined_layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            combined_layers.append(nn.LeakyReLU(0.2))
        combined_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.combined_encoder = nn.Sequential(*combined_layers)
    
    def forward(self, x, a):
        # Encode image
        img_features = self.image_encoder(x)  # [B, 256, 1, 1]
        img_features = img_features.view(img_features.size(0), -1)  # [B, 256]
        
        # Encode action
        action_features = self.action_encoder(a)  # [B, 64]
        if self.image_actions:
            action_features = action_features.view(action_features.size(0), -1)  # [B, 256]
        
        # Combine features
        combined = torch.cat([img_features, action_features], dim=1)  # [B, 320]
        z = self.combined_encoder(combined)  # [B, latent_dim]
        
        return z


class Decoder(nn.Module):
    """
    Generator G(z) that maps latent vector to output image.
    
    Args:
        latent_dim: Dimension of latent vector z
        output_channels: Number of output channels (1 for MNIST)
    """
    
    def __init__(self, latent_dim=64, output_channels=1):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(),
        )
        
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 7x7 -> 14x14
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 14x14 -> 28x28
            nn.ReLU(),
            
            nn.Conv2d(32, output_channels, 3, 1, 1),
            nn.Tanh(),
        )
    
    def forward(self, z):
        h = self.decoder(z)  # [B, 7*7*128]
        x = h.view(h.size(0), 128, 7, 7)  # [B, 128, 7, 7]
        x = self.conv_decoder(x)  # [B, 1, 28, 28]
        return x.view(x.size(0), -1), h


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for predicting a target image x_next given a condition c.

    We use:
      - x = target image (flattened MNIST, normalized to [-1, 1])
      - c = concat(source_image, action)

    Encoder approximates q(z | x, c); decoder models p(x | z, c).
    """

    def __init__(self, image_dim: int, latent_dim: int = 32, hidden_dim: int = 1024, action_dim: int = 1, image_actions=False):
        super().__init__()
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        enc_in = image_dim + action_dim
        # Encoder: 3 conv layers (with ReLU), then flatten, then fully connected
        self.encoder = nn.Sequential(
            nn.Linear(image_dim + self.action_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(True),
        )
        # self.fc_enc = nn.Sequential(
        #     nn.Linear(hidden_dim + self.action_dim, hidden_dim//2),
        #     nn.ReLU(True),
        # )
        # self.encoder = Encoder(input_channels=1, action_dim=action_dim, latent_dim=hidden_dim//2, n_layers=4, hidden_dim=hidden_dim//2, image_actions=image_actions)
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)

        dec_in = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )

        self.fc_out =  nn.Sequential(
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh(),
        )
        # self.decoder = Decoder(latent_dim=latent_dim, output_channels=1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, cond], dim=1))
        # h = self.fc_enc(h)
        return self.fc_mu(h), self.fc_logvar(h), h

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        return self.fc_out(h), h
        # out, h = self.decoder(z)
        # return out, h

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, h_encoder = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon, h_decoder = self.decode(z)
        return recon, mu, logvar, z, h_encoder, h_decoder


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Increase A for more interesting "action -> target digit" transitions.
    # Set A=0 to make target always match the source digit.
    
    for A in [9]:
        print(f"Running for A={A}")
        base_folder = f"figures/A_{A}"
        os.makedirs(base_folder, exist_ok=True)
        import shutil
        for subdir in ["samples", "pca", "pca_centers"]:
            if os.path.exists(f"{base_folder}/{subdir}"):
                shutil.rmtree(f"{base_folder}/{subdir}")
            os.makedirs(f"{base_folder}/{subdir}", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        image_actions = False
        dataset = MNISTActionDataset(A=A, transform=transform, cyclic=False, image_actions=image_actions)
        image, label, target_image, target_label, action, one_hot_action = dataset[0]

        # fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        # axs[0].imshow(image.reshape(28, 28), cmap='gray')
        # fig.suptitle(f'Label: {label}, Target Label: {target_label}, Action: {action}')
        # im = axs[1].imshow(target_image.reshape(28, 28), cmap='gray')
        # fig.colorbar(im, ax=axs[1])
        # plt.show()

        model = ConditionalVAE(image_dim=image.shape[0], latent_dim=32, hidden_dim=1024, action_dim=one_hot_action.shape[0], image_actions=image_actions).to(device)

        # ----------------------------jan
        # Loss + Optimizers 
        # ----------------------------
        lr = 0.0002
        epochs = 10
        batch_size = 64
        beta_kl = 0.025 # KL weight (beta-VAE); tune up/down to trade off sharpness vs structure

        recon_criterion = nn.L1Loss()

        opt = optim.AdamW(model.parameters(), lr=lr)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_total_l = []
        loss_recon_l = []
        loss_kl_l = []
        for epoch in range(1, epochs + 1):
            loss_total_epoch = 0
            loss_recon_epoch = 0
            loss_kl_epoch = 0
            for real_imgs, labels, target_imgs, target_labels, actions, action_obs in train_loader:
                real_imgs = real_imgs.to(device)
                labels = labels.to(device)
                target_imgs = target_imgs.to(device)
                target_labels = target_labels.to(device)
                actions = actions.to(device).float()
                action_obs = action_obs.to(device)
                # Condition: (source image, action)

                # ------------------------
                # Train cVAE
                # ------------------------
                recon, mu, logvar, z, h_encoder, h_decoder = model(real_imgs, action_obs)

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
            loss_total_l.append(loss_total_epoch / len(train_loader))
            loss_recon_l.append(loss_recon_epoch / len(train_loader))
            loss_kl_l.append(loss_kl_epoch / len(train_loader))

            # Save sample grid each epoch
            with torch.no_grad():
                
                hidden_encoder_l = []
                hidden_decoder_l = []
                z_l = []
                mu_l = []
                target_label_l = []
                action_l = []
                for real_imgs, labels, target_imgs, target_labels, actions, action_obs in train_loader:
                    real_imgs = real_imgs.to(device)
                    labels = labels.to(device)
                    target_imgs = target_imgs.to(device)
                    target_labels = target_labels.to(device)
                    actions = actions.to(device).float()
                    action_obs = action_obs.to(device)
                    # Condition: (source image, action)

                    # ------------------------
                    # Train cVAE
                    # ------------------------
                    recon, mu, logvar, z, h_encoder, h_decoder = model(real_imgs, action_obs)

                    # Use mu as a stable "hidden state" for PCA visualization
                    hidden_encoder_l.append(h_encoder.detach().cpu().numpy())
                    hidden_decoder_l.append(h_decoder.detach().cpu().numpy())
                    z_l.append(z.detach().cpu().numpy())
                    mu_l.append(mu.detach().cpu().numpy())
                    target_label_l.append(target_labels.detach().cpu().numpy())
                    action_l.append(actions.detach().cpu().numpy())


                # Reconstructions (posterior) for reference
                recon_vis, _, _, _, _, _ = model(real_imgs, action_obs)

                recons = recon_vis.view(-1, 28, 28)
            print(
                f"Epoch {epoch:03d} | loss_total: {loss_total_l[-1]:.4f} | "
                f"recon: {loss_recon_l[-1]:.4f} | kl: {loss_kl_l[-1]:.4f}"
            )

        print('Finished training, plotting loss, PCA and samples...')

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(loss_total_l, label='loss_total')
        ax.plot(loss_recon_l, label='loss_recon')
        ax.plot(loss_kl_l, label='loss_kl')
        plt.yscale('log')
        ax.legend()
        plt.savefig(f"{base_folder}/loss.png")
        plt.close()

        n = 10
        fig, axs_all = plt.subplots(n, 3*2+1, figsize=(2*(3*2+1), 2*n))
        idx = 0
        for j in range(2):
            for i in range(n):
                axs = axs_all[i, j*4:j*4+3]
                axs[0].imshow(real_imgs[idx].reshape(28, 28).cpu().numpy(), cmap='gray')
                axs[0].set_ylabel(f's: {labels[idx].item()}, s_next: {target_labels[idx].item()}, a: {actions[idx].item()}')

                axs[1].imshow(target_imgs[idx].reshape(28, 28).cpu().numpy(), cmap='gray')
                axs[1].set_title("target")

                axs[2].imshow(recons[idx].cpu().numpy(), cmap='gray')
                axs[2].set_title("reconstructed (posterior)")

                axs[0].axis('off')
                axs[1].axis('off')
                axs[2].axis('off')
                idx += 1
        fig.tight_layout()
        plt.savefig(f"{base_folder}/samples.png")
        plt.close()
    

        # INSERT_YOUR_CODE
        # PCA plot of hidden_l colored by target_label_l
        # INSERT_YOUR_CODE
        for var, var_name in zip([z_l, hidden_encoder_l, hidden_decoder_l, mu_l], ['z', 'hidden_encoder', 'hidden_decoder', 'mu']):
            # Transform lists to numpy arrays for plotting
            var_np = np.concatenate(var, axis=0)
            label_np = np.concatenate(target_label_l, axis=0)
            action_np = np.concatenate(action_l, axis=0)

            # Run PCA on CPU using sklearn

            pca = PCA(n_components=4)
            var_pca = pca.fit_transform(var_np)

            explained_variance_ratio = pca.explained_variance_ratio_

            fig, axs_all = plt.subplots(2, 2, figsize=(10, 10))
            for pcs_i in range(2):
                axs = axs_all[pcs_i]
                for ax, color_var, color_label in zip(axs, [label_np, action_np], ['target label', 'action']):
                    scatter = ax.scatter(var_pca[:, pcs_i*2], var_pca[:, pcs_i*2+1], c=color_var, cmap='coolwarm', alpha=0.7)
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(color_label)
                    ax.set_xlabel(f'Component {2*pcs_i+1} ({100*explained_variance_ratio[2*pcs_i]:.2f}%)')
                    ax.set_ylabel(f'Component {2*pcs_i+2} ({100*explained_variance_ratio[2*pcs_i+1]:.2f}%)')
                    ax.set_title(f"PCA of {var_name} colored by {color_label}")
            plt.tight_layout()
            fig.savefig(f"{base_folder}/pca/pca_{var_name}.png")
            plt.close()

            filder = action_np == 0
            var_pca_a0 = var_pca[filder]
            label_np_a0 = label_np[filder]
            action_np_a0 = action_np[filder]

            fig, axs_all = plt.subplots(2, 2, figsize=(10, 10))
            for pcs_i in range(2):
                axs = axs_all[pcs_i]
                for ax, color_var, color_label in zip(axs, [label_np_a0, action_np_a0], ['target label', 'action']):
                    scatter = ax.scatter(var_pca_a0[:, pcs_i*2], var_pca_a0[:, pcs_i*2+1], c=color_var, cmap='coolwarm', alpha=0.7)
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(color_label)
                    ax.set_xlabel(f'Component {2*pcs_i+1} ({100*explained_variance_ratio[2*pcs_i]:.2f}%)')
                    ax.set_ylabel(f'Component {2*pcs_i+2} ({100*explained_variance_ratio[2*pcs_i+1]:.2f}%)')
                    ax.set_title(f"PCA of {var_name} colored by {color_label}")
                    ax.axis('equal')
            plt.tight_layout()
            fig.savefig(f"{base_folder}/pca/pca_{var_name}_a0.png")
            plt.close()


            fig, axs_all = plt.subplots(2, 2, figsize=(10, 10))
            for pcs_i in range(2):
                axs = axs_all[pcs_i]
                for ax, color_var, color_label in zip(axs, [label_np, action_np], ['target label', 'action']):
                    var_pca_centers = np.array([var_pca[color_var == i].mean(axis=0) for i in np.unique(color_var)])
                    scatter = ax.scatter(var_pca_centers[:, pcs_i*2], var_pca_centers[:, pcs_i*2+1], c=np.unique(color_var), cmap='coolwarm', alpha=0.7, s=100, edgecolors='black')
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(color_label)
                    ax.set_xlabel(f'Component {2*pcs_i+1} ({100*explained_variance_ratio[2*pcs_i]:.2f}%)')
                    ax.set_ylabel(f'Component {2*pcs_i+2} ({100*explained_variance_ratio[2*pcs_i+1]:.2f}%)')
                    ax.set_title(f"PCA of {var_name} colored by {color_label}")
                    ax.axis('equal')
            plt.tight_layout()
            fig.savefig(f"{base_folder}/pca_centers/pca_{var_name}_centers.png")
            plt.close()

            filder = action_np == 0
            var_pca_a0 = var_pca[filder]
            label_np_a0 = label_np[filder]
            action_np_a0 = action_np[filder]

            fig, axs_all = plt.subplots(2, 2, figsize=(10, 10))
            for pcs_i in range(2):
                axs = axs_all[pcs_i]
                for ax, color_var, color_label in zip(axs, [label_np_a0, action_np_a0], ['target label', 'action']):
                    var_pca_centers = np.array([var_pca_a0[color_var == i].mean(axis=0) for i in np.unique(color_var)])
                    scatter = ax.scatter(var_pca_centers[:, pcs_i*2], var_pca_centers[:, pcs_i*2+1], c=np.unique(color_var), cmap='coolwarm', alpha=0.7, s=100, edgecolors='black')
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(color_label)
                    ax.set_xlabel(f'Component {2*pcs_i+1} ({100*explained_variance_ratio[2*pcs_i]:.2f}%)')
                    ax.set_ylabel(f'Component {2*pcs_i+2} ({100*explained_variance_ratio[2*pcs_i+1]:.2f}%)')
                    ax.set_title(f"PCA of {var_name} colored by {color_label}")
                    ax.axis('equal')
            plt.tight_layout()
            fig.savefig(f"{base_folder}/pca_centers/pca_{var_name}_a0_centers.png")
            plt.close()
