
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
from sklearn.metrics.pairwise import euclidean_distances

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
    Encoder: maps (real_imgs, action) -> h_encoder for the conditional GAN generator.
    Input: cond = concat(real_imgs, action_onehot)
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 1024, output_dim: int = 64):
        super().__init__()
        img_dim = input_dim[0]
        action_dim = input_dim[1]
        self.conv = nn.Sequential(
            nn.Unflatten(1, (1, 28, 28)),
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.LeakyReLU(0.2),
        )
        conv_out_dim = 256

        self.mlp = nn.Sequential(
            nn.Linear(conv_out_dim+action_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        img = cond[0]
        action = cond[1]
        img_features = self.conv(img)
        img_features = img_features.view(img_features.size(0), -1)
        combined = torch.cat([img_features, action], dim=1)
        return self.mlp(combined)


class Decoder(nn.Module):
    """
    Decoder: maps (h_encoder, noise) -> target image.
    Returns (x_fake, h_decoder) where h_decoder is the prev-to-last layer activation for PCA.
    """

    def __init__(self, encoder_dim: int, image_dim: int, noise_dim: int = 64, hidden_dim: int = 1024):
        super().__init__()
        self.noise_dim = noise_dim
        dec_in = encoder_dim + noise_dim  # <- encoder features + noise

        self.mlp = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh(),
        )

    def forward(self, h_encoder: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h_encoder: [B, encoder_dim], z: [B, noise_dim]
        h_in = torch.cat([h_encoder, z], dim=1)
        h = self.mlp(h_in)   # h_decoder = prev-to-last
        x_fake = self.fc_out(h)
        return x_fake, h



class Discriminator(nn.Module):
    """Discriminator: (x_candidate) -> real/fake logits."""

    def __init__(self, image_dim: int, hidden_dim: int = 1024, cond_dim: int = 10, encoder_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(image_dim + encoder_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        # self.cond_encoder = nn.Embedding(10, 10)
        self.cond_encoder = Encoder(cond_dim, hidden_dim, encoder_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h_encoder = self.cond_encoder(cond)
        combined = torch.cat([x, h_encoder], dim=1)
        x_features = self.mlp(combined)
        return self.fc_out(x_features)
    

class ConditionalGAN(nn.Module):
    """
    Conditional GAN: receives (real_imgs, action), outputs target images.
    Generator = Encoder + Decoder. Encoder(cond) -> h_encoder; Decoder(h_encoder, noise) -> x_fake.
    """

    def __init__(
        self,
        image_dim: int,
        cond_dim: int = 10,
        encoder_dim: int = 256,
        noise_dim: int = 64,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        # self.encoder = Encoder(image_dim, action_dim, encoder_dim, hidden_dim)
        # self.encoder = nn.Embedding(self.cond_dim, encoder_dim)
        self.cond_encoder = Encoder(cond_dim, hidden_dim, encoder_dim)
        self.decoder = Decoder(encoder_dim, image_dim, noise_dim, hidden_dim)
        self.noise_dim = noise_dim

    def forward(
        self,
        cond,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (x_fake, h_encoder, h_decoder) for PCA.
        This samples a fresh z each call.
        cond: [real_imgs, action_obs] or tensor
        """
        B = cond[0].size(0) if isinstance(cond, (list, tuple)) else cond.size(0)
        h_encoder = self.cond_encoder(cond)
        dev = cond[0].device if isinstance(cond, (list, tuple)) else cond.device
        z = torch.randn(B, self.noise_dim, device=dev)
        x_fake, h_decoder = self.decoder(h_encoder, z)
        return x_fake, h_encoder, h_decoder

    def generate(
        self,
        cond: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        For each (real_img, action), generate n_samples with different z.
        Output shape: [n_samples, B, image_dim] if n_samples > 1, else [B, image_dim].
        """
        B = cond[0].size(0) if isinstance(cond, (list, tuple)) else cond.size(0)
        dev = cond[0].device if isinstance(cond, (list, tuple)) else cond.device
        samples = []
        with torch.no_grad():
            # Encode once, reuse h_encoder for all z’s
            h_encoder = self.cond_encoder(cond)
            for _ in range(n_samples):
                z = torch.randn(B, self.noise_dim, device=dev)
                x_fake, _ = self.decoder(h_encoder, z)
                samples.append(x_fake)

        if n_samples == 1:
            return samples[0]  # [B, image_dim]
        return torch.stack(samples, dim=0)  # [n_samples, B, image_dim]



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Increase A for more interesting "action -> target digit" transitions.
    # Set A=0 to make target always match the source digit.
    
    for A in [5]:
        print(f"Running for A={A}")
        base_folder = f"figures_GAN/A_{A}"
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

        encoder_dim = 128
        noise_dim = 100  # you can tune this
        cond_dim = [image.shape[0], one_hot_action.shape[0]]
        
        G = ConditionalGAN(
            image_dim=image.shape[0],
            cond_dim=cond_dim,
            encoder_dim=encoder_dim,
            noise_dim=noise_dim,
            hidden_dim=1024,
        ).to(device)
        D = Discriminator(
            image_dim=image.shape[0],
            hidden_dim=1024,
            cond_dim=cond_dim,
            encoder_dim=encoder_dim,
        ).to(device)

        # ----------------------------
        # Hyperparameters
        # ----------------------------
        lr = 2e-4
        epochs = 30
        batch_size = 128

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        criterion_rec = nn.L1Loss()

        # lambda_rec = 10.0

        opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


        loss_D_l = []
        loss_G_l = []

        def get_cond(real_imgs, labels, target_imgs, target_labels, actions, action_obs):
            # cond = torch.cat([real_imgs, action_obs], dim=1)
            cond = [real_imgs, action_obs]
            # cond = target_labels
            return cond

        for epoch in range(1, epochs + 1):

            
            with torch.no_grad():
                batch_vis = next(iter(train_loader))
                real_imgs, labels, target_imgs, target_labels, actions, action_obs = [x.to(device) for x in batch_vis]
                cond = get_cond(real_imgs, labels, target_imgs, target_labels, actions, action_obs)
                x_fake_vis, _, _ = G(cond)
                recons = x_fake_vis.view(-1, 28, 28).cpu().numpy()

            # Samples: source | target | G(source, action, noise)
            n = 10
            fig, axs_all = plt.subplots(n, 6, figsize=(2*6, 2*n))
            idx = 0
            for j in range(2):
                for i in range(n):
                    axs = axs_all[i, j*3:(j+1)*3]
                    axs[0].imshow(real_imgs[idx].reshape(28, 28).cpu().numpy(), cmap='gray')
                    axs[0].set_ylabel(f's: {labels[idx].item()}, s_next: {target_labels[idx].item()}, a: {actions[idx].item()}')

                    axs[1].imshow(target_imgs[idx].reshape(28, 28).cpu().numpy(), cmap='gray')
                    axs[1].set_title("Target")

                    axs[2].imshow(recons[idx], cmap='gray')
                    axs[2].set_title("Generated")

                    axs[0].axis('off')
                    axs[1].axis('off')
                    axs[2].axis('off')
                    idx += 1
            fig.tight_layout()
            plt.savefig(f"{base_folder}/samples/samples_epoch_{epoch:03d}.png")
            plt.close()

            loss_D_epoch = 0
            loss_G_epoch = 0

            for real_imgs, labels, target_imgs, target_labels, actions, action_obs in train_loader:
                real_imgs = real_imgs.to(device)
                labels = labels.to(device)
                target_imgs = target_imgs.to(device)
                target_labels = target_labels.to(device)
                actions = actions.to(device).float()
                action_obs = action_obs.to(device)
                B = real_imgs.size(0)

                cond = get_cond(real_imgs, labels, target_imgs, target_labels, actions, action_obs)

                # ---- Discriminator update ----
                x_fake, _, _ = G(cond)
                D_real = D(target_imgs, cond)
                D_fake = D(x_fake.detach(), cond)

                labels_fake = torch.zeros_like(D_fake).float()
                labels_real = torch.ones_like(D_real).float()

                L_D_fake = criterion(D_fake, labels_fake)
                L_D_real = criterion(D_real, labels_real)
                L_D = L_D_fake + L_D_real

                opt_D.zero_grad()
                L_D.backward()
                opt_D.step()

                # ---- Generator update ----
                x_fake, h_encoder, h_decoder = G(cond)
                D_fake = D(x_fake, cond)
                # L_G_rec = criterion_rec(x_fake, target_imgs)
                # L_G_cls = criterion(D_fake, labels_real)
                # L_G = L_G_cls + lambda_rec * L_G_rec
                L_G = criterion(D_fake, labels_real)

                opt_G.zero_grad()
                L_G.backward()
                opt_G.step()

                loss_D_epoch += L_D.item()
                loss_G_epoch += L_G.item()

            loss_D_l.append(loss_D_epoch / len(train_loader))
            loss_G_l.append(loss_G_epoch / len(train_loader))
            print(f"Epoch {epoch:03d} | D: {loss_D_l[-1]:.4f} | G: {loss_G_l[-1]:.4f}")

        # Collect encoder/decoder hidden states for PCA
        with torch.no_grad():
            hidden_encoder_l = []
            hidden_decoder_l = []
            target_label_l = []
            action_l = []
            for real_imgs, labels, target_imgs, target_labels, actions, action_obs in train_loader:
                real_imgs = real_imgs.to(device)
                labels = labels.to(device)
                target_imgs = target_imgs.to(device)
                target_labels = target_labels.to(device)
                actions = actions.to(device).float()
                action_obs = action_obs.to(device)

                cond = get_cond(real_imgs, labels, target_imgs, target_labels, actions, action_obs)
                x_fake, h_encoder, h_decoder = G(cond)

                hidden_encoder_l.append(h_encoder.detach().cpu().numpy())
                hidden_decoder_l.append(h_decoder.detach().cpu().numpy())
                target_label_l.append(target_labels.detach().cpu().numpy())
                action_l.append(actions.detach().cpu().numpy())

            # For samples figure
            batch_vis = next(iter(train_loader))
            real_imgs, labels, target_imgs, target_labels, actions, action_obs = [x.to(device) for x in batch_vis]
            cond = get_cond(real_imgs, labels, target_imgs, target_labels, actions, action_obs)
            x_fake_vis, _, _ = G(cond)
            recons = x_fake_vis.view(-1, 28, 28).cpu().numpy()

        print('Finished training, plotting loss, PCA and samples...')

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(loss_D_l, label='loss_D')
        ax.plot(loss_G_l, label='loss_G')
        plt.yscale('log')
        ax.legend()
        plt.savefig(f"{base_folder}/loss.png")
        plt.close()

        # Samples: source | target | G(source, action, noise)
        n = 10
        fig, axs_all = plt.subplots(n, 6, figsize=(2*6, 2*n))
        idx = 0
        for j in range(2):
            for i in range(n):
                axs = axs_all[i, j*3:(j+1)*3]
                axs[0].imshow(real_imgs[idx].reshape(28, 28).cpu().numpy(), cmap='gray')

                axs[1].imshow(target_imgs[idx].reshape(28, 28).cpu().numpy(), cmap='gray')
                axs[1].set_title("Target")

                axs[2].imshow(recons[idx], cmap='gray')
                axs[2].set_title("Generated")

                axs[0].axis('off')
                axs[1].axis('off')
                axs[2].axis('off')
                idx += 1
        fig.tight_layout()
        plt.savefig(f"{base_folder}/samples.png")
        plt.close()

        # GAN samples: K noise vectors per (source, action)
        K = 5
        n_conds = 8
        with torch.no_grad():
            batch = next(iter(train_loader))
            real_imgs, labels, target_imgs, target_labels, actions, action_obs = [x.to(device) for x in batch]
            real_imgs = real_imgs[:n_conds]
            action_obs = action_obs[:n_conds]
            labels = labels[:n_conds]
            target_labels = target_labels[:n_conds]
            actions = actions[:n_conds]
            cond = get_cond(real_imgs, labels, target_imgs, target_labels, actions, action_obs)
            samples_prior = G.generate(cond, n_samples=K)

        fig, axs_all = plt.subplots(n_conds, 2 + K, figsize=(2*(2+K), 2*n_conds))
        for i in range(n_conds):
            axs_all[i, 0].imshow(real_imgs[i].reshape(28, 28).cpu().numpy(), cmap='gray')
            axs_all[i, 0].set_ylabel(f's:{labels[i].item()} s_next:{target_labels[i].item()} a:{actions[i].item()}')
            axs_all[i, 1].imshow(target_imgs[i].reshape(28, 28).cpu().numpy(), cmap='gray')
            axs_all[i, 1].set_title("target")
            for k in range(K):
                axs_all[i, 2+k].imshow(samples_prior[k, i].cpu().numpy().reshape(28, 28), cmap='gray')
                axs_all[i, 2+k].set_title(f"G sample {k+1}")
            for j in range(2 + K):
                axs_all[i, j].axis('off')
        fig.suptitle("GAN samples: Encoder(source,action)+Decoder(h,noise) - same (source,action), different z")
        fig.tight_layout()
        plt.savefig(f"{base_folder}/samples_prior.png")
        plt.close()
    

        # PCA: Encoder output and Decoder prev-to-last layer
        for var, var_name in zip([hidden_encoder_l, hidden_decoder_l], ['hidden_encoder', 'hidden_decoder']):
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

        # Distance matrices: hidden states ordered by target label
        hidden_encoder_np = np.concatenate(hidden_encoder_l, axis=0)
        hidden_decoder_np = np.concatenate(hidden_decoder_l, axis=0)
        label_np = np.concatenate(target_label_l, axis=0)
        sort_idx = np.argsort(label_np)
        hidden_encoder_sorted = hidden_encoder_np[sort_idx]
        hidden_decoder_sorted = hidden_decoder_np[sort_idx]
        # Subsample for memory (full 60k x 60k would be ~28GB)
        max_samples_dist = 5000
        if len(hidden_encoder_sorted) > max_samples_dist:
            step = len(hidden_encoder_sorted) // max_samples_dist
            idx_sub = np.arange(0, len(hidden_encoder_sorted), step)[:max_samples_dist]
            hidden_encoder_sorted = hidden_encoder_sorted[idx_sub]
            hidden_decoder_sorted = hidden_decoder_sorted[idx_sub]
        dist_encoder = euclidean_distances(hidden_encoder_sorted)
        dist_decoder = euclidean_distances(hidden_decoder_sorted)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axs[0].imshow(dist_encoder, cmap='viridis', aspect='auto')
        axs[0].set_title('Encoder hidden states (ordered by target label)')
        axs[0].set_xlabel('Sample index')
        axs[0].set_ylabel('Sample index')
        plt.colorbar(im0, ax=axs[0], label='Euclidean distance')
        im1 = axs[1].imshow(dist_decoder, cmap='viridis', aspect='auto')
        axs[1].set_title('Decoder hidden states (ordered by target label)')
        axs[1].set_xlabel('Sample index')
        axs[1].set_ylabel('Sample index')
        plt.colorbar(im1, ax=axs[1], label='Euclidean distance')
        fig.tight_layout()
        fig.savefig(f"{base_folder}/distance_matrix.png")
        plt.close()
