
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
    Conditional VAE for predicting a target image x_target given condition c = [source_image, action_onehot].

    Encoder: q(z | x_target, cond)
    Decoder: p(x_target | z, cond)
    """

    def __init__(self, image_dim: int, latent_dim: int = 32, hidden_dim: int = 1024, action_dim: int = 1, image_actions=False):
        super().__init__()
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.cond_dim = image_dim + action_dim  # cond = [source_image, action_onehot]

        # Encoder input: concat(x_target, cond) -> image_dim + cond_dim
        enc_in = image_dim + self.cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)

        # Decoder input: concat(z, cond) -> latent_dim + cond_dim
        dec_in = latent_dim + self.cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh(),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x_target: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q(z | x_target, cond)"""
        h = self.encoder(torch.cat([x_target, cond], dim=1))
        return self.fc_mu(h), self.fc_logvar(h), h

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """p(x_target | z, cond)"""
        h = self.decoder(torch.cat([z, cond], dim=1))
        return self.fc_out(h), h

    def forward(
        self, x_target: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, h_encoder = self.encode(x_target, cond)
        z = self.reparameterize(mu, logvar)
        recon, h_decoder = self.decode(z, cond)
        return recon, mu, logvar, z, h_encoder, h_decoder

    def generate(
        self, source_imgs: torch.Tensor, action_obs: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """Inference: z ~ N(0,I), x_hat = decode(z, cond). Returns [B, 784] or [n_samples, B, 784]."""
        cond = torch.cat([source_imgs, action_obs], dim=1)
        B = source_imgs.size(0)
        samples = []
        for _ in range(n_samples):
            z = torch.randn(B, self.latent_dim, device=source_imgs.device, dtype=source_imgs.dtype)
            x_hat, _ = self.decode(z, cond)
            samples.append(x_hat)
        if n_samples == 1:
            return samples[0]
        return torch.stack(samples, dim=0)  # [n_samples, B, 784]


class MNISTClassifier(nn.Module):
    """Minimal CNN classifier for MNIST. Conv(1→32)+ReLU+MaxPool, Conv(32→64)+ReLU+MaxPool, FC→128+ReLU, FC→10."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28->14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14->7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, 28, 28], returns logits [B, 10]"""
        h = self.conv(x)
        return self.fc(h)


def pretrain_classifier(clf: nn.Module, train_loader: DataLoader, epochs: int = 5, device=None):
    """Pretrain classifier on (image, label) from MNIST. Uses target_imgs/target_labels as proxy for digit."""
    if device is None:
        device = next(clf.parameters()).device
    clf.train()
    opt = optim.Adam(clf.parameters(), lr=0.001)
    for epoch in range(epochs):
        correct, total = 0, 0
        for _, _, target_imgs, target_labels, _, _ in train_loader:
            target_imgs = target_imgs.to(device)
            target_labels = target_labels.to(device)
            # Reshape to [B, 1, 28, 28]
            x = target_imgs.view(-1, 1, 28, 28)
            logits = clf(x)
            loss = nn.CrossEntropyLoss()(logits, target_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=1)
            correct += (pred == target_labels).sum().item()
            total += target_labels.size(0)
        print(f"  Classifier pretrain epoch {epoch+1}/{epochs} acc: {100*correct/total:.2f}%")


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

        # ----------------------------
        # Hyperparameters
        # ----------------------------
        lr = 0.0002
        epochs = 50
        batch_size = 64

        # MNIST classifier (pretrained, frozen)
        clf = MNISTClassifier().to(device)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining classifier...")
        pretrain_classifier(clf, train_loader, epochs=5, device=device)
        for p in clf.parameters():
            p.requires_grad = False

        # Loss weights
        beta_kl_target = 0.025  # target KL weight; use annealing
        lambda_cls = 1.0
        kl_anneal_epochs = 20  # ramp beta_kl from 0 to target over first N epochs

        recon_criterion = nn.L1Loss()
        cls_criterion = nn.CrossEntropyLoss()

        opt = optim.AdamW(model.parameters(), lr=lr)

        loss_total_l = []
        loss_recon_l = []
        loss_kl_l = []
        loss_cls_l = []
        for epoch in range(1, epochs + 1):
            # KL annealing: ramp from 0 to beta_kl_target over first kl_anneal_epochs
            if epoch <= kl_anneal_epochs:
                beta_kl = beta_kl_target * (epoch / kl_anneal_epochs)
            else:
                beta_kl = beta_kl_target

            loss_total_epoch = 0
            loss_recon_epoch = 0
            loss_kl_epoch = 0
            loss_cls_epoch = 0
            for real_imgs, labels, target_imgs, target_labels, actions, action_obs in train_loader:
                real_imgs = real_imgs.to(device)
                labels = labels.to(device)
                target_imgs = target_imgs.to(device)
                target_labels = target_labels.to(device)
                actions = actions.to(device).float()
                action_obs = action_obs.to(device)  # [B, 2A+1]

                cond = torch.cat([real_imgs, action_obs], dim=1)  # [B, 784 + 2A+1]
                recon, mu, logvar, z, h_encoder, h_decoder = model(target_imgs, cond)

                recon_loss = recon_criterion(recon, target_imgs)
                kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

                # Auxiliary classifier loss on generated/reconstructed images
                recon_img = recon.view(-1, 1, 28, 28)
                logits = clf(recon_img)
                cls_loss = cls_criterion(logits, target_labels)

                total_loss = recon_loss + beta_kl * kl_loss + lambda_cls * cls_loss

                opt.zero_grad()
                total_loss.backward()
                opt.step()

                loss_total_epoch += total_loss.item()
                loss_recon_epoch += recon_loss.item()
                loss_kl_epoch += kl_loss.item()
                loss_cls_epoch += cls_loss.item()

            loss_total_l.append(loss_total_epoch / len(train_loader))
            loss_recon_l.append(loss_recon_epoch / len(train_loader))
            loss_kl_l.append(loss_kl_epoch / len(train_loader))
            loss_cls_l.append(loss_cls_epoch / len(train_loader))

            print(
                f"Epoch {epoch:03d} | loss_total: {loss_total_l[-1]:.4f} | "
                f"recon: {loss_recon_l[-1]:.4f} | kl: {loss_kl_l[-1]:.4f} | cls: {loss_cls_l[-1]:.4f}"
            )

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
                cond = torch.cat([real_imgs, action_obs], dim=1)
                recon, mu, logvar, z, h_encoder, h_decoder = model(target_imgs, cond)

                hidden_encoder_l.append(h_encoder.detach().cpu().numpy())
                hidden_decoder_l.append(h_decoder.detach().cpu().numpy())
                z_l.append(z.detach().cpu().numpy())
                mu_l.append(mu.detach().cpu().numpy())
                target_label_l.append(target_labels.detach().cpu().numpy())
                action_l.append(actions.detach().cpu().numpy())

            # Reconstructions (posterior) for reference
            cond_vis = torch.cat([real_imgs, action_obs], dim=1)
            recon_vis, _, _, _, _, _ = model(target_imgs, cond_vis)
            recons = recon_vis.view(-1, 28, 28)

        print('Finished training, plotting loss, PCA and samples...')

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(loss_total_l, label='loss_total')
        ax.plot(loss_recon_l, label='loss_recon')
        ax.plot(loss_kl_l, label='loss_kl')
        ax.plot(loss_cls_l, label='loss_cls')
        plt.yscale('log')
        ax.legend()
        plt.savefig(f"{base_folder}/loss.png")
        plt.close()

        # Posterior reconstructions (encode target, decode)
        n = 10
        fig, axs_all = plt.subplots(n, 3*2+1, figsize=(2*7, 2*n))
        idx = 0
        for j in range(2):
            for i in range(n):
                axs = axs_all[i, j*3+j:j*3+3+j]
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

        # Prior samples from generate(): K samples per (source, action) - different z, same digit
        K = 5
        n_conds = 8
        with torch.no_grad():
            # Get a batch
            batch = next(iter(train_loader))
            real_imgs, labels, target_imgs, target_labels, actions, action_obs = [x.to(device) for x in batch]
            real_imgs = real_imgs[:n_conds]
            action_obs = action_obs[:n_conds]
            labels = labels[:n_conds]
            target_labels = target_labels[:n_conds]
            actions = actions[:n_conds]
            samples_prior = model.generate(real_imgs, action_obs, n_samples=K)  # [K, n_conds, 784]

        fig, axs_all = plt.subplots(n_conds, 2 + K, figsize=(2*(2+K), 2*n_conds))
        for i in range(n_conds):
            axs_all[i, 0].imshow(real_imgs[i].reshape(28, 28).cpu().numpy(), cmap='gray')
            axs_all[i, 0].set_ylabel(f's:{labels[i].item()} s_next:{target_labels[i].item()} a:{actions[i].item()}')
            axs_all[i, 1].imshow(target_imgs[i].reshape(28, 28).cpu().numpy(), cmap='gray')
            axs_all[i, 1].set_title("target")
            for k in range(K):
                axs_all[i, 2+k].imshow(samples_prior[k, i].cpu().numpy().reshape(28, 28), cmap='gray')
                axs_all[i, 2+k].set_title(f"prior sample {k+1}")
            for j in range(2 + K):
                axs_all[i, j].axis('off')
        fig.suptitle("Prior samples: z~N(0,I), decode(z, cond) - same (source,action), different handwriting")
        fig.tight_layout()
        plt.savefig(f"{base_folder}/samples_prior.png")
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
