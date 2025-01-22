import torch
import torch.nn as nn
import math

class DiffusionRegressor(nn.Module):
    """
    A simple conditional diffusion network for pose regression:
    1) LiDAR encoder -> context
    2) Time-step embedding
    3) MLP merges: (noised pose, context, time_embed) => predicted noise
    """
    def __init__(
        self,
        lidar_input_size=360,
        lidar_embed_dim=128,
        pose_dim=3,
        time_embed_dim=32,
        hidden_dim=256,
        num_hidden_layers=2,
        activation_fn=nn.ReLU
    ):
        super().__init__()
        # 1) LiDAR encoder (small MLP)
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_input_size, lidar_embed_dim),
            activation_fn(),
            nn.Linear(lidar_embed_dim, lidar_embed_dim),
            activation_fn()
        )
        # 2) Time-step embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            activation_fn(),
            nn.Linear(time_embed_dim, time_embed_dim),
            activation_fn()
        )
        # 3) Denoise MLP
        # input = [noised_pose (3d), lidar_context (lidar_embed_dim), time_embed (time_embed_dim)]
        # total in_dim = 3 + lidar_embed_dim + time_embed_dim
        in_dim = pose_dim + lidar_embed_dim + time_embed_dim
        layers = []
        prev_dim = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, pose_dim))  # predict noise or original pose
        self.denoise_mlp = nn.Sequential(*layers)

    def forward(self, pose_t, lidar_input, t):
        """
        pose_t: shape (batch, 3) - noised pose
        lidar_input: shape (batch, lidar_input_size)
        t: shape (batch,) - time steps
        Return: shape (batch, 3) - predicted noise (or pose)
        """
        # encode LiDAR
        c = self.lidar_encoder(lidar_input)  # (batch, lidar_embed_dim)
        # embed time
        t = t.unsqueeze(-1).float()  # (batch, 1)
        t_embed = self.time_embed(t)  # (batch, time_embed_dim)

        # concat
        x = torch.cat([pose_t, c, t_embed], dim=-1)  # (batch, in_dim)
        return self.denoise_mlp(x)



def forward_diffusion(pose, t, betas):
    """
    pose: shape (batch, 3)
    t: shape (batch,) in [0..T-1]
    betas: a schedule array shape (T,), betas[t_idx] => float
    We'll compute alpha_bar[t_idx] = prod(1 - betas[k]) for k in [0..t_idx].
    """
    # gather alpha_bar[t] for each sample
    alpha_bar_t = betas_to_alpha_bar(betas, t)  # shape (batch,)

    # random noise
    eps = torch.randn_like(pose)
    sqrt_alpha_bar = alpha_bar_t.sqrt().unsqueeze(-1)   # shape (batch,1)
    sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt().unsqueeze(-1)

    pose_t = sqrt_alpha_bar * pose + sqrt_one_minus_alpha_bar * eps
    return pose_t, eps

def betas_to_alpha_bar(betas, t):
    """
    For each index in t, return alpha_bar[t].
    alpha_bar[t] = product_{k=0..t} (1 - betas[k]).
    We'll precompute alpha_bar array of shape (T,).
    Then gather alpha_bar[t[i]].
    """
    # e.g. if alpha_bar is a global array of shape (T,)
    alpha_bar = compute_alpha_bar_from_betas(betas) # shape (T,)
    return alpha_bar[t]

def compute_alpha_bar_from_betas(betas):
    # betas: shape (T,)
    # alpha = 1 - betas
    # alpha_bar[i] = alpha_bar[i-1]*alpha[i]
    # We'll store in a global array for quick usage
    alpha = 1.0 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)
    return alpha_bar
