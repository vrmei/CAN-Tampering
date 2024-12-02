import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDIMSampler:
    def __init__(self, model, timesteps=1000, eta=0.0):
        """
        初始化 DDIM 采样器
        :param model: 训练好的扩散模型
        :param timesteps: 扩散过程的总步数
        :param eta: 控制采样的随机性，eta=0 时为确定性采样（DDIM）
        """
        self.model = model
        self.timesteps = timesteps
        self.eta = eta
        self.beta = self.linear_beta_schedule(timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alpha)
        self.posterior_variance = self.beta * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
    
    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)
    
    def sample_ddim(self, x_start, device):
        """
        使用 DDIM 进行采样
        :param x_start: 初始噪声，形状 (batch_size, channels, length)
        :param device: 设备
        :return: 生成的数据
        """
        x = x_start
        for i in reversed(range(self.timesteps)):
            t = torch.full((x.shape[0],), i, dtype=torch.long).to(device)
            predicted_noise = self.model(x)
            alpha = self.alpha[t].view(-1, 1, 1)
            alpha_prev = self.alpha_cumprod_prev[t].view(-1, 1, 1)
            sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)
            
            # 计算 x_0
            x0_pred = (x - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha
            sqrt_alpha_prev = torch.sqrt(self.alpha_cumprod_prev[t]).view(-1, 1, 1)  # (batch_size, 1, 1)   
            # 计算 x_t-1
            if i > 0:
                beta = self.beta[t].view(-1, 1, 1)
                posterior_variance = self.posterior_variance[t].view(-1, 1, 1)
                x = sqrt_alpha_prev * x0_pred + torch.sqrt(1 - self.alpha_cumprod_prev[t].view(-1,1,1)) * predicted_noise + torch.sqrt(posterior_variance) * torch.randn_like(x) * self.eta
            else:
                x = sqrt_alpha_prev * x0_pred + torch.sqrt(1 - self.alpha_cumprod_prev[t].view(-1,1,1)) * predicted_noise
        return x
