import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model import UNetWithTransformer
from sampler import DDIMSampler

# 假设您的数据已经准备好，形状为 (num_samples, seq_len, input_dim)
# 这里以随机数据为例
num_samples = 1000
seq_len = 81
input_dim = 1
output_dim = 1
batch_size = 32
num_epochs = 100
timesteps = 1000
eta = 0.0  # 确定性采样

# 生成随机数据作为示例
train_data_np = torch.randn(num_samples, seq_len, input_dim)
train_label_np = torch.randn(num_samples, output_dim)  # 对于生成任务，标签可以是数据本身

# 创建 Dataset 和 DataLoader
train_dataset = TensorDataset(train_data_np, train_label_np)
traindataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 实例化模型
model = UNetWithTransformer(
    input_dim=input_dim,
    output_dim=output_dim,
    base_channels=64,
    transformer_embed_dim=256,
    num_transformer_heads=8,
    num_transformer_layers=4,
    ff_hidden_dim=512,
    dropout=0.1
).to('cuda' if torch.cuda.is_available() else 'cpu')

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in tqdm(traindataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(traindataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 实例化 DDIM 采样器
ddim_sampler = DDIMSampler(model, timesteps=timesteps, eta=eta)

# 生成新数据
model.eval()
with torch.no_grad():
    # 初始化噪声
    x_start = torch.randn(batch_size, seq_len, input_dim).to(device)
    generated_data = ddim_sampler.sample_ddim(x_start, device)
    print("Generated Data Shape:", generated_data.shape)
