import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x


class UNetWithTransformer(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, base_channels=64, transformer_embed_dim=256, num_transformer_heads=8, num_transformer_layers=4, ff_hidden_dim=512, dropout=0.1):
        super(UNetWithTransformer, self).__init__()
        # 编码器
        self.encoder1 = self.conv_block(input_dim, base_channels)
        self.encoder2 = self.conv_block(base_channels, base_channels * 2)
        self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool2d(2)
        
        # Transformer
        self.transformer_input = nn.Conv1d(base_channels * 8, transformer_embed_dim, kernel_size=1)
        self.transformer = nn.TransformerEncoder(
            TransformerBlock(transformer_embed_dim, num_transformer_heads, ff_hidden_dim, dropout),
            num_layers=num_transformer_layers
        )
        self.transformer_output = nn.Conv1d(transformer_embed_dim, base_channels * 8, kernel_size=1)
        
        # 解码器
        self.upconv4 = self.up_conv(base_channels * 8, base_channels * 4)
        self.decoder4 = self.conv_block(base_channels * 8, base_channels * 4)
        self.upconv3 = self.up_conv(base_channels * 4, base_channels * 2)
        self.decoder3 = self.conv_block(base_channels * 4, base_channels * 2)
        self.upconv2 = self.up_conv(base_channels * 2, base_channels)
        self.decoder2 = self.conv_block(base_channels * 2, base_channels)
        self.upconv1 = self.up_conv(base_channels, base_channels // 2)
        self.decoder1 = self.conv_block(base_channels, base_channels // 2)
        
        # 输出层
        self.conv_final = nn.Conv1d(base_channels // 2, output_dim, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)  # (batch, base_channels, L)
        enc2 = self.encoder2(self.pool(enc1))  # (batch, base_channels*2, L/2)
        enc3 = self.encoder3(self.pool(enc2))  # (batch, base_channels*4, L/4)
        enc4 = self.encoder4(self.pool(enc3))  # (batch, base_channels*8, L/8)
        
        # Transformer
        batch, channels, length = enc4.size()
        x_t = self.transformer_input(enc4)  # (batch, transformer_embed_dim, L/8)
        x_t = x_t.permute(2, 0, 1)  # (seq_len, batch, embed_dim)
        x_t = self.transformer(x_t)  # (seq_len, batch, embed_dim)
        x_t = x_t.permute(1, 2, 0)  # (batch, embed_dim, seq_len)
        x_t = self.transformer_output(x_t)  # (batch, base_channels*8, L/8)
        
        # 解码器
        dec4 = self.upconv4(x_t)  # (batch, base_channels*4, L/4)
        dec4 = torch.cat((dec4, enc3), dim=1)  # (batch, base_channels*8, L/4)
        dec4 = self.decoder4(dec4)  # (batch, base_channels*4, L/4)
        
        dec3 = self.upconv3(dec4)  # (batch, base_channels*2, L/2)
        dec3 = torch.cat((dec3, enc2), dim=1)  # (batch, base_channels*4, L/2)
        dec3 = self.decoder3(dec3)  # (batch, base_channels*2, L/2)
        
        dec2 = self.upconv2(dec3)  # (batch, base_channels, L)
        dec2 = torch.cat((dec2, enc1), dim=1)  # (batch, base_channels*2, L)
        dec2 = self.decoder2(dec2)  # (batch, base_channels, L)
        
        dec1 = self.upconv1(dec2)  # (batch, base_channels//2, 2L)
        dec1 = self.decoder1(dec1)  # (batch, base_channels//2, 2L)
        
        # 输出
        out = self.conv_final(dec1)  # (batch, output_dim, 2L)
        return out

