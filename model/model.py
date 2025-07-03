import torch
import torch.nn as nn
from torchvision import models

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class DocumentTransformer(nn.Module):
    def __init__(self, 
                 image_feature_dim=512,
                 text_vocab_size=10000, 
                 text_embed_dim=256, 
                 hidden_dim=512, 
                 num_heads=8, 
                 num_layers=6, 
                 num_classes=16):
        super().__init__()

        # Image Encoder (ResNet18 backbone without final classifier)
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])  # output: (B, 512, H/32, W/32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Text Encoder
        self.embedding = nn.Embedding(text_vocab_size, text_embed_dim)
        self.text_pos_enc = PositionalEncoding(text_embed_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=text_embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

        # Fusion Layer
        self.fc_img = nn.Linear(image_feature_dim, hidden_dim)
        self.fc_txt = nn.Linear(text_embed_dim, hidden_dim)
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=2
        )

        # Classification
        self.cls_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, image, text):
        # Image -> feature vector
        img_feat = self.image_encoder(image)         # (B, 512, H', W')
        img_feat = self.pool(img_feat).squeeze(-1).squeeze(-1)  # (B, 512)
        img_feat = self.fc_img(img_feat).unsqueeze(1)           # (B, 1, hidden_dim)

        # Text -> sequence features
        txt_embed = self.embedding(text)             # (B, T, embed_dim)
        txt_embed = self.text_pos_enc(txt_embed)     # (B, T, embed_dim)
        txt_encoded = self.text_encoder(txt_embed.permute(1, 0, 2))  # (T, B, embed_dim)
        txt_encoded = txt_encoded.permute(1, 0, 2)                   # (B, T, embed_dim)
        txt_encoded = self.fc_txt(txt_encoded)                       # (B, T, hidden_dim)

        # Concatenate and fuse
        fused = torch.cat([img_feat, txt_encoded], dim=1)  # (B, 1+T, hidden_dim)
        fused_encoded = self.fusion(fused.permute(1, 0, 2))
        fused_encoded = fused_encoded.permute(1, 0, 2)     # (B, 1+T, hidden_dim)

        # Classification using [CLS] token (the first one)
        cls_token = fused_encoded[:, 0, :]                 # (B, hidden_dim)
        logits = self.cls_head(cls_token)                  # (B, num_classes)
        return logits
