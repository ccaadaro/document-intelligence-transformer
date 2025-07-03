import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms

class DocumentDataset(Dataset):
    def __init__(self, split='train', vocab_size=10000, seq_len=32, num_classes=16, size=500):
        self.split = split
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Imagen aleatoria RGB (puedes sustituir esto por carga real de documentos)
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        img = self.transform(img)

        # Texto simulado como secuencia de tokens aleatorios
        text = torch.randint(1, self.vocab_size, (self.seq_len,))

        # Etiqueta aleatoria
        label = torch.randint(0, self.num_classes, (1,)).item()

        return img, text, label