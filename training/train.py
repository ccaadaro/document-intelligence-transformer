import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import DocumentTransformer
from dataset import DocumentDataset  # define esto tú o te doy un ejemplo
from sklearn.metrics import classification_report
import os

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 16
lr = 1e-4
num_classes = 16

# Dataset
train_dataset = DocumentDataset(split='train')
val_dataset = DocumentDataset(split='val')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Modelo
model = DocumentTransformer(num_classes=num_classes)
model = model.to(device)

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_val_loss = float('inf')
save_path = "api/model.pt"

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, texts, labels in train_loader:
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"[{epoch+1}/{num_epochs}] Train loss: {avg_loss:.4f}")

    # Validación
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, texts, labels in val_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    print(f"          Val loss: {avg_val_loss:.4f}")
    print(classification_report(all_labels, all_preds))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model to {save_path}")

print("Train Finished!")