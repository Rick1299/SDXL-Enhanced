import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import Counter

# ==================== CONFIG ====================
train_dir = r"C:\Datasets\Generated datasets\Celeb-A-Split SDXL Enhanced merged\train"
test_dir  = r"C:\Datasets\Generated datasets\Celeb-A-Split SDXL Enhanced merged\test"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-4
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===============================================

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load datasets and dataloaders
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir,  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Compute class weights ===
label_counts = Counter([label for _, label in train_dataset])
total_samples = sum(label_counts.values())
class_weights = [total_samples / label_counts[i] for i in range(len(label_counts))]
weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)

# Model dictionary with pretrained weights
model_dict = {
    'convnext_tiny': models.convnext_tiny(weights="DEFAULT"),
    'convnext_base': models.convnext_base(weights="DEFAULT"),
    'efficientnet_v2_s': models.efficientnet_v2_s(weights="DEFAULT"),
    'efficientnet_v2_m': models.efficientnet_v2_m(weights="DEFAULT"),
    'vit_b_16': models.vit_b_16(weights="DEFAULT"),
    'swin_v2_b': models.swin_v2_b(weights="DEFAULT"),
    'densenet201': models.densenet201(weights="DEFAULT"),
    'vgg19_bn': models.vgg19_bn(weights="DEFAULT"),
    'resnet152': models.resnet152(weights="DEFAULT")
}

# Replace classifier
def modify_classifier(model, name):
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
    elif hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, NUM_CLASSES)
    else:
        raise NotImplementedError(f"No classifier head found for model {name}")
    return model

# Training function
def train_model(model, optimizer, criterion, model_name):
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {running_loss:.4f}")

# Evaluation function
def evaluate_model(model):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))

# === Training loop for all models ===
for name, model in model_dict.items():
    print(f"\n🚀 Training and evaluating model: {name}")
    model = modify_classifier(model, name).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    train_model(model, optimizer, criterion, name)
    evaluate_model(model)
    torch.cuda.empty_cache()
