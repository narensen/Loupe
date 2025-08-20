import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from timm.data import create_transform
from timm.optim import create_optimizer_v2 # NEW: Import the timm optimizer factory
from torch.optim.lr_scheduler import CosineAnnealingLR # NEW: Import the scheduler
from tqdm import tqdm
import os

from model import SwinWithLoupe # Import the model definition

# --- Configuration ---
DATA_DIR = "./data"
MODEL_NAME = "swin_base_patch4_window7_224"
NUM_CLASSES = 200
BATCH_SIZE = 16
IMG_SIZE = 224
LEARNING_RATE = 1e-5
EPOCHS = 50
SPARSITY_LAMBDA = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./output"
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

def train_model():
    print(f"--- Starting ADVANCED Swin-Loupe Training on {DEVICE} ---")

    train_transform = create_transform(input_size=IMG_SIZE, is_training=True, auto_augment='rand-m9-mstd0.5')
    val_transform = create_transform(input_size=IMG_SIZE, is_training=False)

    train_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    model = SwinWithLoupe(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True)
    model.to(DEVICE)

    classification_loss_fn = nn.CrossEntropyLoss()

    optimizer = create_optimizer_v2(model, opt='lion', lr=LEARNING_RATE)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_accuracy = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc="Training")
        for images, labels in pbar_train:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits, attention_map = model(images)
            class_loss = classification_loss_fn(logits, labels)
            sparsity_loss = torch.mean(attention_map)
            total_loss = class_loss + SPARSITY_LAMBDA * sparsity_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            pbar_train.set_postfix({"loss": total_loss.item(), "lr": scheduler.get_last_lr()[0]})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="Validating")
            for images, labels in pbar_val:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, attention_map = model(images)
                total_loss = classification_loss_fn(logits, labels) + SPARSITY_LAMBDA * torch.mean(attention_map)
                val_loss += total_loss.item()

                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_predictions / total_samples

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy*100:.2f}% | LR: {scheduler.get_last_lr()[0]:.1e}")

        scheduler.step()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >> New best accuracy! Model saved to {save_path}")

        if best_val_loss - avg_val_loss > EARLY_STOPPING_MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nValidation loss has not improved for {EARLY_STOPPING_PATIENCE} epochs. Stopping early.")
            break

    print("\n--- Training Complete ---")
    print(f"Best validation accuracy: {best_accuracy*100:.2f}%")

train_model()