# src/analyze_confusion.py

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from timm.data import create_transform
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import os

from model import SwinWithLoupe # Our V1 model

def analyze_model_confusion(weights_path, data_dir):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running analysis on device: {DEVICE}")

    # --- Load Model ---
    model = SwinWithLoupe(num_classes=200, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # --- Load Data ---
    transform = create_transform(input_size=224, is_training=False)
    test_dataset = ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Invert the class_to_idx to map index back to class name
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    class_names = [idx_to_class[i].split('.')[-1].replace('_', ' ') for i in range(len(idx_to_class))]

    # --- Get Predictions ---
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on test set"):
            images = images.to(DEVICE)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # --- Analyze Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    
    # Set diagonal to zero to ignore correct predictions
    np.fill_diagonal(cm, 0)
    
    # Find the top N confused pairs
    num_confused_pairs = 10
    indices = np.argsort(cm.flatten())[-num_confused_pairs*2::2] # Get top N pairs
    
    print(f"\n--- Top {num_confused_pairs} Most Confused Pairs of Species ---")
    for index in reversed(indices):
        true_idx, pred_idx = np.unravel_index(index, cm.shape)
        num_errors = cm[true_idx, pred_idx]
        if num_errors > 0:
            print(f"- Ground Truth: '{class_names[true_idx]}' was misclassified as '{class_names[pred_idx]}' {num_errors} times.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze model confusion matrix.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained model weights (.pth file).")
    parser.add_argument('--data_dir', type=str, default="./data", help="Path to the ORGANIZED data folder.")
    args = parser.parse_args()
    from tqdm import tqdm
    analyze_model_confusion(args.weights, args.data_dir)