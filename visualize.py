import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from timm.data import create_transform
from PIL import Image
import numpy as np
import argparse
import os
import random
import cv2

from model import SwinWithLoupe

def visualize_attention_contour(model, image_path, transform, device, save_path, percentile=95):
    """
    Loads an image, gets the attention map, and draws a contour
    around the most highly activated region.
    """
    img_pil = Image.open(image_path).convert("RGB")
    original_size = img_pil.size
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, attention_map = model(img_tensor)
    
    pred_idx = torch.argmax(logits, dim=1).item()
    
    attention_map = attention_map.squeeze().cpu().numpy()
    
    # Upsample the attention map to the original image size
    attention_map_resized = cv2.resize(attention_map, original_size, interpolation=cv2.INTER_CUBIC)
    
    # --- NEW CONTOUR LOGIC ---
    # 1. Find the threshold for the top N percentile of activations
    threshold = np.percentile(attention_map_resized, percentile)
    
    # 2. Create a binary mask where 1 = above threshold, 0 = below
    mask = (attention_map_resized > threshold).astype(np.uint8)
    
    # 3. Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Convert original PIL image to OpenCV format (RGB)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # 5. Draw the contours on the image
    # We'll draw them in a bright color, like lime green, with a thickness of 2
    cv2.drawContours(img_cv, contours, -1, (0, 255, 0), 2)
    
    # Add the prediction text
    text = f"Prediction: Class {pred_idx}"
    cv2.putText(img_cv, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 6. Save the final image
    cv2.imwrite(save_path, img_cv)
    print(f"Contour visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Swin-Loupe attention with contours.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained model weights (.pth file).")
    parser.add_argument('--num_images', type=int, default=10, help="Number of random images from the test set to visualize.")
    parser.add_argument('--output_dir', type=str, default="./visualizations_contour", help="Directory to save output images.")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = SwinWithLoupe(num_classes=200, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.to(DEVICE)
    
    transform = create_transform(input_size=224, is_training=False)

    test_dataset = ImageFolder(root="./data/test")
    random_indices = random.sample(range(len(test_dataset)), args.num_images)
    
    for i, idx in enumerate(random_indices):
        image_path, _ = test_dataset.imgs[idx]
        save_path = os.path.join(args.output_dir, f"contour_vis_{i+1}_{os.path.basename(image_path)}")
        visualize_attention_contour(model, image_path, transform, DEVICE, save_path)

if __name__ == "__main__":
    main()