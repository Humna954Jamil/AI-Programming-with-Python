import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json


parser = argparse.ArgumentParser(description="Predict the class of a flower image")
parser.add_argument('input_image', type=str, help="Path to input image")
parser.add_argument('checkpoint', type=str, help="Path to model checkpoint")
parser.add_argument('--top_k', type=int, default=5, help="Return top K predictions")
parser.add_argument('--category_names', type=str, default=None, help="Path to category to name JSON file")
parser.add_argument('--gpu', action='store_true', help="Use GPU for inference")

args = parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.densenet121(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    image = Image.open(image_path)
    
    # Resize and crop
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    
    # Normalize
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions for PyTorch
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.tensor(np_image, dtype=torch.float32)

def predict(image_path, model, topk, device):
    model.eval()
    model = model.to(device)
    
    # Process the image
    image = process_image(image_path).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        log_ps = model(image)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk, dim=1)
    
    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class.cpu().numpy()[0]]
    top_p = top_p.cpu().numpy()[0]
    
    return top_p, top_class

if __name__ == "__main__":
   
    model = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Predict
    probs, classes = predict(args.input_image, model, args.top_k, device)
    
    # Map classes to category names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]
    
    # Print results
    print("Top K predictions:")
    for prob, class_name in zip(probs, classes):
        print(f"{class_name}: {prob:.3f}")
