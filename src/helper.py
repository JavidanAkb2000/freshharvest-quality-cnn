import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import numpy as np

# ============================================
# Device Configuration
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# INFERENCE TRANSFORM (For Streamlit uploads)
# ============================================
# NO augmentation - just resize and normalize
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================
# Load EfficientNet-B0 Model
# ============================================
def load_efficientnet_model(model_path, num_classes=2):
    """
    Load trained EfficientNet-B0 model

    Args:
        model_path: Path to saved model (.pth file)
        num_classes: Number of output classes
    """
    # Create model architecture (same as training)
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier (same as training)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode

    return model


# ============================================
# Prediction Function
# ============================================
def predict_freshness(image, model):
    """
    Predict if fruit is fresh or spoiled

    Args:
        image: PIL Image object
        model: Trained PyTorch model

    Returns:
        prediction: "Fresh" or "Spoiled"
        confidence: Confidence percentage
    """
    # Apply transform
    img_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Convert to result
    result = "Fresh" if predicted.item() == 0 else "Spoiled"
    confidence_pct = confidence.item() * 100

    return result, confidence_pct


# ============================================
# Denormalize for Display (Optional)
# ============================================
def denormalize_image(tensor):
    """
    Convert normalized tensor back to displayable image
    Useful for showing what the model sees
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Clone tensor to avoid modifying original
    img = tensor.clone().squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)

    return img