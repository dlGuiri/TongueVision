import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# LIME imports
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Point this directly to the image file you want to test
IMAGE_PATH = r"C:\Users\User\Downloads\real_tongue.jpg"
MODEL_PATH = "TongueVision_Double_CNN_v2.pth"  # Ensure this file is in the same directory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set your class names
CLASS_NAMES = ['diabetes', 'healthy'] 

# ==============================================================================
# 2. MODEL ARCHITECTURE (Exact Match)
# ==============================================================================
class AGFFBlock(nn.Module):
    def __init__(self, in_channels=768):
        super(AGFFBlock, self).__init__()
        self.half_channels = in_channels // 2
        
        # Calibration
        self.ln_conv = nn.LayerNorm(in_channels)
        self.ln_swin = nn.LayerNorm(in_channels)
        self.proj_conv = nn.Conv2d(in_channels, self.half_channels, kernel_size=1)
        self.proj_swin = nn.Conv2d(in_channels, self.half_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
        # Spatial Attention (Path A)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel Attention (Path B)
        reduction_dim = max(in_channels // 16, 32)
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, in_channels),
            nn.Sigmoid()
        )

    def forward(self, f_conv, f_swin):
        if f_conv.shape[2:] != f_swin.shape[2:]:
            f_swin = F.interpolate(f_swin, size=f_conv.shape[2:], mode='bilinear', align_corners=False)
            
        f_conv_norm = self.ln_conv(f_conv.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f_swin_norm = self.ln_swin(f_swin.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        f_conv_proj = self.proj_conv(f_conv_norm) * self.alpha
        f_swin_proj = self.proj_swin(f_swin_norm) * self.beta
        
        f_cal = torch.cat([f_conv_proj, f_swin_proj], dim=1)
        
        # Dual Attention
        a_s = self.spatial_gate(f_cal)
        f_spatial = f_cal * a_s
        
        n, c, h, w = f_cal.shape
        z = F.adaptive_avg_pool2d(f_cal, (1, 1)).flatten(1)
        a_c = self.channel_mlp(z).view(n, c, 1, 1)
        f_channel = f_cal * a_c
        
        return f_spatial + f_channel


class TongueVision(nn.Module):
    def __init__(self, num_classes=2):
        super(TongueVision, self).__init__()
        
        # Branch 1: ConvNeXt
        base_convnext1 = models.convnext_tiny(weights=None) 
        self.branch1 = create_feature_extractor(base_convnext1, return_nodes={'features': 'out'})
        
        # Branch 2: ConvNeXt
        base_convnext2 = models.convnext_tiny(weights=None)
        self.branch2 = create_feature_extractor(base_convnext2, return_nodes={'features': 'out'})

        self.agff = AGFFBlock(in_channels=768)
        self.final_ln = nn.LayerNorm(768)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        f_conv1 = self.branch1(x)['out']
        f_conv2 = self.branch2(x)['out']
        
        # Pass both directly to fusion block
        f_fused = self.agff(f_conv1, f_conv2)
        
        f_perm = f_fused.permute(0, 2, 3, 1)
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        v = self.dropout(v)
        logits = self.classifier(v)
        
        return logits


# ==============================================================================
# 3. SINGLE IMAGE INFERENCE LOGIC
# ==============================================================================
def predict_single_image():
    print(f"Device: {DEVICE}")
    
    # 1. Transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load and Prepare Image
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Path not found: {IMAGE_PATH}")
        return

    try:
        image = Image.open(IMAGE_PATH).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    input_tensor = test_transforms(image).unsqueeze(0).to(DEVICE)

    # 3. Load Model
    print("Loading model...")
    model = TongueVision(num_classes=2).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    # 4. Inference
    print(f"\nRunning Inference on: {os.path.basename(IMAGE_PATH)}...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1) 
        
        prob_class_0 = probs[0, 0].item()
        prob_class_1 = probs[0, 1].item()
        
        # Custom threshold logic
        pred_idx = 0 if prob_class_0 > 0.10 else 1
        
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = probs[0, pred_idx].item() * 100
    
    print("\n" + "="*40)
    print("            RESULTS")
    print("="*40)
    print(f"Predicted Class : **{predicted_class.upper()}**")
    print(f"Confidence      : {confidence:.2f}%")
    print("-" * 40)
    print("Raw Probabilities:")
    print(f"  {CLASS_NAMES[0]:<15}: {prob_class_0*100:.2f}%")
    print(f"  {CLASS_NAMES[1]:<15}: {prob_class_1*100:.2f}%")
    print("="*40)

    # ==========================================================================
    # 5. LIME EXPLANATION LOGIC
    # ==========================================================================
    print("\nGenerating LIME explanation (this may take a minute depending on your GPU)...")
    
    # Prepare image for LIME (must be a numpy array of the correct size)
    resized_image = image.resize((224, 224))
    image_array = np.array(resized_image)
    
    # Define the prediction function that LIME will use to test image patches
    def predict_fn(images):
        batch_tensors = []
        for img in images:
            pil_img = Image.fromarray(img.astype(np.uint8))
            batch_tensors.append(test_transforms(pil_img))
        
        batch_tensor = torch.stack(batch_tensors).to(DEVICE)
        
        # Process in batches to prevent Out Of Memory errors
        all_probs = []
        batch_size = 32
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(batch_tensor), batch_size):
                batch = batch_tensor[i:i+batch_size]
                outputs = model(batch)
                batch_probs = F.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(batch_probs)
                
        return np.vstack(all_probs)

    # Initialize LIME Explainer
    explainer = lime_image.LimeImageExplainer(random_state=42)
    
    # Explain the instance
    explanation = explainer.explain_instance(
        image_array,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=300, # You can increase this (e.g., 1000) for a better, but slower, explanation
        random_seed=42
    )
    
    # Extract the explanation mask for the PREDICTED class (using pred_idx from your threshold)
    temp, mask = explanation.get_image_and_mask(
        pred_idx,
        positive_only=False, # Shows both green (supporting) and red (contradicting) regions
        num_features=5,
        hide_rest=False
    )

    

    # Plot the results
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(resized_image)
    plt.title(f'Original Image\n{os.path.basename(IMAGE_PATH)}')
    plt.axis('off')
    
    # Plot 2: LIME Explanation
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title(f'LIME Explanation\n(Explaining: {predicted_class.upper()})')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    save_path = "cnn_lime_result.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"LIME visualization saved successfully as '{save_path}' in your current directory!")

if __name__ == "__main__":
    predict_single_image()