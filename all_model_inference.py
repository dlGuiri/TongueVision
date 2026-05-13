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
IMAGE_PATH = r"C:\Users\User\Downloads\test_tongue.jpg"
# "C:\Users\User\Downloads\diabetic_tongue.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['diabetes', 'healthy'] 

# ==============================================================================
# 2. SHARED FUSION BLOCK
# ==============================================================================
class AGFFBlock(nn.Module):
    def __init__(self, in_channels=768):
        super(AGFFBlock, self).__init__()
        self.half_channels = in_channels // 2
        
        self.ln_conv = nn.LayerNorm(in_channels)
        self.ln_swin = nn.LayerNorm(in_channels)
        self.proj_conv = nn.Conv2d(in_channels, self.half_channels, kernel_size=1)
        self.proj_swin = nn.Conv2d(in_channels, self.half_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
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
        
        a_s = self.spatial_gate(f_cal)
        f_spatial = f_cal * a_s
        
        n, c, h, w = f_cal.shape
        z = F.adaptive_avg_pool2d(f_cal, (1, 1)).flatten(1)
        a_c = self.channel_mlp(z).view(n, c, 1, 1)
        f_channel = f_cal * a_c
        
        return f_spatial + f_channel

# ==============================================================================
# 3. ARCHITECTURE DEFINITIONS
# ==============================================================================
class HybridVision(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_convnext = models.convnext_tiny(weights=None) 
        self.branch1 = create_feature_extractor(base_convnext, return_nodes={'features': 'out'})
        base_swin = models.swin_t(weights=None)
        self.branch2 = create_feature_extractor(base_swin, return_nodes={'features': 'out'})
        self.agff = AGFFBlock(in_channels=768)
        self.final_ln = nn.LayerNorm(768)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        f_conv = self.branch1(x)['out']
        f_swin = self.branch2(x)['out'].permute(0, 3, 1, 2) # Swin needs permutation
        f_fused = self.agff(f_conv, f_swin)
        f_perm = f_fused.permute(0, 2, 3, 1)
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        return self.classifier(v)

class DoubleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_convnext1 = models.convnext_tiny(weights=None) 
        self.branch1 = create_feature_extractor(base_convnext1, return_nodes={'features': 'out'})
        base_convnext2 = models.convnext_tiny(weights=None)
        self.branch2 = create_feature_extractor(base_convnext2, return_nodes={'features': 'out'})
        self.agff = AGFFBlock(in_channels=768)
        self.final_ln = nn.LayerNorm(768)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        f_conv1 = self.branch1(x)['out']
        f_conv2 = self.branch2(x)['out'] # No permutation needed for CNN
        f_fused = self.agff(f_conv1, f_conv2)
        f_perm = f_fused.permute(0, 2, 3, 1)
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        return self.classifier(v)

class DoubleTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_swin1 = models.swin_t(weights=None) 
        self.branch1 = create_feature_extractor(base_swin1, return_nodes={'features': 'out'})
        base_swin2 = models.swin_t(weights=None)
        self.branch2 = create_feature_extractor(base_swin2, return_nodes={'features': 'out'})
        self.agff = AGFFBlock(in_channels=768)
        self.final_ln = nn.LayerNorm(768)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        f_swin1 = self.branch1(x)['out'].permute(0, 3, 1, 2) # Swin needs permutation
        f_swin2 = self.branch2(x)['out'].permute(0, 3, 1, 2) # Swin needs permutation
        f_fused = self.agff(f_swin1, f_swin2)
        f_perm = f_fused.permute(0, 2, 3, 1)
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        return self.classifier(v)

# ==============================================================================
# 4. MODELS TO EVALUATE
# ==============================================================================
# Add or remove models here. Make sure the paths are correct.
MODELS_TO_TEST = [
    {"name": "Hybrid_v10", "path": "TongueVision_Diabetes_Final_v10.pth", "model_class": HybridVision},
    {"name": "Double_CNN", "path": "TongueVision_Double_CNN_v2.pth", "model_class": DoubleCNN},
    {"name": "Double_Transformer", "path": "TongueVision_Double_Transformer_v2.pth", "model_class": DoubleTransformer}
]

#MODELS_TO_TEST = [
   # {"name": "Hybrid_v6", "path": "TongueVision_Diabetes_Final_v6.pth", "model_class": HybridVision},
  # {"name": "Double_CNN", "path": "TongueVision_Double_CNN_v1.pth", "model_class": DoubleCNN},
   # {"name": "Double_Transformer", "path": "TongueVision_Double_Transformer.pth", "model_class": DoubleTransformer}
#]
# ==============================================================================
# 5. MULTI-MODEL INFERENCE LOGIC
# ==============================================================================
def predict_and_explain():
    print(f"Device: {DEVICE}")
    
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Path not found: {IMAGE_PATH}")
        return

    try:
        image = Image.open(IMAGE_PATH).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
    resized_image = image.resize((224, 224))
    image_array = np.array(resized_image)
    image_basename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    # Initialize LIME Explainer once
    explainer = lime_image.LimeImageExplainer(random_state=42)

    # Loop through each model configuration
    for config in MODELS_TO_TEST:
        model_name = config["name"]
        model_path = config["path"]
        ModelClass = config["model_class"]
        
        print("\n" + "="*60)
        print(f"EVALUATING MODEL: {model_name}")
        print("="*60)

        if not os.path.exists(model_path):
            print(f"Weights not found at {model_path}. Skipping...")
            continue

        # Instantiate and load model
        model = ModelClass(num_classes=2).to(DEVICE)
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
        except Exception as e:
            print(f"Error loading weights for {model_name}: {e}")
            continue

        # --- Inference ---
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1) 
            
            prob_class_0 = probs[0, 0].item()
            prob_class_1 = probs[0, 1].item()
            
            pred_idx = 0 if prob_class_0 > 0.10 else 1
            
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = probs[0, pred_idx].item() * 100
        
        print(f"Predicted Class : **{predicted_class.upper()}**")
        print(f"Confidence      : {confidence:.2f}%")
        print("Raw Probabilities:")
        print(f"  {CLASS_NAMES[0]:<15}: {prob_class_0*100:.2f}%")
        print(f"  {CLASS_NAMES[1]:<15}: {prob_class_1*100:.2f}%")

        # --- LIME Prediction Function for Current Model ---
        def predict_fn(images):
            batch_tensors = []
            for img in images:
                pil_img = Image.fromarray(img.astype(np.uint8))
                batch_tensors.append(test_transforms(pil_img))
            
            batch_tensor = torch.stack(batch_tensors).to(DEVICE)
            all_probs = []
            batch_size = 32
            
            with torch.no_grad():
                for i in range(0, len(batch_tensor), batch_size):
                    batch = batch_tensor[i:i+batch_size]
                    out = model(batch)
                    batch_probs = F.softmax(out, dim=1).cpu().numpy()
                    all_probs.append(batch_probs)
            return np.vstack(all_probs)

        # --- Generate Explanation ---
        print(f"Generating LIME explanation for {model_name}...")
        explanation = explainer.explain_instance(
            image_array,
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=300, 
            random_seed=42
        )
        
        temp, mask = explanation.get_image_and_mask(
            pred_idx,
            positive_only=False, 
            num_features=5,
            hide_rest=False
        )

        # --- Plot and Save ---
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(resized_image)
        plt.title(f'Original Image\n{os.path.basename(IMAGE_PATH)}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.title(f'LIME Explanation - {model_name}\n(Explaining: {predicted_class.upper()})')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Dynamic File Naming
        save_filename = f"lime_{model_name}_{image_basename}.png"
        plt.savefig(save_filename, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"LIME visualization saved successfully as '{save_filename}'")

if __name__ == "__main__":
    predict_and_explain()