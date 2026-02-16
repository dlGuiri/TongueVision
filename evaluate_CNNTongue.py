import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from tqdm import tqdm
import numpy as np

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
TEST_DATASET_ROOT = r"C:\Users\User\Downloads\Kaggle_Cleaned_Dataset"

# UPDATED: Matches the filename saved in your training script
MODEL_PATH = "DoubleCNN_TongueVision.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 

# ==============================================================================
# 2. MODEL ARCHITECTURE (MATCHING TRAINING SCRIPT EXACTLY)
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
        # 1. Alignment
        if f_conv.shape[2:] != f_swin.shape[2:]:
            f_swin = F.interpolate(f_swin, size=f_conv.shape[2:], mode='bilinear', align_corners=False)
            
        # 2. Calibration
        # Permute for LayerNorm (N, C, H, W) -> (N, H, W, C)
        f_conv_norm = self.ln_conv(f_conv.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f_swin_norm = self.ln_swin(f_swin.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        f_conv_proj = self.proj_conv(f_conv_norm) * self.alpha
        f_swin_proj = self.proj_swin(f_swin_norm) * self.beta
        
        f_cal = torch.cat([f_conv_proj, f_swin_proj], dim=1) # (B, 768, H, W)
        
        # 3. Dual Attention
        # Spatial
        a_s = self.spatial_gate(f_cal)
        f_spatial = f_cal * a_s
        
        # Channel
        n, c, h, w = f_cal.shape
        z = F.adaptive_avg_pool2d(f_cal, (1, 1)).flatten(1)
        a_c = self.channel_mlp(z).view(n, c, 1, 1)
        f_channel = f_cal * a_c
        
        return f_spatial + f_channel

class TongueVision(nn.Module):
    def __init__(self, num_classes=2):
        super(TongueVision, self).__init__()
        
        # --- UPDATED TO MATCH TRAINING: DOUBLE CONVNEXT ---
        
        # Branch 1: ConvNeXt Tiny
        base_cnn1 = models.convnext_tiny(weights=None) # No weights needed, loading from .pth
        self.branch1 = create_feature_extractor(base_cnn1, return_nodes={'features': 'out'})
        
        # Branch 2: ConvNeXt Tiny (Replaced Swin)
        base_cnn2 = models.convnext_tiny(weights=None)
        self.branch2 = create_feature_extractor(base_cnn2, return_nodes={'features': 'out'})

        self.agff = AGFFBlock(in_channels=768)
        self.final_ln = nn.LayerNorm(768)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        # 1. Extract Features
        f_branch1 = self.branch1(x)['out']  
        f_branch2 = self.branch2(x)['out']
        
        # NOTE: Removed the f_swin.permute(...) line here because 
        # ConvNeXt outputs (N, C, H, W) natively, unlike Swin.
        
        # 3. Fusion
        f_fused = self.agff(f_branch1, f_branch2)
        
        # 4. Classification Head
        f_perm = f_fused.permute(0, 2, 3, 1)
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        logits = self.classifier(v)
        
        return logits

# ==============================================================================
# 3. EVALUATION LOGIC
# ==============================================================================
def evaluate_model():
    print(f"Device: {DEVICE}")
    
    # 1. Transforms (Match Training)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Data
    if not os.path.exists(TEST_DATASET_ROOT):
        print(f"Error: Path not found: {TEST_DATASET_ROOT}")
        return

    test_dataset = datasets.ImageFolder(TEST_DATASET_ROOT, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Class Mapping: {test_dataset.class_to_idx}")
    
    # 3. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = TongueVision(num_classes=2).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Make sure you trained it first!")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Tip: Ensure the architecture in this script matches the training script exactly.")
        return

    model.eval()
    
    all_preds = []
    all_labels = []

    print("Running Inference...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            # Probability calculation
            probs = F.softmax(outputs, dim=1) 
            
            # --- PREDICTION LOGIC ---
            # Assuming Class 0 = Diabetes, Class 1 = Non-Diabetes/Healthy
            # We apply a stricter threshold (0.3) for detecting Diabetes
            preds = torch.where(probs[:, 0] > 0.3, 0, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Set Accuracy: {acc*100:.2f}%")
    
    print("\nClassification Report:")
    target_names = list(test_dataset.class_to_idx.keys())
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\nConfusion Matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_model()