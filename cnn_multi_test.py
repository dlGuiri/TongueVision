import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from tqdm import tqdm
import numpy as np

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
TEST_DATASET_ROOT = r"C:\Users\User\Personal Projects\Final Combined Dataset\test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 

# List all the models you want to evaluate here
MODEL_PATHS = [
    "TongueVision_Double_CNN_v2.pth",
]

#"TongueVision_Diabetes_Final.pth",
# "TongueVision_Diabetes_Final_v2.pth",
# "TongueVision_Diabetes_Final_v4.pth",
# "TongueVision_Diabetes_Final_v5.pth"

# The thresholds you want to test (from 0.10 to 0.95 in steps of 0.05)
THRESHOLDS = np.arange(0.1, 0.96, 0.05)

# ==============================================================================
# 2. MODEL ARCHITECTURE (Unchanged)
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

class TongueVision(nn.Module):
    def __init__(self, num_classes=2):
        super(TongueVision, self).__init__()
        
        # Branch 1: ConvNeXt Tiny
        base_convnext1 = models.convnext_tiny(weights=None) 
        self.branch1 = create_feature_extractor(base_convnext1, return_nodes={'features': 'out'})
        
        # Branch 2: ConvNeXt Tiny (Replacing Swin)
        base_convnext2 = models.convnext_tiny(weights=None)
        self.branch2 = create_feature_extractor(base_convnext2, return_nodes={'features': 'out'})

        self.agff = AGFFBlock(in_channels=768)
        self.final_ln = nn.LayerNorm(768)
        
        # Note: Added dropout to match your training script! 
        # (It automatically turns off during model.eval(), but is needed for load_state_dict)
        self.dropout = nn.Dropout(p=0.3) 
        
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        f_conv1 = self.branch1(x)['out']
        f_conv2 = self.branch2(x)['out']
        
        # NO permute needed for f_conv2 since ConvNeXt naturally outputs (B, C, H, W)
        
        f_fused = self.agff(f_conv1, f_conv2)
        f_perm = f_fused.permute(0, 2, 3, 1)
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        v = self.dropout(v)
        logits = self.classifier(v)
        
        return logits

# ==============================================================================
# 3. AUTOMATED EVALUATION LOGIC
# ==============================================================================
def evaluate_models():
    print(f"Device: {DEVICE}")
    
    # 1. Prepare Data ONCE for all models
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(TEST_DATASET_ROOT):
        print(f"Error: Path not found: {TEST_DATASET_ROOT}")
        return

    test_dataset = datasets.ImageFolder(TEST_DATASET_ROOT, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    target_names = list(test_dataset.class_to_idx.keys())
    print(f"Class Mapping: {test_dataset.class_to_idx}\n")

    # 2. Iterate through each model
    for model_path in MODEL_PATHS:
        print(f"{'='*50}")
        print(f"EVALUATING MODEL: {model_path}")
        print(f"{'='*50}")

        if not os.path.exists(model_path):
            print(f"File not found, skipping: {model_path}")
            continue

        model = TongueVision(num_classes=2).to(DEVICE)
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            continue
        
        all_probs_class_0 = []
        all_labels = []

        # Run inference once per model to get raw probabilities
        print("Running Inference...")
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, leave=False):
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1) 
                
                # Store the probability of Class 0 (Diabetes) and the true labels
                all_probs_class_0.extend(probs[:, 0].cpu().numpy())
                all_labels.extend(labels.numpy())

        all_probs_class_0 = np.array(all_probs_class_0)
        all_labels = np.array(all_labels)

        # 3. Test multiple thresholds on the raw probabilities
        best_f1 = -1
        best_thresh = 0.5
        best_preds = None

        # --- NEW VARIABLES FOR RECALL TRACKING ---
        best_diabetes_recall = -1
        thresh_for_best_recall = 0.5
        acc_for_best_recall = 0.0

        print("\nTesting Thresholds...")
        for t in THRESHOLDS:
            # If prob of Class 0 > t, pick 0, else pick 1
            preds = np.where(all_probs_class_0 > t, 0, 1)
            
            # --- EXISTING F1 LOGIC ---
            current_f1 = f1_score(all_labels, preds, average='macro')
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_thresh = t
                best_preds = preds

            # --- NEW RECALL LOGIC ---
            # Calculate Recall for Diabetes (Assumed Class 0 based on your comments)
            current_recall = recall_score(all_labels, preds, pos_label=0, zero_division=0)
            current_acc = accuracy_score(all_labels, preds)

            # Update if we find a strictly better recall
            if current_recall > best_diabetes_recall:
                best_diabetes_recall = current_recall
                thresh_for_best_recall = t
                acc_for_best_recall = current_acc
            # TIE-BREAKER: If recall is tied (e.g., both are 1.0), pick the one with better accuracy
            elif current_recall == best_diabetes_recall and current_acc > acc_for_best_recall:
                thresh_for_best_recall = t
                acc_for_best_recall = current_acc

        # 4. Report Best Results for this model
        best_acc = accuracy_score(all_labels, best_preds)
        print(f"\n>>> Best Threshold (Optimized for Macro F1): {best_thresh:.2f}")
        print(f">>> Best Macro F1-Score: {best_f1:.4f}")
        print(f">>> Accuracy at F1 Threshold: {best_acc*100:.2f}%")
        
        # --- NEW RECALL REPORTING ---
        print(f"\n>>> Threshold for HIGHEST Diabetes Recall: {thresh_for_best_recall:.2f}")
        print(f">>> Highest Diabetes Recall: {best_diabetes_recall*100:.2f}%")
        print(f">>> Accuracy at Highest Recall Threshold: {acc_for_best_recall*100:.2f}%")
        
        print("\nClassification Report (at best F1 threshold):")
        print(classification_report(all_labels, best_preds, target_names=target_names))

        # 5. Save Confusion Matrix for this specific model
        cm = confusion_matrix(all_labels, best_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        clean_name = os.path.splitext(os.path.basename(model_path))[0]
        plt.title(f'Confusion Matrix\n{clean_name} (Threshold: {best_thresh:.2f})')
        
        cm_filename = f'cm_{clean_name}.png'
        plt.savefig(cm_filename)
        plt.close() # Close plot to avoid overlap in the next loop
        print(f"Confusion Matrix saved as '{cm_filename}'\n")

if __name__ == "__main__":
    evaluate_models()