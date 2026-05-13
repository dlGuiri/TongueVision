import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

# --- Scikit-Learn Imports for the ML Pivot ---
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
TRAIN_DIR = r"C:\Users\User\Personal Projects\Final Combined Dataset\train"
VAL_DIR = r"C:\Users\User\Personal Projects\Final Combined Dataset\val"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_WORKERS = 2
IMAGE_SIZE = 224

# ==============================================================================
# 2. MODEL ARCHITECTURE (Frozen Feature Extractor)
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
        
        base_convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.branch1 = create_feature_extractor(base_convnext, return_nodes={'features': 'out'})
        
        base_swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.branch2 = create_feature_extractor(base_swin, return_nodes={'features': 'out'})

        self.agff = AGFFBlock(in_channels=768)
        self.final_ln = nn.LayerNorm(768)
        
        # We don't need these anymore for the ML Pivot, but we leave them to load weights safely
        self.dropout = nn.Dropout(p=0.5)  
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        f_conv = self.branch1(x)['out']
        f_swin = self.branch2(x)['out']
        f_swin = f_swin.permute(0, 3, 1, 2)
        f_fused = self.agff(f_conv, f_swin)
        f_perm = f_fused.permute(0, 2, 3, 1)
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        
        # This 'v' is the golden 768-dimensional feature vector we want for the SVM
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        
        # We apply the identity function to bypass the dropout and classifier
        v = self.dropout(v) 
        logits = self.classifier(v)
        
        return v  # <--- CRITICAL CHANGE: Returning 'v' directly instead of logits

# ==============================================================================
# 3. FEATURE EXTRACTION FUNCTION
# ==============================================================================
def extract_features(model, dataloader, device):
    """Passes images through the model and returns pure numpy arrays."""
    features_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            # Get the 768-dim vectors
            features = model(inputs)
            
            # Move from GPU back to CPU and convert to standard numpy arrays
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
    # Stack the batches together into one giant matrix
    return np.vstack(features_list), np.concatenate(labels_list)

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print(f"Using Device: {DEVICE}")
    
    # --- Add your Test Directory ---
    TEST_DIR = r"C:\Users\User\Personal Projects\Final Combined Dataset\test"
    
    # --- Data Loading (No Augmentation needed for pure extraction) ---
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform) # <--- Loaded Test Set
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # <--- Test Loader

    # --- Load Pre-trained Model ---
    print("\nLoading Pre-trained Tongue Expert...")
    model = TongueVision(num_classes=2)
    model.load_state_dict(torch.load('TongueVision_Diabetes_Final_v6.pth', map_location=DEVICE))
    
    # Bypass the final layers so model(x) returns the raw features
    model.classifier = nn.Identity()
    model.dropout = nn.Identity()
    model = model.to(DEVICE)
    
    # --- 1. Extract Features ---
    print("\nStarting Phase 1: Feature Extraction...")
    X_train, y_train = extract_features(model, train_loader, DEVICE)
    X_val, y_val = extract_features(model, val_loader, DEVICE)
    X_test, y_test = extract_features(model, test_loader, DEVICE) # <--- Extract Test Features
    
    print(f"Extracted Training Matrix Shape: {X_train.shape}")
    print(f"Extracted Validation Matrix Shape: {X_val.shape}")
    print(f"Extracted Test Matrix Shape: {X_test.shape}")

    # --- 2. Standardize Features ---
    # SVMs work best when all numbers are scaled to a similar range
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test) # <--- Scale Test Features using the Train scaler

    # ==============================================================================
    # 5. OPTIMIZED SVM TRAINING (Grid Search)
    # ==============================================================================
    print("\nStarting Phase 2: Optimizing SVM via Grid Search...")

    # Define the "Grid" of parameters to test
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }

    # Initialize the Grid Search with 5-fold cross-validation
    # We optimize for 'f1_weighted' to balance precision and recall
    grid = GridSearchCV(
        SVC(random_state=42), 
        param_grid, 
        refit=True, 
        verbose=1, 
        cv=5, 
        scoring='f1_weighted'
    )

    grid.fit(X_train_scaled, y_train)

    print(f"\nBest Parameters Found: {grid.best_params_}")
    best_svm = grid.best_estimator_

    # ==============================================================================
    # 6. FINAL EVALUATION ON TEST SET
    # ==============================================================================
    print("\nStarting Phase 3: Final Evaluation with Optimized Model...")

    y_test_pred = best_svm.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\n=========================================")
    print(f"OPTIMIZED TEST ACCURACY: {test_accuracy * 100:.2f}%")
    print(f"=========================================\n")

    print("Detailed Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=test_dataset.classes))

    print("\nTest Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print(conf_matrix)

    # Bonus: Save the final SVM model for your demo/app
    import joblib
    joblib.dump(best_svm, 'TonFusion_SVM_Final.pkl')
    joblib.dump(scaler, 'TonFusion_Scaler.pkl')
    print("\nFinal Model and Scaler saved for deployment!")