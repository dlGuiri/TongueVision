import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm  # For progress bars (pip install tqdm)

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
DATASET_ROOT = r"C:\Users\User\Personal Projects\TongueVision\TongueVision_Mendeley_TrainVal"

# Hardware & Training Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8          # Kept low for RTX 3050 (6GB VRAM). Increase to 16 if stable.
NUM_WORKERS = 2         # Number of CPU threads for data loading
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
IMAGE_SIZE = 224
NUM_CLASSES = 2         # Update this if you have more than 2 classes (e.g., Healthy/Unhealthy)

# ==============================================================================
# 2. MODEL ARCHITECTURE (TongueVision)
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
        print("Initializing TongueVision (Dual-Transformer) Model...")
        
        # Branch 1: Swin Transformer Tiny (Replaces ConvNeXt)
        # We use a separate instance so weights update independently
        base_swin1 = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.branch1 = create_feature_extractor(base_swin1, return_nodes={'features': 'out'})
        
        # Branch 2: Swin Transformer Tiny (Same as before)
        base_swin2 = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.branch2 = create_feature_extractor(base_swin2, return_nodes={'features': 'out'})

        # Fusion (AGFF Block)
        # Note: Both Swin Tiny output 768 channels
        self.agff = AGFFBlock(in_channels=768)
        
        # Head
        self.final_ln = nn.LayerNorm(768)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        # 1. Extract Features
        # Swin outputs are usually (N, H, W, C) - Channels Last
        out1 = self.branch1(x)['out'] 
        out2 = self.branch2(x)['out']
        
        # 2. Permute BOTH to match Standard Layout (N, C, H, W)
        # This is necessary because your AGFFBlock expects (N, C, H, W)
        f_swin1 = out1.permute(0, 3, 1, 2) 
        f_swin2 = out2.permute(0, 3, 1, 2)
        
        # 3. Fusion
        # We pass the permuted features to the fusion block
        f_fused = self.agff(f_swin1, f_swin2)
        
        # 4. Classification Head
        f_perm = f_fused.permute(0, 2, 3, 1) # Back to (N, H, W, C) for LayerNorm
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        logits = self.classifier(v)
        
        return logits

# ==============================================================================
# 3. UTILITIES (Early Stopping)
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='DoubleTransformer_TongueVision.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        print(f'Validation loss decreased. Model saved to {self.path}')

# ==============================================================================
# 4. MAIN EXECUTION LOOP
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"Using Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    # --- Data Loading ---
    if not os.path.exists(DATASET_ROOT):
        print(f"ERROR: Dataset path not found: {DATASET_ROOT}")
        print("Please edit the 'DATASET_ROOT' variable in the script.")
        exit()

    # Data Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(os.path.join(DATASET_ROOT, 'train')):
        print(f"ERROR: 'train' folder not found in {DATASET_ROOT}")
        exit()
    if not os.path.exists(os.path.join(DATASET_ROOT, 'val')):
        print(f"ERROR: 'val' folder not found in {DATASET_ROOT}")
        exit()
    
    train_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, 'train'), transform=data_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, 'val'), transform=data_transforms)

    detected_classes = len(train_dataset.classes)
    print(f"Detected Classes: {train_dataset.classes}")

    if detected_classes != NUM_CLASSES:
        print(f"ERROR: Expected {NUM_CLASSES} classes but found {detected_classes}")
        print(f"Check your dataset folder structure at: {DATASET_ROOT}")
        exit()

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Initialization ---
    model = TongueVision(num_classes=NUM_CLASSES).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Base weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=5, path='DoubleTransformer_TongueVision.pth')
    
    # Mixed Precision Scaler for RTX 3050
    scaler = torch.amp.GradScaler('cuda')

    # --- Training Loop ---
    print("\nStarting Training...")
    
    for epoch in range(NUM_EPOCHS):
        # 1. Train
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # AMP Forward Pass
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # AMP Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / train_size
        epoch_acc = correct_train / total_train

        # 2. Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / val_size
        val_acc = correct_val / total_val
        
        # 3. End of Epoch Stats
        print(f"Epoch {epoch+1} Result: "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # 4. Scheduler Step
        scheduler.step()
        
        # 5. Early Stopping Check
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print("\nTraining Complete. Best model saved as 'CNNTongueVision.pth'.")
