import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
TRAIN_DIR = r"C:\Users\User\Personal Projects\TongueVision\Final_Segmented_Train_Dataset"
VAL_DIR = r"C:\Users\User\Personal Projects\TongueVision\Final_Segmented_Val_Dataset"

# Hardware & Training Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8          # Keep 8 for stability on RTX 3050
NUM_WORKERS = 2
LEARNING_RATE = 5e-5    # Lowered slightly for AdamW + Transformer fine-tuning
WEIGHT_DECAY = 1e-2     # Increased for AdamW to fight overfitting (prev was 1e-4)
NUM_EPOCHS = 50
IMAGE_SIZE = 224
NUM_CLASSES = 2
WARMUP_EPOCHS = 5       # Gradual warmup to prevent early instability

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
        f_conv_norm = self.ln_conv(f_conv.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f_swin_norm = self.ln_swin(f_swin.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        f_conv_proj = self.proj_conv(f_conv_norm) * self.alpha
        f_swin_proj = self.proj_swin(f_swin_norm) * self.beta
        
        f_cal = torch.cat([f_conv_proj, f_swin_proj], dim=1)
        
        # 3. Dual Attention
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
        print("Initializing TongueVision Model...")
        
        # Branch 1: ConvNeXt Tiny
        base_convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.branch1 = create_feature_extractor(base_convnext, return_nodes={'features': 'out'})
        
        # Branch 2: Swin Transformer Tiny
        base_swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.branch2 = create_feature_extractor(base_swin, return_nodes={'features': 'out'})

        # Fusion
        self.agff = AGFFBlock(in_channels=768)
        
        # Head with Dropout (Added Improvement)
        self.final_ln = nn.LayerNorm(768)
        self.dropout = nn.Dropout(p=0.3)  # <--- Added Dropout here
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        # 1. Extract Features
        f_conv = self.branch1(x)['out']
        f_swin = self.branch2(x)['out']
        
        # 2. Permute Swin
        f_swin = f_swin.permute(0, 3, 1, 2)
        
        # 3. Fusion
        f_fused = self.agff(f_conv, f_swin)
        
        # 4. Classification Head
        f_perm = f_fused.permute(0, 2, 3, 1)
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        
        v = self.dropout(v) # <--- Apply Dropout
        logits = self.classifier(v)
        
        return logits

# ==============================================================================
# 3. UTILITIES (Early Stopping)
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='TongueVision_Stabilized_v2.pth'):
        self.patience = patience  # Increased patience to allow scheduler to work
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
    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: Train path not found: {TRAIN_DIR}")
        exit()
    if not os.path.exists(VAL_DIR):
        print(f"ERROR: Val path not found: {VAL_DIR}")
        exit()

    # Data Transforms (Added ColorJitter for robustness)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # <--- Augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms['val'])

    print(f"Detected Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Initialization ---
    model = TongueVision(num_classes=NUM_CLASSES).to(DEVICE)
    
    # IMPROVEMENT: AdamW with higher weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # IMPROVEMENT: Warmup + Cosine Annealing Scheduler
    # Warmup for first 5 epochs, then Cosine decay for the rest
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[WARMUP_EPOCHS])

    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=7, path='TongueVision_Stabilized_v2.pth')
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')

    # --- Training Loop ---
    print("\nStarting Stabilized Training...")
    
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
            
            # IMPROVEMENT: Gradient Clipping logic
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping!
            scaler.unscale_(optimizer)
            
            # Clip gradients to max norm of 1.0 to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
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

        val_loss = val_loss / len(val_dataset)
        val_acc = correct_val / total_val
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']

        # 3. End of Epoch Stats
        print(f"Epoch {epoch+1} Result: "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # 4. Scheduler Step
        scheduler.step()
        
        # 5. Early Stopping Check
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print("\nTraining Complete. Best model saved as 'TongueVision_Stabilized.pth'.")

