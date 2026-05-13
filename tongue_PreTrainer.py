import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# --- Point these to your image folders ---
TRAIN_IMG_DIR = r"C:\Users\User\Personal Projects\shezhen datasets\shezhenv3-coco\shezhenv3-coco\train\images"
VAL_IMG_DIR   = r"C:\Users\User\Personal Projects\shezhen datasets\shezhenv3-coco\shezhenv3-coco\val\images"

# --- Point these to your COCO annotation JSON files ---
TRAIN_ANN = r"C:\Users\User\Personal Projects\shezhen datasets\shezhenv3-coco\shezhenv3-coco\train\annotations\train.json"
VAL_ANN   = r"C:\Users\User\Personal Projects\shezhen datasets\shezhenv3-coco\shezhenv3-coco\val\annotations\val.json"

# Hardware & Training Config
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 8
NUM_WORKERS  = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY  = 1e-2
NUM_EPOCHS    = 50
IMAGE_SIZE    = 224
NUM_CLASSES   = 21   # Step 1: all 21 TCM tongue categories
WARMUP_EPOCHS = 5

# ==============================================================================
# 2. CUSTOM COCO DATASET
# ==============================================================================

class COCOTongueDataset(Dataset):
    """
    Loads images and labels from a COCO-format annotation JSON.

    COCO JSON structure expected:
        {
            "images":      [{"id": int, "file_name": str}, ...],
            "annotations": [{"image_id": int, "category_id": int}, ...],
            "categories":  [{"id": int, "name": str}, ...]
        }

    Each image is assigned the category of its FIRST annotation.
    Multi-label images (multiple annotations) use the dominant category.
    """

    def __init__(self, img_dir, ann_path, transform=None):
        self.img_dir   = img_dir
        self.transform = transform

        # --- Load JSON ---
        with open(ann_path, 'r') as f:
            coco = json.load(f)

        # --- Build category id → 0-indexed label map ---
        # Sorted by category id for consistency
        categories = sorted(coco['categories'], key=lambda c: c['id'])
        self.class_names = [c['name'] for c in categories]
        self.cat_to_label = {c['id']: i for i, c in enumerate(categories)}
        print(f"  Detected {len(self.class_names)} classes: {self.class_names}")

        # --- Map image_id → file_name ---
        id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

        # --- Map image_id → category_id (first annotation wins) ---
        img_to_cat = {}
        for ann in coco['annotations']:
            iid = ann['image_id']
            if iid not in img_to_cat:           # take first annotation per image
                img_to_cat[iid] = ann['category_id']

        # --- Build final sample list [(filepath, label), ...] ---
        self.samples = []
        missing = 0
        for img_id, cat_id in img_to_cat.items():
            fname    = id_to_filename.get(img_id)
            if fname is None:
                continue
            fpath    = os.path.join(img_dir, fname)
            if not os.path.exists(fpath):
                missing += 1
                continue
            label = self.cat_to_label[cat_id]
            self.samples.append((fpath, label))

        if missing:
            print(f"  WARNING: {missing} annotated images not found on disk and were skipped.")
        print(f"  Total usable samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert('RGB')   # always RGB, handles RGBA/grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


# ==============================================================================
# 3. MODEL ARCHITECTURE (TongueVision)
# ==============================================================================

class AGFFBlock(nn.Module):
    def __init__(self, in_channels=768):
        super().__init__()
        self.half_channels = in_channels // 2

        self.ln_conv = nn.LayerNorm(in_channels)
        self.ln_swin = nn.LayerNorm(in_channels)
        self.proj_conv = nn.Conv2d(in_channels, self.half_channels, kernel_size=1)
        self.proj_swin = nn.Conv2d(in_channels, self.half_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.ones(1))

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

        f_conv_norm = self.ln_conv(f_conv.permute(0,2,3,1)).permute(0,3,1,2)
        f_swin_norm = self.ln_swin(f_swin.permute(0,2,3,1)).permute(0,3,1,2)

        f_conv_proj = self.proj_conv(f_conv_norm) * self.alpha
        f_swin_proj = self.proj_swin(f_swin_norm) * self.beta
        f_cal = torch.cat([f_conv_proj, f_swin_proj], dim=1)

        a_s = self.spatial_gate(f_cal)
        f_spatial = f_cal * a_s

        n, c, h, w = f_cal.shape
        z   = F.adaptive_avg_pool2d(f_cal, (1,1)).flatten(1)
        a_c = self.channel_mlp(z).view(n, c, 1, 1)
        f_channel = f_cal * a_c

        return f_spatial + f_channel


class TongueVision(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        print("Initializing TongueVision Model...")

        base_convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.branch1  = create_feature_extractor(base_convnext, return_nodes={'features': 'out'})

        base_swin    = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.branch2 = create_feature_extractor(base_swin, return_nodes={'features': 'out'})

        self.agff       = AGFFBlock(in_channels=768)
        self.final_ln   = nn.LayerNorm(768)
        self.dropout    = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(768, num_classes)   # 21 output nodes for Step 1

    def forward(self, x):
        f_conv = self.branch1(x)['out']
        f_swin = self.branch2(x)['out']
        f_swin = f_swin.permute(0, 3, 1, 2)

        f_fused = self.agff(f_conv, f_swin)

        f_perm  = f_fused.permute(0, 2, 3, 1)
        f_norm  = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v       = F.adaptive_avg_pool2d(f_norm, (1,1)).flatten(1)
        v       = self.dropout(v)
        return self.classifier(v)


# ==============================================================================
# 4. UTILITIES
# ==============================================================================

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='tongue_expert_model.pth'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False
        self.path       = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self._save(val_loss, model)
            self.counter = 0

    def _save(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        print(f'Val loss improved. Model saved → {self.path}')


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print(f"Device : {DEVICE}")
    print(f"Goal   : Step 1 — Train Tongue Expert (21 TCM classes)")

    # --- Validate paths ---
    for p in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_ANN, VAL_ANN]:
        if not os.path.exists(p):
            print(f"ERROR: Path not found → {p}")
            exit()

    # --- Transforms ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # --- Datasets ---
    print("\nLoading Train Dataset:")
    train_dataset = COCOTongueDataset(TRAIN_IMG_DIR, TRAIN_ANN, transform=data_transforms['train'])
    print("\nLoading Val Dataset:")
    val_dataset   = COCOTongueDataset(VAL_IMG_DIR,   VAL_ANN,   transform=data_transforms['val'])

    # --- Verify NUM_CLASSES matches annotation ---
    detected = len(train_dataset.class_names)
    if detected != NUM_CLASSES:
        print(f"\nWARNING: NUM_CLASSES={NUM_CLASSES} but annotation has {detected} categories.")
        print(f"Auto-adjusting NUM_CLASSES to {detected}.")
        NUM_CLASSES = detected

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # --- Model ---
    model     = TongueVision(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
                             milestones=[WARMUP_EPOCHS])

    criterion     = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=7, path='tongue_expert_model.pth')
    scaler        = torch.amp.GradScaler('cuda')

    # --- Training Loop ---
    print("\n" + "="*60)
    print("Starting Step 1: Tongue Expert Training (21 classes)")
    print("="*60)

    for epoch in range(NUM_EPOCHS):

        # -- Train --
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss  += loss.item() * inputs.size(0)
            _, predicted   = torch.max(outputs, 1)
            total_train   += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc  = correct_train / total_train

        # -- Validate --
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val   += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_dataset)
        val_acc  = correct_val / total_val
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:>2} | "
              f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        scheduler.step()
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print("\nStep 1 Complete. Tongue Expert saved as 'tongue_expert_model.pth'.")