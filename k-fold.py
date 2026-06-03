import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.model_selection import StratifiedKFold
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
# Point this to your held-out TEST folder only (diabetes / non_diabetes subfolders)
# These are images the model has NEVER seen during training or validation.
TEST_DATASET_ROOT = r"C:\Users\User\Personal Projects\Final Combined Dataset\test"
MODEL_PATH = "TongueVision_Diabetes_Final_v10.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
K_FOLDS = 5       # How many folds to split the test set into
RANDOM_SEED = 42

# ==============================================================================
# 2. MODEL ARCHITECTURE (Must match Training Script exactly)
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

        base_convnext = models.convnext_tiny(weights=None)
        self.branch1 = create_feature_extractor(base_convnext, return_nodes={'features': 'out'})

        base_swin = models.swin_t(weights=None)
        self.branch2 = create_feature_extractor(base_swin, return_nodes={'features': 'out'})

        self.agff = AGFFBlock(in_channels=768)
        self.final_ln = nn.LayerNorm(768)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        f_conv = self.branch1(x)['out']
        f_swin = self.branch2(x)['out']
        f_swin = f_swin.permute(0, 3, 1, 2)

        f_fused = self.agff(f_conv, f_swin)

        f_perm = f_fused.permute(0, 2, 3, 1)
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        logits = self.classifier(v)

        return logits


# ==============================================================================
# 3. HELPERS
# ==============================================================================
def load_model(model_path, device):
    model = TongueVision(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_inference(model, loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.where(probs[:, 0] > 0.30, 0, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


# ==============================================================================
# 4. MAIN: K-FOLD ON TEST SET ONLY
# ==============================================================================
def kfold_evaluate():
    print(f"Device: {DEVICE}")
    print(f"Evaluating stability using {K_FOLDS}-Fold split of the TEST SET only.")
    print(f"(Model weights are fixed — no retraining occurs.)\n{'='*60}")

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(TEST_DATASET_ROOT):
        print(f"Error: Path not found: {TEST_DATASET_ROOT}")
        return

    test_dataset = datasets.ImageFolder(TEST_DATASET_ROOT, transform=eval_transforms)
    target_names = list(test_dataset.class_to_idx.keys())
    all_labels_array = np.array([label for _, label in test_dataset.samples])

    print(f"Class Mapping : {test_dataset.class_to_idx}")
    print(f"Total Samples : {len(test_dataset)}")
    print(f"Class Counts  : { {name: int((all_labels_array == idx).sum()) for name, idx in test_dataset.class_to_idx.items()} }\n")

    # Load the model once — weights are frozen, same for every fold
    print("Loading model weights...")
    try:
        model = load_model(MODEL_PATH, DEVICE)
        print("Weights loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Stratified split preserves class balance in each fold
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_accuracies = []
    agg_cm = np.zeros((len(target_names), len(target_names)), dtype=int)
    all_true_agg, all_pred_agg = [], []

    indices = np.arange(len(test_dataset))

    for fold, (_, val_idx) in enumerate(skf.split(indices, all_labels_array)):
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold + 1} / {K_FOLDS}  |  Samples in this fold: {len(val_idx)}")
        print(f"{'─'*60}")

        val_sampler = SubsetRandomSampler(val_idx)
        val_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            sampler=val_sampler,
            num_workers=2
        )

        y_true, y_pred = run_inference(model, val_loader, DEVICE)

        acc = accuracy_score(y_true, y_pred)
        fold_accuracies.append(acc)
        cm = confusion_matrix(y_true, y_pred)
        agg_cm += cm
        all_true_agg.extend(y_true)
        all_pred_agg.extend(y_pred)

        print(f"\n  Accuracy: {acc * 100:.2f}%")
        print(f"\n  Classification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names))

        # Per-fold confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix – Fold {fold + 1}')
        plt.tight_layout()
        fname = f'confusion_matrix_fold{fold + 1}.png'
        plt.savefig(fname)
        plt.close()
        print(f"  Saved: {fname}")

    # ==============================================================================
    # 5. AGGREGATE RESULTS
    # ==============================================================================
    print(f"\n{'='*60}")
    print("  CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print(f"\n  Per-Fold Accuracies:")
    for i, a in enumerate(fold_accuracies):
        print(f"    Fold {i+1}: {a*100:.2f}%")

    print(f"\n  Mean Accuracy : {mean_acc*100:.2f}%")
    print(f"  Std Dev       : {std_acc*100:.2f}%")
    print(f"\n  Aggregate Classification Report (all folds combined):")
    print(classification_report(all_true_agg, all_pred_agg, target_names=target_names))

    # Aggregate confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(agg_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Aggregate Confusion Matrix ({K_FOLDS}-Fold, Test Set)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_aggregate.png')
    plt.close()
    print("\n  Saved: confusion_matrix_aggregate.png")

    # Accuracy bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([f"Fold {i+1}" for i in range(K_FOLDS)],
                  [a * 100 for a in fold_accuracies],
                  color='steelblue', edgecolor='black')
    ax.axhline(mean_acc * 100, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {mean_acc*100:.2f}%')
    ax.fill_between(range(K_FOLDS),
                    (mean_acc - std_acc) * 100,
                    (mean_acc + std_acc) * 100,
                    alpha=0.15, color='red', label=f'±1 Std = {std_acc*100:.2f}%')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{K_FOLDS}-Fold Test Set Stability — Accuracy per Fold')
    ax.set_ylim(0, 105)
    ax.legend()
    for bar, a in zip(bars, fold_accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{a*100:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('kfold_accuracy_chart.png')
    plt.close()
    print("  Saved: kfold_accuracy_chart.png")

    # ==============================================================================
    # 6. MISCLASSIFIED IMAGES (full test set pass)
    # ==============================================================================
    print(f"\n{'─'*60}")
    print("  MISCLASSIFIED IMAGES (full test set)")
    print(f"{'─'*60}")

    full_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    y_true_full, y_pred_full = run_inference(model, full_loader, DEVICE)

    image_paths = [s[0] for s in test_dataset.samples]
    wrong_count = 0
    for i in range(len(y_pred_full)):
        if y_pred_full[i] != y_true_full[i]:
            file_name = os.path.basename(image_paths[i])
            true_class = target_names[y_true_full[i]]
            pred_class = target_names[y_pred_full[i]]
            print(f"  File: {file_name} | True: {true_class} | Predicted: {pred_class}")
            wrong_count += 1

    print(f"\n  Total misclassified: {wrong_count} / {len(y_pred_full)}")


if __name__ == "__main__":
    kfold_evaluate()