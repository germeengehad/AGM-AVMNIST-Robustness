import os
import json
import csv
import torch
import torch.nn.functional as F
from argparse import Namespace

from tasks.AVMNIST_task import AVMNIST_Task
from utils.metric import Accuracy


# ==========================================
# 1. Paths and experiment settings
# ==========================================

EXPERIMENT_DIR = "checkpoint/2026_10_23 03:09:23_baseline_agm"

CONFIG_PATH = os.path.join(EXPERIMENT_DIR, "config.json")

CHECKPOINT_PATH = os.path.join(
    EXPERIMENT_DIR,
    "ckpt_full_epoch27.pth.tar"
)

OUTPUT_PATH = "results/agm_noise_robustness.csv"

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]

SEED = 999


# ==========================================
# 2. Load original configuration
# ==========================================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

cfgs = Namespace(**config)

# Evaluation only: do not train
cfgs.mode = "eval"

device = torch.device(cfgs.device)

print("Device:", device)
print("Loading AGM configuration...")


# ==========================================
# 3. Initialize model and dataloader
# ==========================================

task = AVMNIST_Task(cfgs)

model = task.model.to(device)

test_loader = task.test_dataloader

print("Loading checkpoint...")

try:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )
except TypeError:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

model.load_state_dict(checkpoint["state_dict"])

model.mode = "eval"
model.eval()

print("Checkpoint loaded successfully.")
print("Checkpoint epoch:", checkpoint["epoch"])
print("Original best accuracy:", checkpoint["best_acc"])


# ==========================================
# 4. Evaluation function
# ==========================================

def evaluate_noise(model, test_loader, sigma):

    model.mode = "eval"
    model.eval()

    total_batches = len(test_loader)

    multimodal_acc = 0.0
    audio_acc = 0.0
    visual_acc = 0.0

    with torch.no_grad():

        for audio, image, label in test_loader:

            audio = audio.float().to(device)
            image = image.float().to(device)
            label = label.long().to(device)

            # Add Gaussian noise to audio only
            if sigma > 0:

                noise = torch.randn_like(audio) * sigma

                audio = torch.clamp(
                    audio + noise,
                    min=0.0,
                    max=1.0
                )

            # Forward pass
            out_a, out_v, out = model(audio, image)

            # Calculate accuracy
            batch_mm = Accuracy(
                F.softmax(out, dim=1),
                label
            )

            batch_audio = Accuracy(
                F.softmax(out_a, dim=1),
                label
            )

            batch_visual = Accuracy(
                F.softmax(out_v, dim=1),
                label
            )

            # Preserve the original repository's
            # mean-of-batch-accuracies calculation
            multimodal_acc += batch_mm.item() / total_batches
            audio_acc += batch_audio.item() / total_batches
            visual_acc += batch_visual.item() / total_batches

    return multimodal_acc, audio_acc, visual_acc


# ==========================================
# 5. Run experiments
# ==========================================

results = []

print("\nStarting noise robustness evaluation...\n")

for sigma in NOISE_LEVELS:

    # Reset seed for reproducibility
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"Evaluating sigma = {sigma} ...")

    mm_acc, a_acc, v_acc = evaluate_noise(
        model,
        test_loader,
        sigma
    )

    result = {
        "noise_sigma": sigma,
        "multimodal_accuracy": mm_acc,
        "audio_accuracy": a_acc,
        "visual_accuracy": v_acc
    }

    results.append(result)

    print(
        f"Multimodal: {mm_acc * 100:.2f}% | "
        f"Audio: {a_acc * 100:.2f}% | "
        f"Visual: {v_acc * 100:.2f}%"
    )

    print("-" * 60)


# ==========================================
# 6. Save results to CSV
# ==========================================

os.makedirs("results", exist_ok=True)

with open(OUTPUT_PATH, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "noise_sigma",
            "multimodal_accuracy",
            "audio_accuracy",
            "visual_accuracy"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("\nEvaluation completed!")
print("Results saved to:", OUTPUT_PATH)
