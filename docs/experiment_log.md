# Experiment Log — AV-MNIST Robustness Research

## Research Objective

Investigate whether Gaussian noise augmentation applied to the audio modality during AGM training improves robustness to audio corruption on AV-MNIST.

A secondary exploratory objective is to investigate whether a learnable modality gate can improve multimodal feature fusion.

## Shared Experimental Configuration

| Parameter | Setting |
|---|---|
| Dataset | AV-MNIST |
| Modalities | Audio + Image |
| Fusion | Late fusion |
| Random seed | 999 |
| Batch size | 64 |
| Configured epochs | 100 |
| Learning rate | 0.0001 |
| Optimizer | SGD |
| AGM alpha | 2.5 |
| Device | CUDA GPU |

---

## Experiment 1: Original Normal Baseline

- Method: Normal
- Best recorded test accuracy: 70.04%
- Best epoch: 14
- Final epoch: 99
- Final training accuracy: 89.85%
- Final test accuracy: 66.67%
- Checkpoint saving: Disabled in the original run.

Output directory:

`checkpoint/2026_10_22 23:50:33_baseline_normal/`

This original run did not save model checkpoints. A separate Normal training run was therefore performed later to obtain a checkpoint for noise robustness evaluation.

## Experiment 2: AGM Baseline

- Method: Adaptive Gradient Modulation (AGM)
- Best recorded test accuracy: 71.79%
- Best epoch: 27
- Final epoch: 99
- Final test accuracy: 70.97%
- Checkpoint saving: Enabled.

Output directory:

`checkpoint/2026_10_23 03:09:23_baseline_agm/`

Best checkpoint:

`ckpt_full_epoch27.pth.tar`

The original AGM paper reports 72.14% accuracy for late-fusion AGM on AV-MNIST. The recorded result is approximately 0.35 percentage points lower. This is a descriptive reference comparison, not an exact reproduction claim.

## Experiment 3: AGM + Training-Time Audio Noise Augmentation

- Method: AGM
- Experiment name: `agm_noise_aug`
- Augmentation: Gaussian noise
- Training noise sigma: 0.10
- Augmentation modality: Audio only
- Augmentation stage: Training only
- Audio representation: Normalized input tensor, clamped to [0, 1]
- Best recorded test accuracy: 72.15%
- Best epoch: 28
- Final epoch: 99
- Final test accuracy: 71.12%

Output directory:

`checkpoint/2026_10_31 00:49:33_agm_noise_aug/`

The augmentation was applied during training. The evaluation procedure used the same four audio corruption levels as the baseline robustness experiments.

## Experiment 4: Separate Normal Training for Noise Evaluation

The original Normal baseline did not save checkpoints. A separate Normal run was therefore started with checkpoint saving enabled to obtain weights for robustness evaluation.

- Method: Normal
- Experiment name: `normal_noise_eval_checkpointed`
- Random seed: 999
- Configured epochs: 100
- Training status: Interrupted during epoch 77, after batch 200 of 844.
- Best available checkpoint: Epoch 14
- Best recorded test accuracy: 70.04%

Output directory:

`checkpoint/2026_11_07 19:45:31_normal_noise_eval_checkpointed/`

The epoch 14 checkpoint was successfully loaded using strict state-dictionary matching and used for noise robustness evaluation.

**Important:** This is a separate retraining run, not the original Normal baseline checkpoint. The two Normal runs must not be treated as the same training trajectory.

## Experiment 5: Audio Noise Robustness Evaluation

Gaussian noise was added to the audio input during evaluation. The visual input remained unchanged.

Evaluated noise levels:

- sigma = 0.00
- sigma = 0.05
- sigma = 0.10
- sigma = 0.20

### Multimodal Accuracy

| Noise sigma | Normal | AGM | AGM + Noise Augmentation |
|---:|---:|---:|---:|
| 0.00 | 70.04% | 71.79% | 72.15% |
| 0.05 | 68.94% | 71.58% | 72.07% |
| 0.10 | 63.44% | 70.49% | 72.03% |
| 0.20 | 34.39% | 65.46% | 71.25% |

Result files:

- `results/normal_noise_robustness.csv`
- `results/agm_noise_robustness.csv`
- `results/agm_noise_aug_robustness.csv`

At sigma = 0.20, AGM + Noise Augmentation outperformed AGM by approximately 5.80 percentage points.

The Normal result at each noise level comes from the separate retraining run described in Experiment 4.

## Experiment 6: AGM + Learnable Modality Gate

- Experiment name: `agm_gate`
- Method: AGM with a learnable two-way softmax modality gate
- Random seed: 999
- Configured epochs: 100
- Training status: Interrupted during epoch 36 due to a disk-space issue.
- Best available checkpoint: Epoch 21
- Best multimodal test accuracy: 65.05%
- Last completed epoch: 35
- Accuracy at epoch 35: 64.21%

Output directory:

`checkpoint/2026_11_02 18:40:51_agm_gate/`

The gate assigned a mean audio weight of approximately 0.0058 and a visual weight of approximately 0.9942 at the best checkpoint under clean evaluation.

The audio weight remained below 1% across all evaluated noise levels. Gradient-flow and optimizer-update checks confirmed that the gate parameters were connected to training and updated.

The experiment did not improve performance over the AGM baseline. It is retained as an exploratory negative result.

Result files:

- `results/agm_gate_analysis.md`
- `results/agm_gate_epoch21_gate_weights.csv`
- `results/agm_gate_weight_trajectory.csv`

## Experiment 7: AGM + Identity Gate Diagnostic

- Experiment name: `agm_gate_fixed`
- Gate configuration: Fixed equal modality weights (identity gate).
- Training status: Interrupted after 35 epochs.
- Best recorded accuracy: 71.04%
- Best epoch: 19
- Training behavior: Unstable, with a substantial accuracy drop by the final recorded epoch.

Output directory:

`checkpoint/2026_11_03 01:48:53_agm_gate_fixed/`

This experiment is treated as a diagnostic control only. It is not used as evidence of a reliable performance improvement.

---

## Implementation and Checkpoint Validation

- The AGM and AGM + Noise Augmentation checkpoints were loaded with strict state-dictionary matching.
- The Normal noise-evaluation checkpoint was loaded with strict state-dictionary matching using its corresponding model architecture.
- Model construction tests confirmed that the learnable gate is enabled only for the dedicated gate experiments.
- Python syntax compilation succeeded for the modified main training and model files.

These checks support checkpoint compatibility and implementation integrity, but do not replace independent reproduction.

## Methodological Limitations

1. All experiments used one random seed (999).
2. The original implementation evaluates the test split after each epoch and selects the best checkpoint using test accuracy, introducing test-set selection bias.
3. The separate Normal noise-evaluation run was interrupted during epoch 77; its best available checkpoint was from epoch 14.
4. The Normal and AGM implementations use different model classes.
5. The gate experiment was interrupted during epoch 36.
6. The identity-gate diagnostic was also interrupted and showed unstable training.
7. The augmentation experiment tested one training noise strength only.
8. Multiple independent seeds and validation-based checkpoint selection are needed for stronger conclusions.

## Current Project Status

The main controlled comparison is AGM baseline versus AGM + training-time audio noise augmentation.

The Normal robustness evaluation provides an additional reference, with the separate-retraining limitation documented above.

The learnable gate and identity-gate experiments are exploratory diagnostics and are not part of the main proposed method.

The main results and limitations are documented in:

`results/final_comparison.md`
