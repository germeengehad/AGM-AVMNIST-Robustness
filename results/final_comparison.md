# AV-MNIST Research Project — Final Experimental Comparison

## 1. Research Objective

This project investigates the robustness of Adaptive Gradient Modulation (AGM) to audio corruption on AV-MNIST.

The main research question is:

**Can training-time Gaussian noise augmentation applied to the audio modality improve the robustness of AGM under audio corruption while maintaining clean multimodal classification performance?**

An additional exploratory experiment investigates whether a learnable modality gate can improve feature fusion. This gate experiment is reported separately because it did not improve performance.

## 2. Dataset and Experimental Configuration

| Parameter | Setting |
|---|---|
| Dataset | AV-MNIST |
| Modalities | Audio + Image |
| Fusion strategy | Late Fusion |
| Batch size | 64 |
| Configured epochs | 100 |
| Learning rate | 0.0001 |
| Optimizer | SGD |
| Random seed | 999 |
| Device | CUDA GPU |
| AGM alpha | 2.5 |
| Training noise augmentation | Gaussian noise, sigma = 0.10 |
| Augmentation modality | Audio only |
| Augmentation stage | Training only |

The proposed augmentation adds Gaussian noise to the normalized audio input during training. The audio tensor is clamped to the range [0, 1]. Images and labels are not modified.

During robustness evaluation, Gaussian noise is added to the audio input at sigma values of 0.00, 0.05, 0.10, and 0.20. The visual input remains unchanged.

## 3. Main Classification Results

| Model | Best recorded test accuracy | Best epoch |
|---|---:|---:|
| Normal | 70.04% | 14 |
| AGM | 71.79% | 27 |
| AGM + Noise Augmentation | 72.15% | 28 |

AGM achieved 1.75 percentage points higher best recorded accuracy than the Normal result. AGM with audio noise augmentation achieved a further 0.36 percentage-point increase over the AGM result.

The original AGM paper reports 72.14% for late-fusion AGM on AV-MNIST. The recorded AGM result in this project is 71.79%, approximately 0.35 percentage points lower. This is a reference comparison, not a claim of exact reproduction.

**Important architectural note:** The Normal implementation uses `Later_Fusion_Resnet_Sum`, whereas the AGM implementation uses `GradMod`. Therefore, the Normal-versus-AGM comparison is a comparison of the implemented methods and architectures, not a perfectly isolated test of the AGM mechanism alone.

## 4. Audio Corruption Robustness

### 4.1 Multimodal Accuracy

| Audio noise sigma | Normal | AGM | AGM + Noise Augmentation |
|---:|---:|---:|---:|
| 0.00 | 70.04% | 71.79% | **72.15%** |
| 0.05 | 68.94% | 71.58% | **72.07%** |
| 0.10 | 63.44% | 70.49% | **72.03%** |
| 0.20 | 34.39% | 65.46% | **71.25%** |

The augmented AGM model achieved the highest multimodal accuracy at every evaluated noise level.

At sigma = 0.20:
- Normal decreased by approximately 35.65 percentage points relative to its clean result.
- AGM decreased by approximately 6.34 percentage points.
- AGM + Noise Augmentation decreased by approximately 0.90 percentage points.
- AGM + Noise Augmentation outperformed AGM by approximately 5.80 percentage points.
- AGM + Noise Augmentation outperformed Normal by approximately 36.86 percentage points.

### 4.2 Audio-Only Accuracy

| Audio noise sigma | Normal | AGM | AGM + Noise Augmentation |
|---:|---:|---:|---:|
| 0.00 | 23.10% | 28.87% | 28.86% |
| 0.05 | 18.48% | 27.67% | 29.19% |
| 0.10 | 11.35% | 22.93% | 29.35% |
| 0.20 | 9.77% | 14.29% | 27.11% |

At sigma = 0.20, the augmented model achieved 27.11% audio-only accuracy, compared with 14.29% for AGM and 9.77% for Normal.

### 4.3 Visual-Only Accuracy

| Audio noise sigma | Normal | AGM | AGM + Noise Augmentation |
|---:|---:|---:|---:|
| 0.00 | 63.88% | 63.71% | 63.73% |
| 0.05 | 63.88% | 63.71% | 63.73% |
| 0.10 | 63.88% | 63.71% | 63.73% |
| 0.20 | 63.88% | 63.71% | 63.73% |

Visual-only accuracy remained constant across audio noise levels because the visual input was not corrupted.

## 5. Exploratory Experiment: AGM + Learnable Modality Gate

A learnable two-way softmax gate was integrated into the AGM late-fusion architecture. It generated sample-dependent audio and visual weights and applied them to the modality representations before concatenation.

Training was interrupted by a disk-space issue during epoch 36. The best available checkpoint was from epoch 21.

| Metric | Result |
|---|---:|
| Best epoch | 21 |
| Best multimodal test accuracy | 65.05% |
| Last completed epoch | 35 |
| Accuracy at last completed epoch | 64.21% |
| AGM baseline best accuracy | 71.79% |

The gate experiment was approximately 6.74 percentage points below the AGM baseline.

### 5.1 Gate Weight Analysis

| Audio noise sigma | Mean audio weight | Mean visual weight |
|---:|---:|---:|
| 0.00 | 0.0058 | 0.9942 |
| 0.05 | 0.0056 | 0.9944 |
| 0.10 | 0.0051 | 0.9949 |
| 0.20 | 0.0038 | 0.9962 |

The gate strongly favored the visual modality, assigning less than 1% of the mean weight to audio at every evaluated noise level. Increasing audio noise did not produce an adaptive redistribution toward the visual modality; instead, the already small audio weight decreased further.

The initially untrained gate produced approximately balanced weights (audio 50.81%, visual 49.19%). The strong visual preference therefore emerged during training.

### 5.2 Feature and Gradient Diagnostics

The measured mean feature norms at the best checkpoint were:

| Measurement | Audio | Visual |
|---|---:|---:|
| Raw feature norm | 17.2686 | 30.5462 |
| Gated feature norm | 0.0636 | 30.3703 |
| Mean gate weight | 0.0058 | 0.9942 |

Gradient-flow checks showed non-zero gradients for the gate parameters, and optimizer updates modified those parameters. The gate was therefore connected to the computational graph and optimizer.

No explicit auxiliary loss or regularization term was used to prevent one modality from dominating the gate. The results are consistent with modality collapse in this particular implementation, but do not establish that all learnable gating methods are ineffective.

An identity-gate control was also explored. That run was interrupted and showed unstable training, so it is treated as a diagnostic experiment rather than a conclusive comparison.

## 6. Main Findings

1. Training-time Gaussian noise augmentation maintained clean multimodal accuracy while improving performance under all evaluated audio corruption levels.
2. The largest observed difference between AGM and AGM + Noise Augmentation occurred at sigma = 0.20, where the augmented model achieved approximately 5.80 percentage points higher multimodal accuracy.
3. Audio-only accuracy under the strongest evaluated corruption was substantially higher for the augmented model.
4. The visual-only results remained nearly unchanged, consistent with the visual modality being uncorrupted.
5. The exploratory learnable gate did not improve performance. It converged toward a strongly visual-dominant solution and substantially suppressed audio features.

Overall, the results support the potential benefit of training-time audio noise augmentation for improving AGM robustness on AV-MNIST under the tested conditions.

## 7. Methodological Limitations

- All experiments used a single random seed (999). Multiple independent runs are required to estimate variability and statistical significance.
- The original implementation evaluates the test split after every epoch and selects the best checkpoint using test accuracy. This introduces test-set selection bias; the best recorded test accuracy should not be interpreted as an independent final test estimate.
- The Normal robustness evaluation used a separately retrained Normal model. That run was interrupted during epoch 77, and the best available checkpoint was from epoch 14. It was not the checkpoint from the original Normal baseline run.
- The AGM and AGM + Noise Augmentation training runs completed their configured 100-epoch schedules, whereas the later Normal retraining did not. The Normal comparison should therefore be interpreted with this training-schedule limitation in mind.
- The Normal and AGM implementations use different model classes, so their comparison does not isolate the effect of AGM alone.
- The robustness evaluation uses one fixed test set and one random seed. The results are comparative observations under this protocol, not independent estimates of generalization.
- Gaussian noise augmentation is applied to the normalized audio representation rather than to the raw audio waveform.
- Only one training augmentation strength (sigma = 0.10) was tested.
- The gate experiment was interrupted during epoch 36 and evaluated using its best available checkpoint at epoch 21.
- The gate experiment used no explicit regularization or auxiliary objective to prevent modality collapse.
- The reported evaluation accuracies follow the repository's batch-wise accuracy aggregation protocol.

## 8. Reproduction Artifacts

### Main robustness results
- Normal: `results/normal_noise_robustness.csv`
- AGM: `results/agm_noise_robustness.csv`
- AGM + Noise Augmentation: `results/agm_noise_aug_robustness.csv`

### Gate analysis
- Gate weights at epoch 21: `results/agm_gate_epoch21_gate_weights.csv`
- Gate weight trajectory: `results/agm_gate_weight_trajectory.csv`
- Gate analysis: `results/agm_gate_analysis.md`

### Baseline summaries
- Normal baseline: `results/baseline_normal_summary.md`
- AGM baseline: `results/baseline_agm_summary.md`

### Experiment directories
- Original Normal baseline: `checkpoint/2026_10_22 23:50:33_baseline_normal/`
- AGM baseline: `checkpoint/2026_10_23 03:09:23_baseline_agm/`
- AGM + Noise Augmentation: `checkpoint/2026_10_31 00:49:33_agm_noise_aug/`
- Normal robustness checkpoint: `checkpoint/2026_11_07 19:45:31_normal_noise_eval_checkpointed/`
- AGM + Learnable Gate: `checkpoint/2026_11_02 18:40:51_agm_gate/`
- AGM + Identity Gate diagnostic: `checkpoint/2026_11_03 01:48:53_agm_gate_fixed/`

Checkpoint directories are local experiment artifacts and are not intended to be committed to the source repository.

## 9. Conclusion

In this AV-MNIST experimental setting, adding Gaussian noise to the normalized audio representation during AGM training improved robustness to audio corruption while maintaining clean multimodal performance. The strongest observed benefit occurred at the highest evaluated noise level.

The learnable gate experiment provided a useful negative result: the implemented gate became strongly visual-dominant and suppressed audio features, reducing classification performance relative to the AGM baseline.

These findings are preliminary. Further work should use multiple random seeds, validation-based checkpoint selection, matched completed training schedules, and additional augmentation strengths before drawing broader conclusions.
