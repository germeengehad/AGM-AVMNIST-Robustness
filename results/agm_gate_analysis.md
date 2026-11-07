# AGM + Learnable Modality Gate: Experimental Analysis

## 1. Experiment Overview

**Experiment name:** `agm_gate`

**Objective:**

This experiment investigates whether integrating a learnable modality gate into the Adaptive Gradient Modulation (AGM) framework can improve multimodal feature fusion by dynamically adjusting the contributions of audio and visual modalities.

The experiment uses the AV-MNIST dataset, which contains audio and image modalities for digit classification.

The gate is designed to generate two sample-dependent weights using a softmax function:

- Audio weight
- Visual weight

These weights are applied to the corresponding modality feature representations before concatenation and classification.

**Training configuration:**

- Dataset: AV-MNIST
- Fusion strategy: Late fusion
- Training method: AGM
- Random seed: 999
- Batch size: 64
- Number of epochs: 100
- Learning rate: 0.0001
- AGM alpha: 2.5
- Noise augmentation: Disabled
- Checkpoint saving: Enabled

The experiment was interrupted by a disk-space issue during epoch 36. Therefore, the best available checkpoint from epoch 21 was used for evaluation.

---

## 2. Classification Performance

The best checkpoint was obtained at epoch 21.

| Metric | Result |
|---|---:|
| Best epoch | 21 |
| Best multimodal test accuracy | 65.05% |
| Final completed epoch | 35 |
| Final completed epoch accuracy | 64.21% |

### Comparison with the AGM Baseline

| Experiment | Best Test Accuracy |
|---|---:|
| AGM Baseline | 71.79% |
| AGM + Learnable Gate | 65.05% |

The learnable gate experiment achieved an accuracy approximately 6.74 percentage points lower than the AGM baseline.

This indicates that introducing the current gating mechanism did not improve classification performance.

---

## 3. Gate Weight Analysis

The gate generates a pair of softmax-normalized weights for each sample.

The mean weights were evaluated under different audio noise levels using the best checkpoint (epoch 21).

| Audio Noise Sigma | Mean Audio Weight | Mean Visual Weight |
|---|---:|---:|
| 0.00 | 0.0058 | 0.9942 |
| 0.05 | 0.0056 | 0.9944 |
| 0.10 | 0.0051 | 0.9949 |
| 0.20 | 0.0038 | 0.9962 |

### Observations

1. The gate strongly favors the visual modality, even when the audio input is clean.
2. The mean audio weight remains below 1% across all evaluated noise levels.
3. Increasing audio noise does not produce a meaningful adaptive redistribution of modality weights.
4. The gate therefore fails to demonstrate the intended noise-responsive fusion behavior.

The results suggest that the learned gate converged toward a visual-dominant solution rather than learning balanced, noise-sensitive modality fusion.

---

## 4. Gate Weight Trajectory During Training

To investigate when the modality imbalance emerged, gate weights were evaluated across available checkpoints.

| Epoch | Mean Audio Weight | Mean Visual Weight |
|---:|---:|---:|
| 1 | 0.0114 | 0.9886 |
| 2 | 0.0058 | 0.9942 |
| 5 | 0.0037 | 0.9963 |
| 8 | 0.0040 | 0.9960 |
| 9 | 0.0041 | 0.9959 |
| 12 | 0.0047 | 0.9953 |
| 13 | 0.0046 | 0.9954 |
| 18 | 0.0049 | 0.9951 |
| 21 | 0.0058 | 0.9942 |

### Observation

The untrained gate initially produced approximately balanced weights:

- Audio: 50.81%
- Visual: 49.19%

However, the gate became strongly visual-dominant during early training.

This indicates that the imbalance was learned during optimization rather than being caused by an initially biased gate configuration.

---

## 5. Feature Representation Analysis

The mean feature norms were measured before and after applying the gate.

| Measurement | Audio | Visual |
|---|---:|---:|
| Raw feature norm | 17.2686 | 30.5462 |
| Gated feature norm | 0.0636 | 30.3703 |
| Mean gate weight | 0.0058 | 0.9942 |

The gated audio-to-visual feature norm ratio was approximately:

**0.0021 (0.21%)**

This confirms that the audio representation was substantially suppressed before reaching the classification head.

Although the visual features had a larger raw norm, the difference between the original feature norms alone does not explain the extreme suppression of audio features.

---

## 6. Implementation and Gradient Flow Verification

Additional tests were conducted to verify whether the gate was correctly integrated into the computational graph.

The following checks were completed:

- The gate was included in the AGM model architecture.
- All gate parameters were included in the optimizer.
- Backpropagation produced non-zero gradients for all gate parameter tensors.
- Optimizer updates successfully modified the gate parameters.
- The model produced finite classification outputs with the expected dimensions.

Therefore, the observed collapse was not caused by a completely disconnected gate or missing optimizer updates.

### Training Objective

The main classification loss was calculated using the fused multimodal output.

No explicit auxiliary objective or regularization term was introduced to prevent one modality from dominating the gate.

This is an important characteristic of the current implementation and a possible factor contributing to modality collapse.

---

## 7. Interpretation and Discussion

The experiment revealed a modality collapse phenomenon in the implemented learnable gating architecture.

Although the gate was initially close to balanced, training caused it to assign almost all feature weight to the visual modality.

Consequently, the audio representation contributed very little to the final multimodal representation.

This behavior was accompanied by a reduction in classification accuracy compared with the AGM baseline.

The results demonstrate that simply introducing a learnable modality gate does not guarantee effective multimodal fusion or robustness to modality corruption.

A gate may instead converge toward a solution that relies excessively on the stronger modality while suppressing the other modality.

However, the current experiment does not establish that all gated fusion approaches are ineffective. The observation is specific to the implemented architecture, training objective, and experimental configuration.

---

## 8. Limitations

The following limitations should be considered:

1. The experiment was conducted using a single random seed (999).
2. Training was interrupted during epoch 36 because of insufficient disk space.
3. Evaluation was performed using the best available checkpoint at epoch 21.
4. The current gate uses a two-way softmax, forcing the modality weights to sum to one.
5. No explicit regularization or auxiliary objective was used to prevent modality collapse.
6. The interaction between AGM's gradient modulation mechanism and the learnable gate requires further investigation.

These limitations should be considered when interpreting the results.

---

## 9. Conclusion and Future Investigation

The current AGM + Learnable Gate experiment did not improve multimodal classification performance.

Instead, it revealed a strong preference for the visual modality and severe suppression of audio features.

This finding motivates further investigation into methods that can prevent modality collapse while preserving the ability to adapt modality contributions under noisy conditions.

Potential directions for future experiments include:

- Investigating fixed equal modality weights as a diagnostic control.
- Introducing mechanisms to prevent excessive modality suppression.
- Exploring auxiliary modality-specific supervision.
- Investigating the interaction between AGM and adaptive feature fusion.

The existing experiment is retained as a documented negative result and will serve as a reference for evaluating improved gating strategies.
