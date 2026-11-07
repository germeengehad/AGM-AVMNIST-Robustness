# AGM Noise Robustness Evaluation

## 1. Objective

This experiment evaluates the robustness of the reproduced Adaptive Gradient Modulation (AGM) model against Gaussian noise added to the audio modality in the AV-MNIST dataset.

The objective is to investigate how audio degradation affects multimodal classification performance and whether the visual modality can compensate for the degraded audio input.

## 2. Model and Configuration

| Parameter | Value |
|---|---|
| Dataset | AV-MNIST |
| Method | AGM |
| Fusion strategy | Late Fusion |
| Modalities | Audio + Visual |
| Best checkpoint | Epoch 27 |
| Best recorded accuracy | 71.79% |
| Random seed | 999 |
| Batch size | 64 |
| Learning rate | 0.0001 |
| Optimizer | SGD |
| Configured epochs | 100 |

Checkpoint:

`checkpoint/2026_10_23 03:09:23_baseline_agm/ckpt_full_epoch27.pth.tar`

The previously trained AGM checkpoint was loaded. No additional training was performed.

## 3. Experimental Setup

A standalone script, `evaluate_noise.py`, was implemented to evaluate the saved model under different audio noise conditions.

Gaussian noise was added to the normalized audio spectrogram:

`audio_noisy = clamp(audio + noise, 0, 1)`

Where:

`noise ~ N(0, σ²)`

Only the audio modality was modified. Visual inputs and labels remained unchanged.

The original dataset files were not modified.

### Noise Levels

| Condition | Noise standard deviation (σ) |
|---|---:|
| Clean | 0.00 |
| Low | 0.05 |
| Medium | 0.10 |
| High | 0.20 |

A fixed random seed (999) was used for reproducibility.

**Note:** Noise was added directly to normalized spectrogram representations, not to raw audio waveforms. These values should not be interpreted as SNR measurements.

## 4. Results

| Noise Level (σ) | Multimodal Accuracy | Audio Accuracy | Visual Accuracy |
|---|---:|---:|---:|
| 0.00 | 71.79% | 28.87% | 63.71% |
| 0.05 | 71.58% | 27.67% | 63.71% |
| 0.10 | 70.49% | 22.93% | 63.71% |
| 0.20 | 65.46% | 14.29% | 63.71% |

### Key Observations

1. Audio accuracy decreases as Gaussian noise intensity increases, dropping from 28.87% to 14.29%.
2. Visual accuracy remains constant at 63.71%, as the visual inputs are unchanged.
3. Multimodal accuracy decreases from 71.79% to 65.46% at σ = 0.20.
4. The visual modality does not fully compensate for the degradation introduced into the audio modality.

The absolute decrease in multimodal accuracy between clean audio and σ = 0.20 is 6.33 percentage points.

## 5. Initial Conclusion

The results indicate that the reproduced AGM model is sensitive to Gaussian noise in the audio spectrogram.

Although the visual branch maintains stable performance, multimodal classification accuracy still decreases under audio degradation.

These findings motivate the following research question:

**Can audio noise augmentation during training improve AGM's robustness to degraded audio inputs?**

This will be investigated in the next experimental stage.

## 6. Limitations

- Noise was applied to spectrogram representations rather than raw audio waveforms.
- The evaluation used the first 90% of the test arrays, following the original repository's data loader.
- The original training implementation selected the best checkpoint based on test accuracy, introducing test-set model-selection bias.
- The baseline was trained with one random seed.
- The current experiment evaluates Gaussian noise robustness only.

## 7. Files

| File | Description |
|---|---|
| `evaluate_noise.py` | Noise robustness evaluation script |
| `results/agm_noise_robustness.csv` | Numerical evaluation results |
| `results/agm_noise_robustness_summary.md` | Experiment documentation |
| `checkpoint/2026_10_23 03:09:23_baseline_agm/config.json` | Original configuration |
| `checkpoint/2026_10_23 03:09:23_baseline_agm/ckpt_full_epoch27.pth.tar` | AGM checkpoint |

## 8. Project Progress

| Experiment | Status |
|---|---|
| Normal baseline reproduction | Completed |
| AGM baseline reproduction | Completed |
| Baseline comparison | Completed |
| Gaussian audio noise evaluation | Completed |
| Audio noise augmentation during training | Pending |
| Improved model evaluation | Pending |
| Final comparison | Pending |
