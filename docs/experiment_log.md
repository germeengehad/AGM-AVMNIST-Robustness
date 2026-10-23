# Experiment Log — AV-MNIST Robustness Research

## Research Question
How does audio corruption affect Adaptive Gradient Modulation (AGM) on AV-MNIST, and can a gate-based modification improve robustness to noisy audio?

## Experiment 1: Normal Baseline

- Method: Normal
- Dataset: AV-MNIST
- Modality: Multimodal
- Fusion: Late fusion
- Random seed: 999
- Batch size: 64
- Configured epochs: 100
- Learning rate: 0.0001
- Optimizer: SGD
- Alpha: 2.5
- Best recorded test accuracy: 70.04%
- Best epoch: 14
- Final epoch: 99
- Final training accuracy: 89.85%
- Final test accuracy: 66.67%

Output directory:
`checkpoint/2026_10_22 23:50:33_baseline_normal/`

## Experiment 2: AGM Baseline

- Method: Adaptive Gradient Modulation (AGM)
- Dataset: AV-MNIST
- Modality: Multimodal
- Fusion: Late fusion
- Random seed: 999
- Batch size: 64
- Configured epochs: 100
- Learning rate: 0.0001
- Optimizer: SGD
- Alpha: 2.5
- Best recorded test accuracy: 71.792996%
- Best epoch: 27
- Final epoch: 99
- Final test accuracy: 70.97%
- Final audio-only accuracy: 30.57%
- Final visual-only accuracy: 62.08%

Output directory:
`checkpoint/2026_10_23 03:09:23_baseline_agm/`

Best checkpoint:
`ckpt_full_epoch27.pth.tar`

## Published Reference

The original AGM paper reports 72.14% accuracy for late-fusion AGM on AV-MNIST.

Our recorded best AGM accuracy is approximately 0.35 percentage points lower. This comparison is descriptive and does not establish an exact reproduction.

## Evaluation Limitation

The original implementation evaluates the test split after every epoch and tracks the best epoch using test accuracy. Consequently, the best recorded test accuracy is subject to test-set selection bias and must not be presented as an independent final test estimate.

## Next Steps

1. Reproduce the original baseline under the documented environment.
2. Define a fixed audio-corruption protocol.
3. Evaluate Normal and AGM under matched clean and corrupted conditions.
4. Implement and evaluate a gate-based modification.
5. Run an ablation or control experiment.
6. Report task quality and resource usage.
7. Avoid tuning modifications on the test set.
