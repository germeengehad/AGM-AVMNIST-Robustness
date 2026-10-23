# AV-MNIST — AGM Baseline

## Experiment configuration
- Method: Adaptive Gradient Modulation (AGM)
- Dataset: AV-MNIST
- Modality: Multimodal
- Fusion: Late fusion
- Random seed: 999
- Batch size: 64
- Configured epochs: 100
- Learning rate: 0.0001
- Alpha: 2.5
- Optimizer: SGD
- Device: CUDA GPU
- Checkpoint saving: Enabled

## Our measured results
- Best recorded test accuracy: 71.792996%
- Best epoch: 27
- Final epoch: 99
- Final test accuracy: 70.97%
- Final audio-only accuracy: 30.57%
- Final visual-only accuracy: 62.08%

## Published result
- AGM late-fusion accuracy reported in the original paper on AV-MNIST: 72.14%

The published score and our measured result come from different experiment runs. Their comparison does not establish exact reproduction.

## Comparison with the Normal baseline
- Normal best recorded test accuracy: 70.04%
- AGM best recorded test accuracy: 71.793%
- Difference between the recorded best accuracies: approximately 1.75 percentage points.

This comparison is descriptive. The original implementation selects the best epoch using test accuracy, so these best scores are not independent final test estimates.

## Output files
- Experiment directory: checkpoint/2026_10_23 03:09:23_baseline_agm/
- Configuration: config.json
- Training log: train_logger.log
- TensorBoard logs: tensorboard_out/
- Best checkpoint: ckpt_full_epoch27.pth.tar

## Checkpoint verification
The saved checkpoint was inspected. It contains the model state dictionary, optimizer state, epoch, and recorded best accuracy.

## Evaluation limitation
The original implementation evaluates the test split every epoch and tracks the best epoch based on test accuracy. This introduces test-set selection bias. Future robustness experiments must use a predefined evaluation protocol and must not tune the proposed modification on the test set.

## Reproducibility
The experiment used the configuration saved in the experiment directory. The repository's README and reproducibility documentation should record the exact software environment, dataset access instructions, source revision, and command used to run the experiment.
