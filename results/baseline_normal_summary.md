# AV-MNIST — Normal Baseline

## Experiment configuration
- Method: Normal
- Dataset: AV-MNIST
- Modality: Multimodal
- Fusion: Late fusion
- Random seed: 999
- Batch size: 64
- Configured epochs: 100
- Learning rate: 0.0001
- Alpha: 2.5
- Device: CUDA GPU

## Recorded results
- Best recorded test accuracy: 70.04%
- Best epoch: 14
- Final epoch: 99
- Final training accuracy: 89.85%
- Final test accuracy: 66.67%

## Important evaluation note
The original implementation evaluates the test split every epoch and tracks the best epoch using test accuracy. Therefore, the best test accuracy is not an independent final test estimate.

## Output files
- Experiment directory: checkpoint/2026_10_22 23:50:33_baseline_normal/
- Configuration: config.json
- Training log: train_logger.log
- TensorBoard logs: tensorboard_out/

## Checkpoint status
No model checkpoint was listed in the Normal experiment directory during the initial file inspection. The model weights have not yet been confirmed as saved.

## Reproducibility
These recorded metrics came from the completed Normal experiment. Verify the original log and config before finalizing the report.
