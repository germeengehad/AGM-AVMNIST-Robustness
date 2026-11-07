# Boosting Multi-modal Model Performance with Adaptive Gradient Modulation
Here is the official Pytorch implementation of AGM proposed in "Boosting Multi-modal Model Performance with Adaptive Gradient Modulation".

**Paper Title: Boosting Multi-modal Model Performance with Adaptive Gradient Modulation**

**Authors: Hong Li<sup> * </sup>, Xingyu Li<sup> * </sup>, Pengbo Hu, Yinuo Lei, Chunxiao Li, Yi Zhou**

**Accepted by: ICCV 2023**

[[arXiv](https://arxiv.org/abs/2308.07686)] [[ICCV Proceedings](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Boosting_Multi-modal_Model_Performance_with_Adaptive_Gradient_Modulation_ICCV_2023_paper.html)]

## Dataset
### 1. AV-MNIST

This dataset can be downloaded from [here](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing).

### 2. CREMA-D

This dataset can be downloaded from [here](https://github.com/CheyneyComputerScience/CREMA-D). Data preprocessing can refer to [here](https://github.com/GeWu-Lab/OGM-GE_CVPR2022/tree/main/data/CREMAD).

### 3. UR-Funny

This raw dataset can be downloaded from [here](https://github.com/ROC-HCI/UR-FUNNY). Also, the processed data can be obtained from [here](https://github.com/ROC-HCI/UR-FUNNY).

### 4. AVE
This dataset can be downloaded from [here](https://sites.google.com/view/audiovisualresearch).

### 5. CMU-MOSEI

This dataset can be downloaded from [here](https://drive.google.com/file/d/1tcVYIMcZdlDzGuJvnMtbMchKIK9ulW1P/view?usp=sharing).

## Training

### Environment config
1. Python: 3.9.13
2. CUDA Version: 11.3
3. Pytorch: 1.12.1
4. Torchvision: 0.13.1
### Train
To train the model using the following command:
```python 
python main.py --data_root '' --device cuda:0 --methods Normal --modality Multimodal --fusion_type late_fusion --random_seed 999 --expt_dir checkpoint --expt_name test --batch_size 64 --EPOCHS 100 --learning_rate 0.0001 --dataset AV-MNIST --alpha 2.5 --SHAPE_contribution False
```

## Citation
```
@inproceedings{li2023boosting,
  title={Boosting Multi-modal Model Performance with Adaptive Gradient Modulation},
  author={Li, Hong and Li, Xingyu and Hu, Pengbo and Lei, Yinuo and Li, Chunxiao and Zhou, Yi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22214--22224},
  year={2023}
}
```

---

# Research Extension: Robust Multimodal Learning under Audio Corruption

This repository also contains an experimental extension based on the AGM
implementation. The project investigates whether training-time Gaussian
noise augmentation can improve the robustness of multimodal learning
when the audio modality is corrupted.

## Research Objective

**Research question:** To what extent can Gaussian noise augmentation
during AGM training improve robustness to audio corruption on AV-MNIST?

The project evaluates three configurations:

1. Normal baseline.
2. Adaptive Gradient Modulation (AGM) baseline.
3. AGM with training-time Gaussian noise augmentation applied to audio.

A learnable modality gate was also explored as a secondary experiment.

## Method

Gaussian noise with a standard deviation of 0.10 was added to the
normalized audio input during training in the augmentation experiment.
The augmented audio tensor was clamped to the range [0, 1].
The visual modality was not modified.

During evaluation, Gaussian noise was applied to the audio modality
at four corruption levels: 0.00, 0.05, 0.10, and 0.20.

## Main Results

Multimodal test accuracy (%):

| Noise sigma | Normal | AGM | AGM + Noise Augmentation |
|---:|---:|---:|---:|
| 0.00 | 70.04 | 71.79 | 72.15 |
| 0.05 | 68.94 | 71.58 | 72.07 |
| 0.10 | 63.44 | 70.49 | 72.03 |
| 0.20 | 34.39 | 65.46 | 71.25 |

At the highest evaluated noise level (sigma = 0.20), AGM with
training-time noise augmentation achieved 71.25% accuracy, compared
with 65.46% for the AGM baseline.

These results suggest that training-time audio noise augmentation
can improve robustness to audio corruption in this experimental
setting.

## Experimental Environment

The experiments were conducted using a different software environment
from the one specified in the original repository.

See [docs/environment.md](docs/environment.md) for hardware and
software details.

## Documentation and Results

- [Final comparison and analysis](results/final_comparison.md)
- [Detailed experiment log](docs/experiment_log.md)
- [Experimental environment](docs/environment.md)
- [Noise robustness results](results/)

## Limitations

- Experiments were conducted using a single random seed (999).
- The Normal robustness evaluation used a separate training run because
  the original Normal baseline did not save model checkpoints. The
  separate run was interrupted before completing all configured epochs.
- Normal and AGM use different model architectures in the original
  implementation.
- The learnable modality gate experiment was exploratory and did not
  improve upon the AGM baseline.
- Further experiments with multiple random seeds and validation-based
  checkpoint selection are needed to establish the generality of the
  findings.

This research extension is an experimental study built upon the
original AGM implementation. It does not claim to be an exact
reproduction of every result in the original paper.
