# ResNet50 ImageNet Training

This repository contains a modular PyTorch implementation for training ResNet50 from scratch on ImageNet. The implementation is designed to achieve 70% top-1 accuracy and includes support for both full ImageNet training and training on a subset of data.

## Project Structure 
```
resnet50_training/
├── config/
│   └── config.yaml         # Training configuration
├── models/
│   ├── __init__.py
│   └── resnet.py          # ResNet50 model implementation
├── data/
│   └── dataset.py         # Data loading and augmentation
├── utils/
│   ├── __init__.py
│   ├── logger.py          # Logging utilities
│   └── metrics.py         # Training metrics
├── trainer/
│   ├── __init__.py
│   └── trainer.py         # Training loop implementation
├── main.py                # Training entry point
└── requirements.txt       # Project dependencies
```

## Requirements

- Python 3.7+
- CUDA-capable GPU (recommended)
- ImageNet dataset

## Installation

1. Clone the repository: 
```
bash
git clone https://github.com/yourusername/resnet50-training.git
cd resnet50-training
```
2. Install dependencies:
```
bash
pip install -r requirements.txt
```
3. Configure the dataset path:
Edit `config/config.yaml` and update the following paths:
```
yaml
data:
train_path: "/path/to/imagenet/train"
val_path: "/path/to/imagenet/val"
```

## Configuration

The training configuration is defined in `config/config.yaml`. Key parameters include:

- `batch_size`: Training batch size (default: 256)
- `num_epochs`: Number of training epochs (default: 90)
- `learning_rate`: Initial learning rate (default: 0.1)
- `use_subset`: Whether to use a subset of ImageNet (default: true)
- `subset_size`: Number of training samples if using subset (default: 100,000)

## Training

1. For training on a subset of ImageNet (recommended for initial testing): 
```
python main.py
```

2. For full ImageNet training, modify `config.yaml`:
```yaml
data:
  use_subset: false
```
Then run:
```bash
python main.py
```

## Features

- Modular and clean implementation
- Multi-GPU training support
- TensorBoard logging
- Checkpoint saving
- Standard ImageNet augmentations
- Learning rate scheduling
- Progress monitoring

## Training Details

- **Architecture**: ResNet50 (from scratch, no pre-training)
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate Schedule**: MultiStepLR with milestones at epochs [30, 60, 80]
- **Data Augmentation**: Random resized crop, horizontal flip, color jitter
- **Weight Initialization**: Kaiming normal initialization

## Monitoring Training

1. Launch TensorBoard:
```bash
tensorboard --logdir experiments
```

2. Open your browser and navigate to `http://localhost:6006`

## Expected Results
When training on the full ImageNet dataset:
- Target Top-1 Accuracy: 70%
- Training Time: ~3-4 days on 4 V100 GPUs
- Final Model Size: ~98MB

## Checkpoints
Checkpoints are saved in the `experiments/experiment_N` directory:
- `checkpoint_epochX.pth`: Regular checkpoints
- `model_best.pth`: Best performing model
