import yaml
import torch
from models.resnet import ResNet50
from data.dataset import ImageNetDataset
from trainer.trainer import Trainer
from utils.logger import Logger

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create logger
    logger = Logger(config)

    # Create model
    model = ResNet50(num_classes=config['model']['num_classes'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Create data loaders
    dataset = ImageNetDataset(config)
    train_loader, val_loader = dataset.get_loaders()

    # Create trainer and start training
    trainer = Trainer(model, config, logger)
    trainer.train(train_loader, val_loader)

    logger.close()

if __name__ == '__main__':
    main() 