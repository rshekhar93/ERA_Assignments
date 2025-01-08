import logging
import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, config):
        self.config = config
        self.exp_dir = os.path.join(config['logging']['save_dir'], 
                                   f"experiment_{self._get_experiment_id()}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.exp_dir)
        self._setup_logging()

    def _get_experiment_id(self):
        existing_experiments = os.listdir(self.config['logging']['save_dir'])
        return len(existing_experiments)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.exp_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )

    def log_metrics(self, metrics, step, phase='train'):
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{metric_name}', value, step)
            logging.info(f'{phase.capitalize()} - Step {step}: {metric_name}: {value:.4f}')

    def close(self):
        self.writer.close() 