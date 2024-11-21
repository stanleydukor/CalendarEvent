import torch
from trainer import TrainingModule
from utils import *

config = load_config('config.yaml')
MODEL_DIR = 'checkpoints'
MODEL = get_last_checkpoint(MODEL_DIR, return_best=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Model: {MODEL}')
module = TrainingModule(config).load_from_checkpoint(MODEL)
pytorch_model = module.model
torch.save(pytorch_model.state_dict(), 'Inference/model.pth')

