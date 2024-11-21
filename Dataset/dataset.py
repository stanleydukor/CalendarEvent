import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer

class ChatDataset(Dataset):
    def __init__(self, path):
        data = pd.read_csv(path)
        self.messages = data['messages'].to_list()
        self.labels = data['labels'].to_list()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, idx):
        message = self.messages[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            message,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'message': message,
            'label': label
        }