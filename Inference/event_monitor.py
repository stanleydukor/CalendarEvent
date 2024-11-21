import os
import json
import torch
from transformers import BertTokenizer

from Inference.utils import *
from Inference.cluster import ConversationClusterer
from Model.model import CalendarEventDetector

class ChatMessage:
    def __init__(self, seqid, ts, user, message):
        self.ts = ts
        self.user = user
        self.seqid = seqid
        self.message = message
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            ts=data['ts'],
            user=data['user'],
            seqid=data['seqid'],
            message=data['message']
        )

class ChatEventMonitor:
    def __init__(self, model_path=None, output_path='./results'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.event_counter = 0
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.clusterer = ConversationClusterer()
        self.calendar_detector = CalendarEventDetector()
        self.calendar_detector.load_state_dict(torch.load(model_path, map_location=self.device))
        self.calendar_detector.to(self.device)
        self.calendar_detector.eval()

    def save_json(self, message):
        self.event_counter += 1
        message_data = {"lines": [{"ts": msg.ts, "user": msg.user, "seqid": msg.seqid, "message": msg.message} for msg in message]}
        filename = os.path.join(self.output_path, f'event_{self.event_counter:04d}.json')
        with open(filename, 'w') as f:
            json.dump(message_data, f, indent=2)
        print(f"Event saved to {filename}")
    
    def process_message(self, message_data):
        if message_data == None:
            is_message_similar = False
            message = None 
        else:
            message = ChatMessage.from_dict(message_data)
            is_message_similar = self.clusterer.is_message_similar(message)
        if not is_message_similar:
            clusters = self.clusterer.get_conversation_clusters()
            if message is not None:
                self.clusterer.initialize_conversation_clusters(message)
            text = ' '.join([format_message(item) for item in clusters])
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                prediction = self.calendar_detector(
                    inputs['input_ids'], 
                    inputs['attention_mask']
                )
            if prediction.item() > 0.95:
                print("Calendar event detected!")
                self.save_json(clusters)
            else:
                print("No calendar event detected.")