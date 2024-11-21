from Inference.utils import *
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


class ConversationClusterer:
    def __init__(self, bert_model='bert-base-uncased', 
                 similarity_threshold=0.8,
                 max_cluster_size=20):
        self.similarity_threshold = similarity_threshold
        self.max_cluster_size = max_cluster_size
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert = BertModel.from_pretrained(bert_model)
        self.bert.eval()
        self.messages = []
    
    def get_message_embedding(self, message):
        with torch.no_grad():
            inputs = self.tokenizer(message, 
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=512,
                                  padding=True)
            outputs = self.bert(**inputs)
            return outputs.last_hidden_state[:, 0, :].numpy()
    
    def is_message_similar(self, new_message):
        if not self.messages:
            self.messages.append(new_message)
            return True
        new_format_message = format_message(new_message)
        new_embedding = self.get_message_embedding(new_format_message)
        msg_embedding = self.get_message_embedding(' '.join([format_message(msg) for msg in self.messages[-self.max_cluster_size:]]))
        similarity = cosine_similarity(new_embedding, msg_embedding)[0][0]
        if similarity > self.similarity_threshold:
            self.messages.append(new_message)
            return True
        return False
    
    def get_conversation_clusters(self):
        return self.messages
    
    def initialize_conversation_clusters(self, message):
        print(f"Initializing new conversation cluster with message: {message.message}")
        self.messages = [message]