import argparse
import asyncio
import websockets
import json
import pandas as pd
from utils import *
from transformers import BertTokenizer
from Model.model import CalendarEventDetector

data_dict = {
    'text': [],
    'label': []
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(model_path):
    model = CalendarEventDetector()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def predict(model, tokenizer, text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: inputs[key].to(DEVICE) for key in inputs}
    with torch.no_grad():
        output = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    return int(output.item() > 0.5)

async def listen(url):
    model, tokenizer = get_model('Inference/model.pth')
    async with websockets.connect(url) as websocket:
        count = 0
        last_class = 0
        while count < 100:
            message = await websocket.recv()
            data = json.loads(message)
            if len(data['message']) > 30:
                if predict(model, tokenizer, data['message']) == last_class:
                    data_dict['text'].append(data['message'])
                    data_dict['label'].append(last_class)
                    print(f"Received message: {data['message']}, Predicted class: {last_class}")
                    last_class = 1 - last_class
                    count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A WebSocket client to connect to a specified host and port.")
    parser.add_argument("url", type=str, help="A URL corresponding to the WebSocket server (e.g., ws://127.0.0.1:8000)")
    args = parser.parse_args()
    try:
        asyncio.run(listen(args.url))
    except:
        pass
    finally:
        df = pd.DataFrame(data_dict)
        df.to_csv('Data/test.csv', index=False)