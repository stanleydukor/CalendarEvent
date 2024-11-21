import argparse
import asyncio
import websockets
import json

from Inference.event_monitor import ChatEventMonitor

monitor = ChatEventMonitor(model_path='Inference/model.pth', output_path='./results')

sample_message_data = [
    {"ts": 1618329600, "user": "Alice", "seqid": 1, "message": "Hello!"},
    {"ts": 1618329601, "user": "Bob", "seqid": 2, "message": "Hi!"},
    {"seqid": 45624, "ts": 1704100456.5682841, "user": "hstefan", "message": "hey e_t_, want to jump on a call?"},
    {"seqid": 45625, "ts": 1704100456.5682841, "user": "e_t_", "message": "Sure, let's do it! 3pm works for me."},
    {"seqid": 45626, "ts": 1704100456.5682841, "user": "hstefan", "message": "Great! I'll send you a calendar invite."},
    {"seqid": 45627, "ts": 1704100456.5682841, "user": "e_t_", "message": "Sounds good!"},
    {"seqid": 45628, "ts": 1704100456.5682841, "user": "hstefan", "message": "See you then!"},
    {"ts": 1618329601, "user": "Bob", "seqid": 3, "message": "How are you?"},
    {"ts": 1618329600, "user": "Alice", "seqid": 4, "message": "I'm good, thanks!"},
    {"ts": 1618329601, "user": "Bob", "seqid": 5, "message": "Great!"},
    {"ts": 1618329600, "user": "Alice", "seqid": 6, "message": "What about you?"}
]

async def listen(url):
    async with websockets.connect(url) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(data)
            monitor.process_message(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A WebSocket client to connect to a specified host and port.")
    parser.add_argument("url", type=str, help="A URL corresponding to the WebSocket server (e.g., ws://127.0.0.1:8000)")
    args = parser.parse_args()
    try:
        asyncio.run(listen(args.url))
    except:
        pass
    finally:
        monitor.process_message(None)