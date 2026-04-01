import json
import os

MEMORY_FILE = "memory\memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(messages):
    with open(MEMORY_FILE, "w") as f:
        json.dump(messages, f, indent=2)