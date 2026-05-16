import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Path to memory file
MEMORY_PATH = Path("memory/memory.json")


def load_memory():
    """
    Loads memory from memory/memory.json.
    If the file doesn't exist, returns an empty list.
    """
    if not MEMORY_PATH.exists():
        return []

    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):
    """
    Saves memory back to memory/memory.json.
    """
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4, ensure_ascii=False)


def add_memory(role, content):
    """
    Adds a new memory entry.
    """
    memory = load_memory()

    memory.append({
        "role": role,
        "content": content
    })

    save_memory(memory)


def chat_with_memory(user_input):
    """
    Sends conversation history + current input to OpenAI.
    """

    memory = load_memory()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant with long-term memory."
        }
    ]

    # Add stored memory
    messages.extend(memory)

    # Add latest user message
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.7
    )

    assistant_reply = response.choices[0].message.content

    # Save conversation to memory
    add_memory("user", user_input)
    add_memory("assistant", assistant_reply)

    return assistant_reply


if __name__ == "__main__":
    print("AI Assistant with Memory")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        reply = chat_with_memory(user_input)

        print(f"\nAssistant: {reply}\n")