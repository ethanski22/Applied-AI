import os
from openai import OpenAI
from dotenv import load_dotenv
from memory import (
  load_memory,
  save_memory,
  add_to_memory,
  retrieve_memory
)

load_dotenv()

memory = load_memory()

client = OpenAI(
  base_url="https://api.featherless.ai/v1",
  api_key=os.getenv("FEATHERLESS_API_KEY"),
)

# Ensure system message exists
if not any(msg["role"] == "system" for msg in memory):
  memory.insert(0, {
    "role": "system",
    "content": "You are a helpful assistant with long-term memory."
  })


msg_counter = len(memory)

while True:
  user_input = input("You: ")

  # Retrieve relevant past memories
  relevant_memories = retrieve_memory(user_input, n_results=5)

  # Build context
  messages = [memory[0]]  # system message
  messages.extend(relevant_memories)
  messages.append({"role": "user", "content": user_input})

  response = client.chat.completions.create(
    model='Qwen3-Embedding-4B-IC',
    messages=messages,
    temperature=0.8,
  )

  reply = response.choices[0].message.content
  print("AI:", reply)

  # Save to JSON memory
  memory.append({"role": "user", "content": user_input})
  memory.append({"role": "assistant", "content": reply})
  save_memory(memory)

  # Save to vector DB
  add_to_memory("user", user_input, msg_counter)
  msg_counter += 1

  add_to_memory("assistant", reply, msg_counter)
  msg_counter += 1