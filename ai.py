import os
from openai import OpenAI
from dotenv import load_dotenv
from memory.memory import load_memory, save_memory

load_dotenv()

memory = load_memory()

client = OpenAI(
  base_url="https://api.featherless.ai/v1",
  api_key=os.getenv("FEATHERLESS_API_KEY"),
)

# Add system message once
if not any(msg["role"] == "system" for msg in memory):
  memory.insert(0, {"role": "system", "content": "You are a helpful assistant with memory."})

while True:
  # Add user input
  user_input = input("You: ")
  memory.append({"role": "user", "content": user_input})

  response = client.chat.completions.create(
      model='Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2',
      messages=memory,
      temperature=0.8,
  )

  reply = response.choices[0].message.content
  print("AI:", reply)

  # Save AI response
  memory.append({"role": "assistant", "content": reply})

  save_memory(memory)