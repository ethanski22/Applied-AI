import os
import json
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

MEMORY_FILE = "memory/memory.json"

load_dotenv()

# Initialize OpenAI client (for embeddings)
client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=os.getenv("FEATHERLESS_API_KEY"),
)

# Initialize Chroma
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="memory")


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memory(messages):
    with open(MEMORY_FILE, "w") as f:
        json.dump(messages, f, indent=2)


# 🔑 Create embedding
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ➕ Add memory to vector DB
def add_to_memory(role, content, msg_id):
    collection.add(
        documents=[content],
        embeddings=[embed(content)],
        metadatas=[{"role": role}],
        ids=[str(msg_id)]
    )


# 🔍 Retrieve relevant memories
def retrieve_memory(query, n_results=5):
    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=n_results
    )

    memories = []
    for docs, metas in zip(results["documents"], results["metadatas"]):
        for doc, meta in zip(docs, metas):
            memories.append({
                "role": meta["role"],
                "content": doc
            })

    return memories