# Intelligent-Ai-Agent.py
# A simple multimodal AI agent with short-term and long-term memory, calculator, and Wikipedia integration

# Install required libraries if not already installed
# Uncomment these lines if running for the first time
# !pip install transformers sentence-transformers faiss-cpu wikipedia --quiet

# Imports
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import wikipedia

# Load a free HuggingFace text generation model
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    device=0  # 0 = GPU, -1 = CPU
)

# Short-term memory (stores recent conversations)
short_term_memory = []

def add_to_memory(user, response):
    short_term_memory.append({"user": user, "response": response})
    if len(short_term_memory) > 5:  # Keep only last 5 conversations
        short_term_memory.pop(0)

# Load embedding model for long-term memory
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create FAISS index
dimension = 384  # embedding size of model
index = faiss.IndexFlatL2(dimension)
long_term_texts = []

def add_long_term_memory(text):
    embedding = embedding_model.encode([text])
    index.add(np.array(embedding))
    long_term_texts.append(text)

def search_long_term_memory(query):
    if len(long_term_texts) == 0:
        return ""
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    return long_term_texts[I[0][0]]

# Calculator tool
def calculator_tool(expression):
    try:
        return str(eval(expression))
    except:
        return "Invalid calculation"

# Wikipedia tool
def wiki_tool(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return "No information found."

# Reasoning logic for the agent
def agent_reasoning(user_input):
    print("🧠 Reasoning Step:")

    # Detect math expressions
    if any(char.isdigit() for char in user_input):
        print("→ Detected mathematical expression. Using calculator tool.")
        return calculator_tool(user_input)

    # Detect knowledge queries
    elif any(word in user_input.lower() for word in ["who", "what", "where", "when"]):
        print("→ Detected knowledge query. Using Wikipedia tool.")
        return wiki_tool(user_input)

    else:
        print("→ Using LLM for general response.")
        context = search_long_term_memory(user_input)
        prompt = f"""
        Context: {context}
        User: {user_input}
        Answer:
        """
        result = generator(prompt, max_length=200)
        return result[0]['generated_text']

# Intelligent agent interface
def intelligent_agent(user_input):
    response = agent_reasoning(user_input)
    add_to_memory(user_input, response)
    add_long_term_memory(user_input + " " + response)
    return response

# Example usage
if __name__ == "__main__":
    print(intelligent_agent("What is Artificial Intelligence?"))
    print(intelligent_agent("2 + 56 * 3"))
    print(intelligent_agent("Who is Elon Musk?"))
