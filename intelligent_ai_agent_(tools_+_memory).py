from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import wikipedia

# Load model
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    device=-1  # CPU (important for deployment)
)

short_term_memory = []

def add_to_memory(user, response):
    short_term_memory.append({"user": user, "response": response})
    if len(short_term_memory) > 5:
        short_term_memory.pop(0)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384
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

def calculator_tool(expression):
    try:
        return str(eval(expression))
    except:
        return "Invalid calculation"

def wiki_tool(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return "No information found."

def agent_reasoning(user_input):

    if any(char.isdigit() for char in user_input):
        return calculator_tool(user_input)

    elif any(word in user_input.lower() for word in ["who", "what", "where", "when"]):
        return wiki_tool(user_input)

    else:
        context = search_long_term_memory(user_input)
        prompt = f"""
        Context: {context}
        User: {user_input}
        Answer:
        """
        result = generator(prompt, max_length=200)
        return result[0]['generated_text']

def intelligent_agent(user_input):
    response = agent_reasoning(user_input)
    add_to_memory(user_input, response)
    add_long_term_memory(user_input + " " + response)
    return response
