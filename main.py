from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from rag_utils import create_embeddings, get_faiss_indices, get_rag_query, setup_rag
import json

# get data
with open("recipes.json", "r") as file:
    data = json.load(file)

custom_ingredients = data['ingredients']
custom_recipes = data['recipes']

# load embedding model and setup rag
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
custom_data = setup_rag(embedding_model, custom_ingredients, custom_recipes)


# load LLM and query
llm = pipeline("text-generation", model="distilgpt2", device=-1)
query = "What can I cook with tomatoes and chicken?"
rag_query = get_rag_query(embedding_model, query, custom_data, top_k=3)

# increasing max_length might give error depends on which llm is used
response = llm(rag_query, max_length=1024, num_return_sequences=1) 

print(f"{response[0]['generated_text'].split('Method:')[1]}")



