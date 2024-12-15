import faiss

def create_embeddings(text_list, model):
    embeddings = model.encode(text_list, convert_to_numpy=True)
    return embeddings


def get_faiss_indices(ingredients_embeddings, recipes_embeddings):
    # Normalize embeddings
    faiss.normalize_L2(ingredients_embeddings)
    faiss.normalize_L2(recipes_embeddings)
    
    ingredients_index = faiss.IndexFlatL2(ingredients_embeddings.shape[1])
    recipes_index = faiss.IndexFlatL2(recipes_embeddings.shape[1])

    # Add embeddings to FAISS Indices

    ingredients_index.add(ingredients_embeddings)
    recipes_index.add(recipes_embeddings)

    return ingredients_index, recipes_index

def get_rag_query(embedding_model, query, custom_data, top_k=5):
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)

    ingredients, recipes = custom_data['data']
    ingredients_index, recipes_index = custom_data['indices']
    
    
    # using the query find top relevant ingredients and recipes
    
    ing_dist, top_ingredients = ingredients_index.search(query_embedding, top_k)
    recipes_dist, top_recipes = recipes_index.search(query_embedding, top_k)
    
    ingredients_metadata = [f"Ingredient {ingredients[i]}" for i in top_ingredients[0]]
    recipes_metadata = [f"Recipes {recipes[i]}" for i in top_recipes[0]]
    
    # Augment query
    augmented_query = (
        f"User Query: {query}\n\n"
        f"Top Ingredients: {', '.join(ingredients_metadata)}\n"
        f"Top Recipes: {', '.join(recipes_metadata)}\n"
        f"Method:"
    )


    return augmented_query
    
def setup_rag(embedding_model, ingredients, recipes):
    # create eembeddings
    ingredients_embeddings = create_embeddings(ingredients, embedding_model)
    recipes_embeddings = create_embeddings(recipes, embedding_model)
    
    # Normalize embeddings for FAISS and Create index
    ingredients_index, recipes_index = get_faiss_indices(ingredients_embeddings, recipes_embeddings)
    custom_data = {'data': [ingredients, recipes], 'indices':[ingredients_index, recipes_index ]}

    return custom_data

   



    
    
        
    