from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
from app.config import SIMILARITY_MODEL_NAME
from app.services.cache import embedding_cache, cache_stats

print("🤖 Loading AI similarity model...")
SIMILARITY_MODEL = SentenceTransformer(SIMILARITY_MODEL_NAME)
print("✅ AI model loaded! Using semantic similarity for intelligent matching.")

def get_embedding(text: str):
    """
    Get embedding for text with caching
    Embeddings are cached to avoid recomputing for same products
    """
    # Normalize text for cache key
    text_normalized = text.lower().strip()
    
    # Check cache first
    if text_normalized in embedding_cache:
        cache_stats["embedding_hits"] += 1
        return embedding_cache[text_normalized]
    
    # Cache miss - compute embedding
    cache_stats["embedding_misses"] += 1
    embedding = SIMILARITY_MODEL.encode(text, convert_to_tensor=True)
    
    # Store in cache
    embedding_cache[text_normalized] = embedding
    
    return embedding

def calculate_semantic_similarity(query: str, product_name: str) -> int:
    """
    Calculate semantic similarity using AI embeddings
    Returns score from 0-100
    """
    query_clean = query.lower().strip()
    product_clean = product_name.lower().strip()
    
    # Get embeddings (cached automatically)
    query_embedding = get_embedding(query_clean)
    product_embedding = get_embedding(product_clean)
    
    # Calculate cosine similarity
    similarity = util.cos_sim(query_embedding, product_embedding)
    
    # Convert to 0-100 scale
    score = float(similarity[0][0].item() * 100)
    
    return int(score)

def filter_by_similarity(products: List[dict], query: str, threshold: int = 70) -> List[dict]:
    """Filter products based on AI semantic similarity"""
    filtered = []
    
    print(f"🤖 Running AI semantic similarity for {len(products)} products...")
    
    for product in products:
        similarity = calculate_semantic_similarity(query, product['product_name'])
        product['similarity_score'] = similarity
        product['similarity_type'] = 'semantic'
        
        if threshold == 0 or similarity >= threshold:
            filtered.append(product)
    
    # Sort by similarity score (highest first)
    filtered.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"✅ Filtered to {len(filtered)} products above {threshold}% semantic similarity")
    
    return filtered
