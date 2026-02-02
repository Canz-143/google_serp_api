import hashlib

def generate_cache_key(query: str, num_results: int, similarity_threshold: int) -> str:
    """Generate a unique cache key for a search query"""
    normalized_query = query.lower().strip()
    cache_string = f"{normalized_query}|{num_results}|{similarity_threshold}"
    cache_key = hashlib.md5(cache_string.encode()).hexdigest()
    return cache_key
