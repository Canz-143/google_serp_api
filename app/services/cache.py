from cachetools import TTLCache

# Cache for search results (TTL = 15 minutes, max 1000 entries)
search_cache = TTLCache(maxsize=1000, ttl=900)

# Cache for exchange rates (TTL = 1 hour, max 1 entry needed)
exchange_rate_cache = TTLCache(maxsize=1, ttl=3600)

# Cache for embeddings (TTL = 1 hour, max 10000 entries)
embedding_cache = TTLCache(maxsize=10000, ttl=3600)

# Cache statistics
cache_stats = {
    "search_hits": 0,
    "search_misses": 0,
    "rate_hits": 0,
    "rate_misses": 0,
    "embedding_hits": 0,
    "embedding_misses": 0
}

def get_cache_info():
    """Get current cache statistics"""
    total_searches = cache_stats["search_hits"] + cache_stats["search_misses"]
    hit_rate = (cache_stats["search_hits"] / total_searches * 100) if total_searches > 0 else 0
    
    total_embeddings = cache_stats["embedding_hits"] + cache_stats["embedding_misses"]
    embedding_hit_rate = (cache_stats["embedding_hits"] / total_embeddings * 100) if total_embeddings > 0 else 0
    
    return {
        "search_cache_size": len(search_cache),
        "search_cache_max": search_cache.maxsize,
        "search_hits": cache_stats["search_hits"],
        "search_misses": cache_stats["search_misses"],
        "search_hit_rate": f"{hit_rate:.1f}%",
        "embedding_cache_size": len(embedding_cache),
        "embedding_hits": cache_stats["embedding_hits"],
        "embedding_misses": cache_stats["embedding_misses"],
        "embedding_hit_rate": f"{embedding_hit_rate:.1f}%",
        "rate_cache_size": len(exchange_rate_cache),
        "rate_hits": cache_stats["rate_hits"],
        "rate_misses": cache_stats["rate_misses"]
    }
