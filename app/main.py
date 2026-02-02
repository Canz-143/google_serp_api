from fastapi import FastAPI, Query
from datetime import datetime

from app.models import SearchResponse, Data, Product
from app.services.cache import search_cache, exchange_rate_cache, embedding_cache, cache_stats, get_cache_info
from app.services.ai import calculate_semantic_similarity, filter_by_similarity, SIMILARITY_MODEL_NAME
from app.services.search import search_google_shopping_triple_region_parallel
from app.utils import generate_cache_key

app = FastAPI(
    title="Google Shopping Search API - AI Semantic Similarity Edition",
    description="Search Google Shopping with AI-powered semantic matching using Sentence Transformers",
    version="8.0.0"
)

@app.get("/search", response_model=SearchResponse)
async def search_products(
    q: str = Query(..., description="Search query for products"),
    num_results: int = Query(60, ge=3, le=300, description="Total number of results"),
    similarity_threshold: int = Query(50, ge=0, le=100, description="Minimum semantic similarity percentage")
):
    """
    Search with AI-powered semantic similarity
    
    The AI understands meaning, not just characters:
    - "laptop" matches "notebook computer"
    - "iPhone" matches "Apple smartphone"
    - "gaming laptop" ranks gaming products higher
    """
    start_time = datetime.now()
    
    cache_key = generate_cache_key(q, num_results, similarity_threshold)
    
    if cache_key in search_cache:
        cache_stats["search_hits"] += 1
        cached_response = search_cache[cache_key]
        print(f"✓ Cache HIT for query '{q}' (key: {cache_key[:8]}...)")
        
        cached_response["timestamp"] = datetime.now().isoformat()
        cached_response["cache_hit"] = True
        cached_response["processing_time_seconds"] = round((datetime.now() - start_time).total_seconds(), 4)
        
        return SearchResponse(**cached_response)
    
    cache_stats["search_misses"] += 1
    print(f"⚠ Cache MISS for query '{q}' (key: {cache_key[:8]}...) - performing search...")
    
    products, exchange_rates = await search_google_shopping_triple_region_parallel(q, num_results)
    
    # Apply AI semantic similarity filter
    if similarity_threshold > 0:
        products = filter_by_similarity(products, q, similarity_threshold)
    else:
        for product in products:
            product['similarity_score'] = calculate_semantic_similarity(q, product['product_name'])
            product['similarity_type'] = 'semantic'
    
    ph_count = sum(1 for p in products if p['region'] == 'Philippines')
    au_count = sum(1 for p in products if p['region'] == 'Australia')
    us_count = sum(1 for p in products if p['region'] == 'United States')
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    response_data = {
        "query": q,
        "total_results": len(products),
        "ph_results": ph_count,
        "au_results": au_count,
        "us_results": us_count,
        "filtered_results": len(products),
        "exchange_rates": exchange_rates,
        "similarity_threshold": similarity_threshold if similarity_threshold > 0 else None,
        "similarity_method": "AI Semantic (Sentence Transformers)",
        "data": {"ecommerce_links": products},
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": round(processing_time, 2),
        "cache_hit": False
    }
    
    search_cache[cache_key] = response_data
    print(f"✓ Results cached with key: {cache_key[:8]}... (TTL: 15 min)")
    
    return SearchResponse(**response_data)

@app.get("/cache/stats")
async def cache_statistics():
    """Get detailed cache statistics including AI embeddings"""
    cache_info = get_cache_info()
    
    sample_keys = list(search_cache.keys())[:5]
    
    return {
        "cache_statistics": cache_info,
        "search_cache": {
            "current_size": len(search_cache),
            "max_size": search_cache.maxsize,
            "ttl_seconds": 900,
            "ttl_human": "15 minutes",
            "sample_cached_keys": sample_keys
        },
        "embedding_cache": {
            "current_size": len(embedding_cache),
            "max_size": embedding_cache.maxsize,
            "ttl_seconds": 3600,
            "ttl_human": "1 hour",
            "description": "Caches AI embeddings for products to speed up similarity calculations"
        },
        "exchange_rate_cache": {
            "current_size": len(exchange_rate_cache),
            "max_size": exchange_rate_cache.maxsize,
            "ttl_seconds": 3600,
            "ttl_human": "1 hour",
            "is_cached": "rates" in exchange_rate_cache
        },
        "performance_impact": {
            "cache_hit_speed": "~0.01 seconds",
            "cache_miss_speed": "~2-3 seconds",
            "speed_improvement": "~200-300x faster",
            "ai_similarity": "~0.01s per comparison (with embedding cache)"
        },
        "ai_features": {
            "model": SIMILARITY_MODEL_NAME,
            "model_size": "80MB",
            "similarity_type": "semantic",
            "cost": "FREE (runs locally)",
            "embedding_cache_enabled": True
        },
        "provider": "SearchAPI.io"
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches including embeddings"""
    search_cache.clear()
    exchange_rate_cache.clear()
    embedding_cache.clear()
    
    cache_stats["search_hits"] = 0
    cache_stats["search_misses"] = 0
    cache_stats["rate_hits"] = 0
    cache_stats["rate_misses"] = 0
    cache_stats["embedding_hits"] = 0
    cache_stats["embedding_misses"] = 0
    
    return {
        "status": "success",
        "message": "All caches cleared (including AI embeddings)",
        "search_cache_size": len(search_cache),
        "rate_cache_size": len(exchange_rate_cache),
        "embedding_cache_size": len(embedding_cache)
    }

@app.get("/health")
async def health_check():
    """Health check with cache and AI model info"""
    cache_info = get_cache_info()
    
    return {
        "status": "healthy", 
        "service": "Google Shopping Search API - AI Semantic Edition",
        "version": "8.0.0",
        "provider": "SearchAPI.io",
        "ai_model": {
            "name": SIMILARITY_MODEL_NAME,
            "type": "Sentence Transformers",
            "size": "80MB",
            "loaded": True,
            "similarity_type": "semantic"
        },
        "regions": ["Philippines", "Australia", "United States"],
        "features": [
            "ai_semantic_similarity",
            "parallel_region_searches",
            "parallel_exchange_rates",
            "intelligent_caching",
            "embedding_caching",
            "multi_currency_conversion",
            "multi_format_export"
        ],
        "performance": {
            "search_speed": "2-3s (first) → 0.01s (cached)",
            "cache_hit_rate": cache_info["search_hit_rate"],
            "embedding_hit_rate": cache_info["embedding_hit_rate"]
        },
        "cache_status": {
            "search_cache_entries": cache_info["search_cache_size"],
            "embedding_cache_entries": cache_info["embedding_cache_size"],
            "total_requests": cache_info["search_hits"] + cache_info["search_misses"]
        }
    }
