from pydantic import BaseModel
from typing import List, Optional

class Product(BaseModel):
    product_name: str
    price_combined: str
    currency_code: Optional[str] = "N/A"
    website_url: str
    img: str
    website_name: str
    rating: Optional[str] = "N/A"
    reviews: Optional[str] = "N/A"
    region: str
    similarity_score: Optional[int] = None
    similarity_type: Optional[str] = "semantic"

class Data(BaseModel):
    ecommerce_links: List[Product]

class SearchResponse(BaseModel):
    query: str
    total_results: int
    ph_results: int
    au_results: int
    us_results: int
    filtered_results: int
    exchange_rates: dict
    similarity_threshold: Optional[int] = None
    similarity_method: str
    data: Data
    timestamp: str
    processing_time_seconds: float
    cache_hit: bool
