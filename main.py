from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from serpapi import GoogleSearch
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
import requests
import re
from fuzzywuzzy import fuzz

load_dotenv()

app = FastAPI(
    title="Google Shopping Search API - Quad Region (PH, AU, US, EU) with Fuzzy Matching",
    description="Search Google Shopping from Philippines, Australia, United States, and Europe with similarity filtering",
    version="4.0.0"
)

API_KEY = os.getenv("API_KEY")

# Pydantic models for request/response
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

class Data(BaseModel):
    ecommerce_links: List[Product]

class SearchResponse(BaseModel):
    query: str
    total_results: int
    ph_results: int
    au_results: int
    us_results: int
    eu_results: int
    filtered_results: int
    exchange_rates: dict
    similarity_threshold: Optional[int] = None
    data: Data
    timestamp: str

def calculate_similarity(query: str, product_name: str) -> int:
    """
    Calculate similarity between query and product name using multiple methods.
    Returns a score from 0-100.
    """
    query_clean = query.lower().strip()
    product_clean = product_name.lower().strip()
    
    score = fuzz.token_sort_ratio(query_clean, product_clean)
    
    return score

def filter_by_similarity(products: List[dict], query: str, threshold: int = 70) -> List[dict]:
    """
    Filter products based on similarity to query.
    Adds similarity_score to each product.
    """
    filtered = []
    
    for product in products:
        similarity = calculate_similarity(query, product['product_name'])
        product['similarity_score'] = similarity
        
        if similarity >= threshold:
            filtered.append(product)
    
    # Sort by similarity score (highest first)
    filtered.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return filtered

def get_exchange_rates() -> dict:
    """Get exchange rates for AUD, USD, and EUR to PHP"""
    rates = {}
    default_rates = {
        "AUD": 37.5,
        "USD": 56.5,
        "EUR": 61.0
    }
    
    # Fetch AUD to PHP
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/AUD", timeout=5)
        if response.status_code == 200:
            data = response.json()
            aud_rate = data.get("rates", {}).get("PHP")
            if aud_rate:
                rates["AUD"] = aud_rate
                print(f"✓ AUD Exchange rate fetched: 1 AUD = {aud_rate} PHP")
    except Exception as e:
        print(f"AUD Exchange rate API error: {str(e)}")
    
    # Fetch USD to PHP
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        if response.status_code == 200:
            data = response.json()
            usd_rate = data.get("rates", {}).get("PHP")
            if usd_rate:
                rates["USD"] = usd_rate
                print(f"✓ USD Exchange rate fetched: 1 USD = {usd_rate} PHP")
    except Exception as e:
        print(f"USD Exchange rate API error: {str(e)}")
    
    # Fetch EUR to PHP
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/EUR", timeout=5)
        if response.status_code == 200:
            data = response.json()
            eur_rate = data.get("rates", {}).get("PHP")
            if eur_rate:
                rates["EUR"] = eur_rate
                print(f"✓ EUR Exchange rate fetched: 1 EUR = {eur_rate} PHP")
    except Exception as e:
        print(f"EUR Exchange rate API error: {str(e)}")
    
    # Use defaults for missing rates
    for currency, default_rate in default_rates.items():
        if currency not in rates:
            rates[currency] = default_rate
            print(f"⚠ Using default exchange rate: 1 {currency} = {default_rate} PHP")
    
    return rates

def extract_numeric_price(price_str: str) -> Optional[float]:
    """Extract numeric value from price string"""
    if not price_str or price_str == "N/A":
        return None
    
    cleaned = re.sub(r'[^\d.]', '', price_str)
    
    try:
        return float(cleaned)
    except ValueError:
        return None

def convert_price_to_php(price_str: str, currency_code: str, exchange_rates: dict) -> str:
    """Convert price to PHP based on currency code"""
    if currency_code == "PHP":
        return price_str
    
    if currency_code in exchange_rates:
        numeric_price = extract_numeric_price(price_str)
        if numeric_price:
            php_price = numeric_price * exchange_rates[currency_code]
            return f"₱{php_price:,.2f}"
    
    return "N/A"

def search_google_shopping_quad_region(search_query: str, num_results: int = 80) -> tuple[List[dict], dict]:
    """
    Search Google Shopping from Philippines, Australia, United States, and Europe
    """
    exchange_rates = get_exchange_rates()
    results_per_region = num_results // 4  # 20 results per region
    
    regions = [
        {"location": "Philippines", "gl": "ph", "currency": "PHP"},
        {"location": "Australia", "gl": "au", "currency": "AUD"},
        {"location": "United States", "gl": "us", "currency": "USD"},
        {"location": "Germany", "gl": "de", "currency": "EUR"}  # Using Germany to represent Europe
    ]
    
    all_products = []
    
    for region in regions:
        params = {
            "engine": "google_shopping",
            "q": search_query,
            "api_key": API_KEY,
            "location": region["location"],
            "hl": "en",
            "gl": region["gl"],
            "num": results_per_region
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "error" in results:
                print(f"API Error for {region['location']}: {results['error']}")
                continue
            
            shopping_results = results.get("shopping_results", [])
            shopping_results = shopping_results[:results_per_region]
            
            for product in shopping_results:
                try:
                    website_url = (
                        product.get("product_link") or 
                        product.get("link") or 
                        product.get("product_url") or
                        "N/A"
                    )
                    
                    img = product.get("thumbnail", "N/A")
                    rating = product.get("rating", "N/A")
                    reviews = product.get("reviews", "N/A")
                    currency_code = region["currency"]
                    
                    # Detect currency from price string
                    price_str = str(product.get("price", ""))
                    if "USD" in price_str or "US$" in price_str or "$" in price_str:
                        if region["gl"] == "us":
                            currency_code = "USD"
                    elif "EUR" in price_str or "€" in price_str:
                        currency_code = "EUR"
                    elif "AUD" in price_str or "A$" in price_str:
                        currency_code = "AUD"
                    
                    original_price = product.get("price", "N/A")
                    price_combined = convert_price_to_php(original_price, currency_code, exchange_rates)
                    
                    # Display region name (use "Europe (Germany)" for clarity)
                    region_display = "Europe (Germany)" if region["location"] == "Germany" else region["location"]
                    
                    all_products.append({
                        "product_name": product.get("title", "N/A"),
                        "price_combined": price_combined,
                        "currency_code": currency_code,
                        "website_url": website_url,
                        "img": img,
                        "website_name": product.get("source", "N/A"),
                        "rating": str(rating) if rating != "N/A" else "N/A",
                        "reviews": str(reviews) if reviews != "N/A" else "N/A",
                        "region": region_display
                    })
                
                except Exception as product_error:
                    print(f"Error processing product in {region['location']}: {str(product_error)}")
                    continue
        
        except Exception as e:
            print(f"Error for {region['location']}: {str(e)}")
            continue
    
    return all_products, exchange_rates

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Google Shopping Search API - Quad Region with Fuzzy Matching</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
            h1 { color: #e74c3c; }
            .endpoint { background: #f4f4f4; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #333; color: #fff; padding: 2px 6px; border-radius: 3px; }
            a { color: #3498db; text-decoration: none; }
            .badge { background: #27ae60; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
            .new { background: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; }
        </style>
    </head>
    <body>
        <h1>🛍️ Google Shopping Search API <span class="badge">4 REGIONS</span> <span class="new">+ Fuzzy Match</span></h1>
        <p>Search Google Shopping from <strong>Philippines, Australia, United States, and Europe</strong> with intelligent similarity filtering!</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <h3>GET /search</h3>
            <p>Search for products with optional similarity filtering</p>
            <p><strong>Parameters:</strong></p>
            <ul>
                <li><code>q</code> - Search query (required)</li>
                <li><code>num_results</code> - Total number of results (default: 80, 20 per region)</li>
                <li><code>similarity_threshold</code> - Minimum similarity % (default: 65)</li>
            </ul>
            <p><strong>Examples:</strong></p>
            <p><a href="/search?q=iPhone 15 Pro&num_results=80&similarity_threshold=70">/search?q=iPhone 15 Pro&similarity_threshold=70</a></p>
            <p><a href="/search?q=MacBook Air&num_results=80&similarity_threshold=80">/search?q=MacBook Air&similarity_threshold=80</a></p>
        </div>
        
        <div class="endpoint">
            <h3>GET /search/html</h3>
            <p>Visual HTML grid with similarity scores (all 4 regions)</p>
            <p><strong>Example:</strong> <a href="/search/html?q=Sony headphones&num_results=80&similarity_threshold=75">/search/html?q=Sony headphones&similarity_threshold=75</a></p>
        </div>
        
        <div class="endpoint">
            <h3>GET /search/csv</h3>
            <p>Download CSV with similarity scores (all 4 regions)</p>
            <p><strong>Example:</strong> <a href="/search/csv?q=gaming laptop&num_results=80&similarity_threshold=70">/search/csv?q=gaming laptop&similarity_threshold=70</a></p>
        </div>
        
        <h2>Supported Regions:</h2>
        <ul>
            <li>🇵🇭 <strong>Philippines</strong> - PHP (Philippine Peso)</li>
            <li>🇦🇺 <strong>Australia</strong> - AUD (Australian Dollar → PHP)</li>
            <li>🇺🇸 <strong>United States</strong> - USD (US Dollar → PHP)</li>
            <li>🇪🇺 <strong>Europe</strong> - EUR (Euro → PHP)</li>
        </ul>
        
        <h2>Similarity Filtering:</h2>
        <ul>
            <li><strong>0</strong>: No filtering (returns all results)</li>
            <li><strong>50-60</strong>: Very loose matching</li>
            <li><strong>70</strong>: Recommended - Good balance (filters irrelevant products)</li>
            <li><strong>80-90</strong>: Strict matching (very specific results)</li>
            <li><strong>100</strong>: Exact match only</li>
        </ul>
        
        <h2>Interactive API Documentation:</h2>
        <p>Visit <a href="/docs">/docs</a> for interactive Swagger UI</p>
    </body>
    </html>
    """
    return html_content

@app.get("/search", response_model=SearchResponse)
async def search_products(
    q: str = Query(..., description="Search query for products"),
    num_results: int = Query(80, ge=4, le=200, description="Total number of results (divisible by 4)"),
    similarity_threshold: int = Query(50, ge=0, le=100, description="Minimum similarity percentage (0=disabled, default=65)")
):
    """
    Search Google Shopping across 4 regions with optional similarity filtering
    
    Set similarity_threshold to 70+ to filter out irrelevant products
    """
    products, exchange_rates = search_google_shopping_quad_region(q, num_results)
    
    # Apply similarity filter if threshold > 0
    if similarity_threshold > 0:
        products = filter_by_similarity(products, q, similarity_threshold)
    else:
        # Add similarity scores even if not filtering
        for product in products:
            product['similarity_score'] = calculate_similarity(q, product['product_name'])
    
    ph_count = sum(1 for p in products if p['region'] == 'Philippines')
    au_count = sum(1 for p in products if p['region'] == 'Australia')
    us_count = sum(1 for p in products if p['region'] == 'United States')
    eu_count = sum(1 for p in products if 'Europe' in p['region'])
    
    return SearchResponse(
        query=q,
        total_results=len(products),
        ph_results=ph_count,
        au_results=au_count,
        us_results=us_count,
        eu_results=eu_count,
        filtered_results=len(products),
        exchange_rates=exchange_rates,
        similarity_threshold=similarity_threshold if similarity_threshold > 0 else None,
        data=Data(ecommerce_links=products),
        timestamp=datetime.now().isoformat()
    )

@app.get("/search/html", response_class=HTMLResponse)
async def search_products_html(
    q: str = Query(..., description="Search query for products"),
    num_results: int = Query(80, ge=4, le=200, description="Total number of results"),
    similarity_threshold: int = Query(65, ge=0, le=100, description="Minimum similarity percentage")
):
    """Search and return visual HTML grid with similarity scores"""
    products, exchange_rates = search_google_shopping_quad_region(q, num_results)
    
    # Apply similarity filter
    if similarity_threshold > 0:
        products = filter_by_similarity(products, q, similarity_threshold)
    else:
        for product in products:
            product['similarity_score'] = calculate_similarity(q, product['product_name'])
    
    ph_count = sum(1 for p in products if p['region'] == 'Philippines')
    au_count = sum(1 for p in products if p['region'] == 'Australia')
    us_count = sum(1 for p in products if p['region'] == 'United States')
    eu_count = sum(1 for p in products if 'Europe' in p['region'])
    
    similarity_info = f" | <strong>Similarity Filter:</strong> ≥{similarity_threshold}%" if similarity_threshold > 0 else ""
    
    # Format exchange rates for display
    rates_display = f"1 USD = {exchange_rates['USD']:.2f} PHP | 1 EUR = {exchange_rates['EUR']:.2f} PHP | 1 AUD = {exchange_rates['AUD']:.2f} PHP"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results: {q}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
            h1 {{ color: #333; text-align: center; }}
            .info {{ text-align: center; color: #666; margin-bottom: 30px; }}
            .region-badge {{ 
                display: inline-block;
                padding: 2px 8px; 
                border-radius: 3px; 
                font-size: 10px; 
                font-weight: bold;
                color: white;
            }}
            .badge-ph {{ background: #3498db; }}
            .badge-au {{ background: #27ae60; }}
            .badge-us {{ background: #e74c3c; }}
            .badge-eu {{ background: #9b59b6; }}
            .similarity-badge {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: #f39c12;
                color: white;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            .grid {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
            .product-card {{ 
                border: 1px solid #ddd; 
                padding: 15px; 
                border-radius: 8px; 
                width: 220px; 
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
                position: relative;
            }}
            .product-card:hover {{ transform: translateY(-5px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
            .product-card img {{ width: 200px; height: 200px; object-fit: contain; }}
            .product-name {{ font-size: 13px; margin: 10px 0 5px 0; height: 40px; overflow: hidden; }}
            .price-combined {{ font-size: 16px; color: #e74c3c; margin: 5px 0; font-weight: bold; }}
            .website-name {{ font-size: 11px; color: #666; margin: 5px 0; }}
            .product-rating {{ font-size: 11px; color: #f39c12; margin: 5px 0; }}
            .product-link {{ 
                display: inline-block;
                font-size: 12px; 
                color: #3498db; 
                text-decoration: none;
                margin-top: 10px;
                padding: 5px 10px;
                border: 1px solid #3498db;
                border-radius: 4px;
            }}
            .product-link:hover {{ background: #3498db; color: white; }}
        </style>
    </head>
    <body>
        <h1>🛍️ Search Results (4 Regions)</h1>
        <div class="info">
            <p><strong>Query:</strong> {q} | <strong>Results:</strong> {len(products)}{similarity_info}</p>
            <p>
                <span class="region-badge badge-ph">🇵🇭 Philippines: {ph_count}</span>
                <span class="region-badge badge-au">🇦🇺 Australia: {au_count}</span>
                <span class="region-badge badge-us">🇺🇸 United States: {us_count}</span>
                <span class="region-badge badge-eu">🇪🇺 Europe: {eu_count}</span>
            </p>
            <p style="font-size: 12px; color: #888;">Exchange Rates: {rates_display}</p>
        </div>
        <div class="grid">
    """
    
    for i, product in enumerate(products, 1):
        if product['img'] != "N/A":
            rating_text = f"⭐ {product['rating']}" if product['rating'] != "N/A" else ""
            reviews_text = f"({product['reviews']} reviews)" if product['reviews'] != "N/A" else ""
            
            # Determine region class and flag
            if product['region'] == "Philippines":
                region_class = "badge-ph"
                region_flag = "🇵🇭"
            elif product['region'] == "Australia":
                region_class = "badge-au"
                region_flag = "🇦🇺"
            elif product['region'] == "United States":
                region_class = "badge-us"
                region_flag = "🇺🇸"
            else:  # Europe
                region_class = "badge-eu"
                region_flag = "🇪🇺"
            
            similarity_score = product.get('similarity_score', 0)
            
            html += f"""
            <div class="product-card">
                <span class="similarity-badge">{similarity_score}%</span>
                <span class="region-badge {region_class}">{region_flag} {product['region']}</span>
                <img src="{product['img']}" alt="{product['product_name']}" onerror="this.src='https://via.placeholder.com/200x200?text=No+Image'">
                <div class="product-name"><strong>{i}. {product['product_name'][:60]}...</strong></div>
                <div class="price-combined">{product['price_combined']}</div>
                <div class="website-name">{product['website_name']}</div>
                <div class="product-rating">{rating_text} {reviews_text}</div>
                <a href="{product['website_url']}" target="_blank" class="product-link">View Product →</a>
            </div>
            """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

@app.get("/search/csv")
async def search_products_csv(
    q: str = Query(..., description="Search query for products"),
    num_results: int = Query(80, ge=4, le=200, description="Total number of results"),
    similarity_threshold: int = Query(0, ge=0, le=100, description="Minimum similarity percentage")
):
    """Search and download results as CSV with similarity scores"""
    products, exchange_rates = search_google_shopping_quad_region(q, num_results)
    
    if similarity_threshold > 0:
        products = filter_by_similarity(products, q, similarity_threshold)
    else:
        for product in products:
            product['similarity_score'] = calculate_similarity(q, product['product_name'])
    
    df = pd.DataFrame(products)
    
    # Reorder columns to show similarity first
    columns = ['similarity_score', 'region'] + [col for col in df.columns if col not in ['similarity_score', 'region']]
    df = df[columns]
    
    filename = f"google_shopping_{q.replace(' ', '_')}_similarity{similarity_threshold}_4REGIONS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    
    return FileResponse(
        filename, 
        media_type='text/csv',
        filename=filename
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Google Shopping Search API - Quad Region with Fuzzy Matching",
        "regions": ["Philippines", "Australia", "United States", "Europe (Germany)"],
        "features": ["similarity_filtering", "multi_currency_conversion", "multi_format_export"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)