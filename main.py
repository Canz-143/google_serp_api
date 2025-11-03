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

load_dotenv()

app = FastAPI(
    title="Google Shopping Search API - Dual Region (PH & AU)",
    description="Search Google Shopping from Philippines and Australia simultaneously",
    version="2.0.0"
)

# Your API Key (In production, use environment variables!)
API_KEY = os.getenv("API_KEY")

# Pydantic models for request/response
class Product(BaseModel):
    product_name: str
    price_combined: str  # Now contains PHP converted price
    currency_code: Optional[str] = "N/A"
    website_url: str
    img: str
    website_name: str
    rating: Optional[str] = "N/A"
    reviews: Optional[str] = "N/A"
    region: str  # New field to identify which region the product is from

class Data(BaseModel):
    ecommerce_links: List[Product]

class SearchResponse(BaseModel):
    query: str
    total_results: int
    ph_results: int  # Number of results from Philippines
    au_results: int  # Number of results from Australia
    exchange_rate: Optional[float] = None  # AUD to PHP exchange rate
    data: Data
    timestamp: str

def get_exchange_rate() -> float:
    """
    Get AUD to PHP exchange rate from exchangerate-api.com (free tier)
    Falls back to a default rate if API fails
    """
    try:
        # Using exchangerate-api.com free tier (no API key needed)
        response = requests.get("https://api.exchangerate-api.com/v4/latest/AUD", timeout=5)
        if response.status_code == 200:
            data = response.json()
            rate = data.get("rates", {}).get("PHP")
            if rate:
                print(f"✓ Exchange rate fetched: 1 AUD = {rate} PHP")
                return rate
    except Exception as e:
        print(f"Exchange rate API error: {str(e)}")
    
    # Fallback to approximate rate
    default_rate = 37.5  # Approximate AUD to PHP rate
    print(f"⚠ Using default exchange rate: 1 AUD = {default_rate} PHP")
    return default_rate

def extract_numeric_price(price_str: str) -> Optional[float]:
    """
    Extract numeric value from price string
    Examples: "$45.99" -> 45.99, "₱1,234.50" -> 1234.50
    """
    if not price_str or price_str == "N/A":
        return None
    
    # Remove currency symbols and commas
    cleaned = re.sub(r'[^\d.]', '', price_str)
    
    try:
        return float(cleaned)
    except ValueError:
        return None

def convert_price_to_php(price_str: str, currency_code: str, exchange_rate: float) -> str:
    """
    Convert price to PHP if it's in AUD
    """
    if currency_code == "PHP":
        return price_str  # Already in PHP
    
    if currency_code == "AUD":
        numeric_price = extract_numeric_price(price_str)
        if numeric_price:
            php_price = numeric_price * exchange_rate
            return f"₱{php_price:,.2f}"
    
    return "N/A"

def search_google_shopping_dual_region(search_query: str, num_results: int = 40) -> tuple[List[dict], float]:
    """
    Search Google Shopping from both Philippines and Australia
    Splits results evenly between the two regions
    
    Args:
        search_query: The search term
        num_results: Total number of results (split 50/50 between PH and AU)
    
    Returns:
        Tuple of (list of product dictionaries, exchange_rate)
    """
    # Get exchange rate first
    exchange_rate = get_exchange_rate()
    
    # Split results between two regions
    results_per_region = num_results // 2
    
    # Define regions with their country codes
    regions = [
        {"location": "Philippines", "gl": "ph", "currency": "PHP"},
        {"location": "Australia", "gl": "au", "currency": "AUD"}
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
            
            # Check for errors
            if "error" in results:
                print(f"API Error for {region['location']}: {results['error']}")
                continue
            
            shopping_results = results.get("shopping_results", [])
            
            # Limit to exactly the number requested per region
            shopping_results = shopping_results[:results_per_region]
            
            # Extract relevant information
            for product in shopping_results:
                try:
                    # Try multiple URL fields
                    website_url = (
                        product.get("product_link") or 
                        product.get("link") or 
                        product.get("product_url") or
                        "N/A"
                    )
                    
                    # Get thumbnail image
                    img = product.get("thumbnail", "N/A")
                    
                    # Convert rating and reviews to strings
                    rating = product.get("rating", "N/A")
                    reviews = product.get("reviews", "N/A")
                    
                    # Get currency code - Default to region's currency
                    currency_code = region["currency"]  # Default: PHP for Philippines, AUD for Australia
                    
                    # Override only if we can explicitly detect a different currency from the price string
                    price_str = str(product.get("price", ""))
                    if "USD" in price_str or "US$" in price_str:
                        currency_code = "USD"
                    elif "EUR" in price_str or "€" in price_str:
                        currency_code = "EUR"
                    # Otherwise keep the region's default currency
                    
                    # Get original price and convert to PHP
                    original_price = product.get("price", "N/A")
                    price_combined = convert_price_to_php(original_price, currency_code, exchange_rate)
                    
                    all_products.append({
                        "product_name": product.get("title", "N/A"),
                        "price_combined": price_combined,  # Now contains PHP price
                        "currency_code": currency_code,
                        "website_url": website_url,
                        "img": img,
                        "website_name": product.get("source", "N/A"),
                        "rating": str(rating) if rating != "N/A" else "N/A",
                        "reviews": str(reviews) if reviews != "N/A" else "N/A",
                        "region": region["location"]
                    })
                
                except Exception as product_error:
                    print(f"Error processing product in {region['location']}: {str(product_error)}")
                    print(f"Problematic product data: {product}")
                    continue
        
        except Exception as e:
            print(f"Error for {region['location']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_products, exchange_rate

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Google Shopping Search API - Dual Region</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
            h1 { color: #e74c3c; }
            .endpoint { background: #f4f4f4; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #333; color: #fff; padding: 2px 6px; border-radius: 3px; }
            a { color: #3498db; text-decoration: none; }
            .badge { background: #27ae60; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1>🛍️ Google Shopping Search API <span class="badge">PH + AU</span></h1>
        <p>Search Google Shopping from <strong>Philippines</strong> and <strong>Australia</strong> simultaneously!</p>
        <p>Results are automatically split 50/50 between both regions.</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <h3>GET /search</h3>
            <p>Search for products and get JSON response from both regions</p>
            <p><strong>Parameters:</strong></p>
            <ul>
                <li><code>q</code> - Search query (required)</li>
                <li><code>num_results</code> - Total number of results (default: 40, split 20 PH + 20 AU)</li>
            </ul>
            <p><strong>Example:</strong> <a href="/search?q=KitchenAid toaster&num_results=40">/search?q=KitchenAid toaster&num_results=40</a></p>
        </div>
        
        <div class="endpoint">
            <h3>GET /search/html</h3>
            <p>Search for products and get visual HTML grid from both regions</p>
            <p><strong>Parameters:</strong> Same as /search</p>
            <p><strong>Example:</strong> <a href="/search/html?q=KitchenAid toaster&num_results=40">/search/html?q=KitchenAid toaster&num_results=40</a></p>
        </div>
        
        <div class="endpoint">
            <h3>GET /search/csv</h3>
            <p>Search for products and download as CSV file (includes region column)</p>
            <p><strong>Parameters:</strong> Same as /search</p>
            <p><strong>Example:</strong> <a href="/search/csv?q=KitchenAid toaster&num_results=40">/search/csv?q=KitchenAid toaster&num_results=40</a></p>
        </div>
        
        <h2>Interactive API Documentation:</h2>
        <p>Visit <a href="/docs">/docs</a> for interactive Swagger UI</p>
        <p>Visit <a href="/redoc">/redoc</a> for ReDoc documentation</p>
    </body>
    </html>
    """
    return html_content

@app.get("/search", response_model=SearchResponse)
async def search_products(
    q: str = Query(..., description="Search query for products"),
    num_results: int = Query(40, ge=2, le=100, description="Total number of results (split between PH and AU)")
):
    """
    Search Google Shopping for products from both Philippines and Australia
    Results are split 50/50 between the two regions
    
    Returns JSON response with product details
    """
    products, exchange_rate = search_google_shopping_dual_region(q, num_results)
    
    # Count results per region
    ph_count = sum(1 for p in products if p['region'] == 'Philippines')
    au_count = sum(1 for p in products if p['region'] == 'Australia')
    
    return SearchResponse(
        query=q,
        total_results=len(products),
        ph_results=ph_count,
        au_results=au_count,
        exchange_rate=exchange_rate,
        data=Data(ecommerce_links=products),
        timestamp=datetime.now().isoformat()
    )

@app.get("/search/html", response_class=HTMLResponse)
async def search_products_html(
    q: str = Query(..., description="Search query for products"),
    num_results: int = Query(40, ge=2, le=100, description="Total number of results (split between PH and AU)")
):
    """
    Search Google Shopping and return visual HTML grid from both regions
    """
    products, exchange_rate = search_google_shopping_dual_region(q, num_results)
    
    # Count results per region
    ph_count = sum(1 for p in products if p['region'] == 'Philippines')
    au_count = sum(1 for p in products if p['region'] == 'Australia')
    
    # Generate HTML grid
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
        <h1>🛍️ Search Results (PH + AU)</h1>
        <div class="info">
            <p><strong>Query:</strong> {q} | <strong>Total Results:</strong> {len(products)}</p>
            <p>
                <span class="region-badge badge-ph">🇵🇭 Philippines: {ph_count}</span>
                <span class="region-badge badge-au">🇦🇺 Australia: {au_count}</span>
            </p>
            <p style="font-size: 12px; color: #888;">Exchange Rate: 1 AUD = {exchange_rate:.2f} PHP</p>
            <p style="font-size: 12px; color: #888; font-style: italic;">All prices shown in PHP (₱)</p>
        </div>
        <div class="grid">
    """
    
    for i, product in enumerate(products, 1):
        if product['img'] != "N/A":
            rating_text = f"⭐ {product['rating']}" if product['rating'] != "N/A" else ""
            reviews_text = f"({product['reviews']} reviews)" if product['reviews'] != "N/A" else ""
            
            # Determine region badge
            region_class = "badge-ph" if product['region'] == "Philippines" else "badge-au"
            region_flag = "🇵🇭" if product['region'] == "Philippines" else "🇦🇺"
            
            html += f"""
            <div class="product-card">
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
    num_results: int = Query(40, ge=2, le=100, description="Total number of results (split between PH and AU)")
):
    """
    Search Google Shopping and download results as CSV from both regions
    """
    products, exchange_rate = search_google_shopping_dual_region(q, num_results)
    
    # Create DataFrame
    df = pd.DataFrame(products)
    
    # Reorder columns to put region first
    columns = ['region'] + [col for col in df.columns if col != 'region']
    df = df[columns]
    
    # Save to CSV
    filename = f"google_shopping_{q.replace(' ', '_')}_PH_AU_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
        "service": "Google Shopping Search API - Dual Region (PH + AU)",
        "regions": ["Philippines", "Australia"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)