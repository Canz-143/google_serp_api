from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from serpapi import GoogleSearch
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Google Shopping Search API",
    description="Search Google Shopping and retrieve product information",
    version="1.0.0"
)

# Your API Key (In production, use environment variables!)
API_KEY = os.getenv("API_KEY")

# Pydantic models for request/response
class Product(BaseModel):
    product_name: str
    price_combined: str
    website_url: str
    img: str
    website_name: str
    rating: Optional[str] = "N/A"
    reviews: Optional[str] = "N/A"

class Data(BaseModel):
    ecommerce_links: List[Product]

class SearchResponse(BaseModel):
    query: str
    total_results: int
    data: Data
    timestamp: str

def search_google_shopping(search_query: str, location: str = "Philippines", num_results: int = 20) -> List[dict]:
    """
    Search Google Shopping and return product information
    
    Args:
        search_query: The search term
        location: Location for search results (default: Philippines)
        num_results: Number of results to return (default: 20, max: 100)
    
    Returns:
        List of dictionaries with product information
    """
    params = {
        "engine": "google_shopping",
        "q": search_query,
        "api_key": API_KEY,
        "location": location,
        "hl": "en",
        "gl": "ph",
        "num": num_results
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Check for errors
        if "error" in results:
            raise HTTPException(status_code=400, detail=f"API Error: {results['error']}")
        
        shopping_results = results.get("shopping_results", [])
        
        # Extract relevant information
        products = []
        for product in shopping_results:
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
            
            products.append({
                "product_name": product.get("title", "N/A"),
                "price_combined": product.get("price", "N/A"),
                "website_url": website_url,
                "img": img,
                "website_name": product.get("source", "N/A"),
                "rating": str(rating) if rating != "N/A" else "N/A",
                "reviews": str(reviews) if reviews != "N/A" else "N/A"
            })
        
        return products
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Google Shopping Search API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
            h1 { color: #e74c3c; }
            .endpoint { background: #f4f4f4; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #333; color: #fff; padding: 2px 6px; border-radius: 3px; }
            a { color: #3498db; text-decoration: none; }
        </style>
    </head>
    <body>
        <h1>🛍️ Google Shopping Search API</h1>
        <p>Search Google Shopping and retrieve product information via API</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <h3>GET /search</h3>
            <p>Search for products and get JSON response</p>
            <p><strong>Parameters:</strong></p>
            <ul>
                <li><code>q</code> - Search query (required)</li>
                <li><code>location</code> - Location (default: Philippines)</li>
                <li><code>num_results</code> - Number of results (default: 20, max: 100)</li>
            </ul>
            <p><strong>Example:</strong> <a href="/search?q=KitchenAid toaster&num_results=10">/search?q=KitchenAid toaster&num_results=10</a></p>
        </div>
        
        <div class="endpoint">
            <h3>GET /search/html</h3>
            <p>Search for products and get visual HTML grid</p>
            <p><strong>Parameters:</strong> Same as /search</p>
            <p><strong>Example:</strong> <a href="/search/html?q=KitchenAid toaster&num_results=10">/search/html?q=KitchenAid toaster&num_results=10</a></p>
        </div>
        
        <div class="endpoint">
            <h3>GET /search/csv</h3>
            <p>Search for products and download as CSV file</p>
            <p><strong>Parameters:</strong> Same as /search</p>
            <p><strong>Example:</strong> <a href="/search/csv?q=KitchenAid toaster&num_results=10">/search/csv?q=KitchenAid toaster&num_results=10</a></p>
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
    location: str = Query("Philippines", description="Location for search results"),
    num_results: int = Query(20, ge=1, le=100, description="Number of results to return")
):
    """
    Search Google Shopping for products
    
    Returns JSON response with product details
    """
    products = search_google_shopping(q, location, num_results)
    
    return SearchResponse(
        query=q,
        total_results=len(products),
        data=Data(ecommerce_links=products),
        timestamp=datetime.now().isoformat()
    )

@app.get("/search/html", response_class=HTMLResponse)
async def search_products_html(
    q: str = Query(..., description="Search query for products"),
    location: str = Query("Philippines", description="Location for search results"),
    num_results: int = Query(20, ge=1, le=100, description="Number of results to return")
):
    """
    Search Google Shopping and return visual HTML grid
    """
    products = search_google_shopping(q, location, num_results)
    
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
            .grid {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
            .product-card {{ 
                border: 1px solid #ddd; 
                padding: 15px; 
                border-radius: 8px; 
                width: 220px; 
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
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
        <h1>🛍️ Search Results</h1>
        <div class="info">
            <p><strong>Query:</strong> {q} | <strong>Results:</strong> {len(products)} | <strong>Location:</strong> {location}</p>
        </div>
        <div class="grid">
    """
    
    for i, product in enumerate(products, 1):
        if product['img'] != "N/A":
            rating_text = f"⭐ {product['rating']}" if product['rating'] != "N/A" else ""
            reviews_text = f"({product['reviews']} reviews)" if product['reviews'] != "N/A" else ""
            
            html += f"""
            <div class="product-card">
                <img src="{product['img']}" alt="{product['product_name']}">
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
    location: str = Query("Philippines", description="Location for search results"),
    num_results: int = Query(20, ge=1, le=100, description="Number of results to return")
):
    """
    Search Google Shopping and download results as CSV
    """
    products = search_google_shopping(q, location, num_results)
    
    # Create DataFrame
    df = pd.DataFrame(products)
    
    # Save to CSV
    filename = f"google_shopping_{q.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    
    return FileResponse(
        filename, 
        media_type='text/csv',
        filename=filename
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Google Shopping Search API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)