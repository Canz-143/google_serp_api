import httpx
import asyncio
from datetime import datetime
from typing import List, Tuple
from app.config import API_KEY, SEARCHAPI_BASE_URL, REGIONS
from app.services.rates import get_exchange_rates_async, convert_price_to_php

async def search_single_region(region: dict, search_query: str, results_per_region: int, exchange_rates: dict) -> List[dict]:
    """Search a single region asynchronously using SearchAPI.io"""
    params = {
        "engine": "google_shopping",
        "q": search_query,
        "api_key": API_KEY,
        "location": region["location"],
        "hl": "en",
        "gl": region["gl"],
    }
    
    products = []
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(SEARCHAPI_BASE_URL, params=params)
            
            if response.status_code != 200:
                print(f"API Error for {region['location']}: Status {response.status_code}")
                return products
            
            results = response.json()
            
            if "error" in results:
                print(f"API Error for {region['location']}: {results['error']}")
                return products
            
            shopping_results = results.get("shopping_results", [])
            shopping_results = shopping_results[:results_per_region]
            
            print(f"✓ {region['location']}: Retrieved {len(shopping_results)} products")
            
            for product in shopping_results:
                try:
                    website_url = (
                        product.get("link") or
                        product.get("product_link") or
                        product.get("product_url") or
                        "N/A"
                    )
                    
                    img = product.get("thumbnail", "N/A")
                    rating = product.get("rating", "N/A")
                    reviews = product.get("reviews", "N/A")
                    currency_code = region["currency"]
                    
                    price_str = str(product.get("price", ""))
                    if "USD" in price_str or "US$" in price_str or "$" in price_str:
                        if region["gl"] == "us":
                            currency_code = "USD"
                    elif "AUD" in price_str or "A$" in price_str:
                        currency_code = "AUD"
                    
                    original_price = product.get("price", "N/A")
                    
                    if product.get("extracted_price"):
                        extracted = product.get("extracted_price")
                        if currency_code == "PHP":
                            price_combined = f"₱{extracted:,.2f}"
                        else:
                            price_combined = convert_price_to_php(str(extracted), currency_code, exchange_rates)
                    else:
                        price_combined = convert_price_to_php(original_price, currency_code, exchange_rates)
                    
                    seller = product.get("seller", product.get("source", "N/A"))
                    
                    products.append({
                        "product_name": product.get("title", "N/A"),
                        "price_combined": price_combined,
                        "currency_code": currency_code,
                        "website_url": website_url,
                        "img": img,
                        "website_name": seller,
                        "rating": str(rating) if rating != "N/A" else "N/A",
                        "reviews": str(reviews) if reviews != "N/A" else "N/A",
                        "region": region["location"]
                    })
                
                except Exception as product_error:
                    print(f"Error processing product in {region['location']}: {str(product_error)}")
                    continue
    
    except Exception as e:
        print(f"Error for {region['location']}: {str(e)}")
    
    return products

async def search_google_shopping_triple_region_parallel(search_query: str, num_results: int = 90) -> Tuple[List[dict], dict]:
    """Search Google Shopping from all 3 regions IN PARALLEL using SearchAPI.io"""
    start_time = datetime.now()
    
    exchange_rates = await get_exchange_rates_async()
    
    results_per_region = num_results // 3
    
    tasks = [
        search_single_region(region, search_query, results_per_region, exchange_rates)
        for region in REGIONS
    ]
    
    print(f"🚀 Starting parallel search across {len(REGIONS)} regions (SearchAPI.io)...")
    region_results = await asyncio.gather(*tasks)
    
    all_products = []
    for products_list in region_results:
        all_products.extend(products_list)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✅ Parallel search completed in {elapsed:.2f} seconds")
    
    return all_products, exchange_rates
