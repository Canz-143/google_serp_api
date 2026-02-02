import asyncio
import httpx
import re
from typing import Optional
from app.services.cache import exchange_rate_cache, cache_stats

async def get_exchange_rates_async() -> dict:
    """Get exchange rates with 1-hour caching"""
    if "rates" in exchange_rate_cache:
        cache_stats["rate_hits"] += 1
        print("✓ Exchange rates loaded from cache (1 hour TTL)")
        return exchange_rate_cache["rates"]
    
    cache_stats["rate_misses"] += 1
    print("⚠ Cache miss - fetching fresh exchange rates...")
    
    rates = {}
    default_rates = {
        "AUD": 37.5,
        "USD": 56.5
    }
    
    async def fetch_rate(currency: str, url: str):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    rate = data.get("rates", {}).get("PHP")
                    if rate:
                        print(f"✓ {currency} Exchange rate fetched: 1 {currency} = {rate} PHP")
                        return currency, rate
        except Exception as e:
            print(f"{currency} Exchange rate API error: {str(e)}")
        return currency, None
    
    tasks = [
        fetch_rate("AUD", "https://api.exchangerate-api.com/v4/latest/AUD"),
        fetch_rate("USD", "https://api.exchangerate-api.com/v4/latest/USD")
    ]
    
    results = await asyncio.gather(*tasks)
    
    for currency, rate in results:
        if rate:
            rates[currency] = rate
        else:
            rates[currency] = default_rates[currency]
            print(f"⚠ Using default exchange rate: 1 {currency} = {default_rates[currency]} PHP")
    
    exchange_rate_cache["rates"] = rates
    print("✓ Exchange rates cached for 1 hour")
    
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
