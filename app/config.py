import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SEARCHAPI_KEY")
SEARCHAPI_BASE_URL = "https://www.searchapi.io/api/v1/search"
SIMILARITY_MODEL_NAME = 'all-MiniLM-L6-v2'

# Region Configuration
REGIONS = [
    {"location": "Philippines", "gl": "ph", "currency": "PHP"},
    {"location": "Australia", "gl": "au", "currency": "AUD"},
    {"location": "United States", "gl": "us", "currency": "USD"}
]
