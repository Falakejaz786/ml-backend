"""MongoDB connection and utilities for the backend."""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# Load environment variables from .env file
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/medley")
print(f"[MongoDB Connection] Using URI: {MONGODB_URI}")

_client = None
_db = None


def get_db():
    """Get MongoDB database connection."""
    global _client, _db

    if _db is None:
        try:
            _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            _client.admin.command('ping')
            _db = _client.get_database()
        except ServerSelectionTimeoutError:
            print(f"Error: Could not connect to MongoDB at {MONGODB_URI}")
            raise

    return _db


def close_db():
    """Close MongoDB connection."""
    global _client
    if _client:
        _client.close()
        _client = None
