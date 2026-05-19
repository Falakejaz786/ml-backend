import os
import threading

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/medley")

_client: MongoClient | None = None
_db = None
_lock = threading.Lock()  # safe for multi-threaded servers (Flask, FastAPI, etc.)


def get_db():
    global _client, _db

    if _db is not None:
        return _db

    with _lock:
        if _db is not None:
            return _db

        try:
            _client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10_000,
            )
            _client.admin.command('ping')
            _db = _client.get_database()
            print(f"[MongoDB] Connected → database name: '{_db.name}'")
            print(f"[MongoDB] Collections: {_db.list_collection_names()}")
        except (ServerSelectionTimeoutError, ConnectionFailure) as exc:
            print(f"[MongoDB] Could not connect to {MONGODB_URI}: {exc}")
            raise

    return _db


def close_db():
    global _client, _db

    with _lock:
        if _client:
            _client.close()
            print("[MongoDB] Connection closed.")
        _client = None
        _db = None
