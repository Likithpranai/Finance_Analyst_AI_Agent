"""
Cache Manager for Finance Analyst AI Agent

This module provides caching functionality to reduce API calls and improve performance.
It uses Redis for caching frequently queried data with configurable expiration times.
"""

import os
import json
import time
import redis
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Redis client for caching
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("Redis cache connected successfully")
except Exception as e:
    print(f"Redis cache not available: {str(e)}")
    REDIS_AVAILABLE = False


def cache_result(expiry_seconds=300):
    """
    Decorator to cache function results in Redis
    
    Args:
        expiry_seconds: Time in seconds before cache entry expires
        
    Returns:
        Decorated function with caching capability
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip caching if Redis is not available
            if not REDIS_AVAILABLE:
                return func(*args, **kwargs)
            
            # Generate a cache key based on function name and arguments
            func_name = func.__name__
            # Convert args and kwargs to a string representation
            args_str = json.dumps(args, sort_keys=True, default=str)
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
            cache_key = f"finance_agent:{func_name}:{args_str}:{kwargs_str}"
            
            # Try to get from cache
            try:
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                print(f"Cache retrieval error: {str(e)}")
                # Continue with function execution if cache retrieval fails
            
            # Execute the function if not in cache
            result = func(*args, **kwargs)
            
            # Save to cache
            try:
                redis_client.setex(
                    cache_key,
                    expiry_seconds,
                    json.dumps(result, default=str)
                )
            except Exception as e:
                print(f"Cache saving error: {str(e)}")
            
            return result
        return wrapper
    return decorator


class CacheManager:
    """Utility class for managing cache operations"""
    
    @staticmethod
    def is_redis_available():
        """Check if Redis is available"""
        try:
            r = CacheManager.get_redis_connection()
            return r is not None and r.ping()
        except Exception:
            return False
    
    @staticmethod
    def get_redis_connection():
        """Get a Redis connection using environment variables"""
        try:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", 6379))
            db = int(os.getenv("REDIS_DB", 0))
            password = os.getenv("REDIS_PASSWORD", None)
            
            r = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_timeout=5
            )
            
            # Test connection
            r.ping()
            return r
        except Exception as e:
            print(f"Redis cache not available: {str(e)}")
            return None
    
    @staticmethod
    def clear_cache(pattern="finance_agent:*"):
        """
        Clear cache entries matching a pattern
        
        Args:
            pattern: Redis key pattern to match for deletion
            
        Returns:
            Number of keys deleted
        """
        if not REDIS_AVAILABLE:
            return 0
        
        try:
            # Find keys matching the pattern
            keys = redis_client.keys(pattern)
            
            # Delete the keys
            if keys:
                return redis_client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Cache clearing error: {str(e)}")
            return 0
    
    @staticmethod
    def get_cache_stats():
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        if not REDIS_AVAILABLE:
            return {"status": "Redis not available"}
        
        try:
            # Get Redis info
            info = redis_client.info()
            
            # Count finance agent keys
            finance_keys = len(redis_client.keys("finance_agent:*"))
            
            return {
                "status": "Connected",
                "total_keys": info.get("db0", {}).get("keys", 0),
                "finance_agent_keys": finance_keys,
                "memory_used": f"{info.get('used_memory_human', '0')}",
                "uptime": f"{info.get('uptime_in_seconds', 0) / 3600:.1f} hours",
                "clients_connected": info.get("connected_clients", 0)
            }
        except Exception as e:
            print(f"Cache stats error: {str(e)}")
            return {"status": f"Error: {str(e)}"}


# Example usage
if __name__ == "__main__":
    # Example function with caching
    @cache_result(expiry_seconds=60)
    def example_function(param1, param2):
        print("Executing expensive operation...")
        time.sleep(2)  # Simulate expensive operation
        return {"result": param1 + param2, "timestamp": time.time()}
    
    # Test the cached function
    print("First call (should execute):")
    result1 = example_function(1, 2)
    print(result1)
    
    print("\nSecond call (should use cache):")
    result2 = example_function(1, 2)
    print(result2)
    
    print("\nDifferent parameters (should execute):")
    result3 = example_function(3, 4)
    print(result3)
    
    # Test cache stats
    print("\nCache statistics:")
    stats = CacheManager.get_cache_stats()
    print(json.dumps(stats, indent=2))
    
    # Test cache clearing
    print("\nClearing cache:")
    cleared = CacheManager.clear_cache()
    print(f"Cleared {cleared} cache entries")
