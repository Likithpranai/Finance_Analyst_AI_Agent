"""
Agent Memory System for Finance Analyst AI Agent Framework
Provides persistent memory capabilities for maintaining context across conversations
"""
import os
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import redis

class AgentMemory:
    """
    Memory system for Finance Analyst AI Agent Framework
    Provides persistent storage for conversation history, analysis results, and user preferences
    """
    
    def __init__(self, use_redis: bool = True):
        """
        Initialize the agent memory system
        
        Args:
            use_redis: Whether to use Redis for memory storage (falls back to in-memory if Redis is unavailable)
        """
        self.use_redis = use_redis
        self.redis_client = self._initialize_redis() if use_redis else None
        self.in_memory_storage = {
            "conversations": [],
            "analysis_results": {},
            "user_preferences": {},
            "watched_symbols": [],
            "alerts": []
        }
    
    def _initialize_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection with fallback to in-memory storage"""
        try:
            # Try to connect to Redis using environment variables or default values
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            db = int(os.getenv("REDIS_DB", "0"))
            password = os.getenv("REDIS_PASSWORD", None)
            
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=5,
                decode_responses=True
            )
            
            # Test connection
            client.ping()
            print("Successfully connected to Redis for memory storage")
            return client
        except Exception as e:
            print(f"Failed to connect to Redis: {str(e)}")
            print("Falling back to in-memory storage for agent memory")
            self.use_redis = False
            return None
    
    def is_redis_available(self) -> bool:
        """Check if Redis is available"""
        if not self.use_redis or self.redis_client is None:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def _get_redis_key(self, memory_type: str, key: str = None) -> str:
        """Generate a Redis key with proper namespacing"""
        base_key = f"finance_agent:memory:{memory_type}"
        if key:
            return f"{base_key}:{key}"
        return base_key
    
    def add_conversation_entry(self, user_query: str, agent_response: str, 
                              symbols: List[str] = None, analysis_types: List[str] = None) -> None:
        """
        Add a conversation entry to memory
        
        Args:
            user_query: User's query
            agent_response: Agent's response
            symbols: List of stock symbols mentioned in the conversation
            analysis_types: List of analysis types performed (technical, fundamental, etc.)
        """
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "user_query": user_query,
            "agent_response": agent_response,
            "symbols": symbols or [],
            "analysis_types": analysis_types or []
        }
        
        if self.is_redis_available():
            # Store in Redis
            key = self._get_redis_key("conversations", timestamp)
            self.redis_client.set(key, json.dumps(entry))
            
            # Add to conversation index
            index_key = self._get_redis_key("conversation_index")
            self.redis_client.lpush(index_key, timestamp)
            
            # Trim to last 100 conversations
            self.redis_client.ltrim(index_key, 0, 99)
            
            # Add symbol references
            for symbol in (symbols or []):
                symbol_key = self._get_redis_key("symbol_conversations", symbol)
                self.redis_client.lpush(symbol_key, timestamp)
                self.redis_client.ltrim(symbol_key, 0, 49)  # Keep last 50 conversations per symbol
        else:
            # Store in memory
            self.in_memory_storage["conversations"].append(entry)
            
            # Trim to last 100 conversations
            if len(self.in_memory_storage["conversations"]) > 100:
                self.in_memory_storage["conversations"] = self.in_memory_storage["conversations"][-100:]
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """
        Get recent conversations from memory
        
        Args:
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation entries
        """
        if self.is_redis_available():
            # Get conversation timestamps from index
            index_key = self._get_redis_key("conversation_index")
            timestamps = self.redis_client.lrange(index_key, 0, limit - 1)
            
            # Get conversation entries
            conversations = []
            for timestamp in timestamps:
                key = self._get_redis_key("conversations", timestamp)
                entry_json = self.redis_client.get(key)
                if entry_json:
                    conversations.append(json.loads(entry_json))
            
            return conversations
        else:
            # Get from in-memory storage
            return self.in_memory_storage["conversations"][-limit:]
    
    def get_symbol_conversations(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get conversations related to a specific symbol
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation entries related to the symbol
        """
        if self.is_redis_available():
            # Get conversation timestamps for symbol
            symbol_key = self._get_redis_key("symbol_conversations", symbol)
            timestamps = self.redis_client.lrange(symbol_key, 0, limit - 1)
            
            # Get conversation entries
            conversations = []
            for timestamp in timestamps:
                key = self._get_redis_key("conversations", timestamp)
                entry_json = self.redis_client.get(key)
                if entry_json:
                    conversations.append(json.loads(entry_json))
            
            return conversations
        else:
            # Filter in-memory storage by symbol
            symbol_conversations = [
                entry for entry in self.in_memory_storage["conversations"]
                if symbol in (entry.get("symbols") or [])
            ]
            return symbol_conversations[-limit:]
    
    def store_analysis_result(self, symbol: str, analysis_type: str, result: Any) -> None:
        """
        Store an analysis result in memory
        
        Args:
            symbol: Stock symbol
            analysis_type: Type of analysis (technical, fundamental, etc.)
            result: Analysis result data
        """
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "symbol": symbol,
            "analysis_type": analysis_type,
            "result": result
        }
        
        if self.is_redis_available():
            # Store in Redis
            key = self._get_redis_key("analysis", f"{symbol}:{analysis_type}")
            self.redis_client.set(key, json.dumps(entry))
            
            # Set expiration (24 hours)
            self.redis_client.expire(key, 86400)
        else:
            # Store in memory
            key = f"{symbol}:{analysis_type}"
            self.in_memory_storage["analysis_results"][key] = entry
    
    def get_analysis_result(self, symbol: str, analysis_type: str) -> Optional[Dict]:
        """
        Get a stored analysis result from memory
        
        Args:
            symbol: Stock symbol
            analysis_type: Type of analysis (technical, fundamental, etc.)
            
        Returns:
            Analysis result entry or None if not found
        """
        if self.is_redis_available():
            # Get from Redis
            key = self._get_redis_key("analysis", f"{symbol}:{analysis_type}")
            entry_json = self.redis_client.get(key)
            
            if entry_json:
                return json.loads(entry_json)
            return None
        else:
            # Get from in-memory storage
            key = f"{symbol}:{analysis_type}"
            return self.in_memory_storage["analysis_results"].get(key)
    
    def set_user_preference(self, preference_name: str, preference_value: Any) -> None:
        """
        Set a user preference in memory
        
        Args:
            preference_name: Name of the preference
            preference_value: Value of the preference
        """
        if self.is_redis_available():
            # Store in Redis
            key = self._get_redis_key("preferences", preference_name)
            self.redis_client.set(key, json.dumps(preference_value))
        else:
            # Store in memory
            self.in_memory_storage["user_preferences"][preference_name] = preference_value
    
    def get_user_preference(self, preference_name: str, default_value: Any = None) -> Any:
        """
        Get a user preference from memory
        
        Args:
            preference_name: Name of the preference
            default_value: Default value if preference is not found
            
        Returns:
            Preference value or default value if not found
        """
        if self.is_redis_available():
            # Get from Redis
            key = self._get_redis_key("preferences", preference_name)
            value_json = self.redis_client.get(key)
            
            if value_json:
                return json.loads(value_json)
            return default_value
        else:
            # Get from in-memory storage
            return self.in_memory_storage["user_preferences"].get(preference_name, default_value)
    
    def add_watched_symbol(self, symbol: str) -> None:
        """
        Add a symbol to the watched symbols list
        
        Args:
            symbol: Stock symbol to watch
        """
        if self.is_redis_available():
            # Add to Redis set
            key = self._get_redis_key("watched_symbols")
            self.redis_client.sadd(key, symbol)
        else:
            # Add to in-memory list
            if symbol not in self.in_memory_storage["watched_symbols"]:
                self.in_memory_storage["watched_symbols"].append(symbol)
    
    def remove_watched_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from the watched symbols list
        
        Args:
            symbol: Stock symbol to remove
        """
        if self.is_redis_available():
            # Remove from Redis set
            key = self._get_redis_key("watched_symbols")
            self.redis_client.srem(key, symbol)
        else:
            # Remove from in-memory list
            if symbol in self.in_memory_storage["watched_symbols"]:
                self.in_memory_storage["watched_symbols"].remove(symbol)
    
    def get_watched_symbols(self) -> List[str]:
        """
        Get the list of watched symbols
        
        Returns:
            List of watched stock symbols
        """
        if self.is_redis_available():
            # Get from Redis set
            key = self._get_redis_key("watched_symbols")
            return list(self.redis_client.smembers(key))
        else:
            # Get from in-memory list
            return self.in_memory_storage["watched_symbols"]
    
    def add_alert(self, symbol: str, alert_type: str, threshold: float, 
                 message: str = None, expiration: datetime = None) -> str:
        """
        Add a price alert for a symbol
        
        Args:
            symbol: Stock symbol
            alert_type: Type of alert (price_above, price_below, percent_change, etc.)
            threshold: Alert threshold value
            message: Custom alert message
            expiration: Alert expiration datetime
            
        Returns:
            Alert ID
        """
        alert_id = f"{symbol}:{alert_type}:{int(time.time())}"
        timestamp = datetime.now().isoformat()
        expiration_str = expiration.isoformat() if expiration else None
        
        alert = {
            "id": alert_id,
            "timestamp": timestamp,
            "symbol": symbol,
            "alert_type": alert_type,
            "threshold": threshold,
            "message": message,
            "expiration": expiration_str,
            "triggered": False
        }
        
        if self.is_redis_available():
            # Store in Redis
            key = self._get_redis_key("alerts", alert_id)
            self.redis_client.set(key, json.dumps(alert))
            
            # Add to symbol alerts index
            symbol_key = self._get_redis_key("symbol_alerts", symbol)
            self.redis_client.sadd(symbol_key, alert_id)
            
            # Set expiration if provided
            if expiration:
                expiration_seconds = int((expiration - datetime.now()).total_seconds())
                if expiration_seconds > 0:
                    self.redis_client.expire(key, expiration_seconds)
        else:
            # Store in memory
            self.in_memory_storage["alerts"].append(alert)
        
        return alert_id
    
    def get_alerts(self, symbol: str = None) -> List[Dict]:
        """
        Get alerts from memory
        
        Args:
            symbol: Optional symbol to filter alerts
            
        Returns:
            List of alerts
        """
        if self.is_redis_available():
            if symbol:
                # Get alerts for specific symbol
                symbol_key = self._get_redis_key("symbol_alerts", symbol)
                alert_ids = self.redis_client.smembers(symbol_key)
                
                alerts = []
                for alert_id in alert_ids:
                    key = self._get_redis_key("alerts", alert_id)
                    alert_json = self.redis_client.get(key)
                    if alert_json:
                        alerts.append(json.loads(alert_json))
                
                return alerts
            else:
                # Get all alerts (limited implementation for simplicity)
                # In a real system, you'd want to use a more efficient approach
                watched_symbols = self.get_watched_symbols()
                all_alerts = []
                
                for symbol in watched_symbols:
                    symbol_alerts = self.get_alerts(symbol)
                    all_alerts.extend(symbol_alerts)
                
                return all_alerts
        else:
            # Get from in-memory storage
            if symbol:
                return [alert for alert in self.in_memory_storage["alerts"] if alert["symbol"] == symbol]
            else:
                return self.in_memory_storage["alerts"]
    
    def mark_alert_triggered(self, alert_id: str) -> None:
        """
        Mark an alert as triggered
        
        Args:
            alert_id: ID of the alert to mark as triggered
        """
        if self.is_redis_available():
            # Update in Redis
            key = self._get_redis_key("alerts", alert_id)
            alert_json = self.redis_client.get(key)
            
            if alert_json:
                alert = json.loads(alert_json)
                alert["triggered"] = True
                self.redis_client.set(key, json.dumps(alert))
        else:
            # Update in memory
            for alert in self.in_memory_storage["alerts"]:
                if alert["id"] == alert_id:
                    alert["triggered"] = True
                    break
    
    def remove_alert(self, alert_id: str) -> None:
        """
        Remove an alert from memory
        
        Args:
            alert_id: ID of the alert to remove
        """
        if self.is_redis_available():
            # Get alert to find symbol
            key = self._get_redis_key("alerts", alert_id)
            alert_json = self.redis_client.get(key)
            
            if alert_json:
                alert = json.loads(alert_json)
                symbol = alert["symbol"]
                
                # Remove from symbol alerts index
                symbol_key = self._get_redis_key("symbol_alerts", symbol)
                self.redis_client.srem(symbol_key, alert_id)
                
                # Remove alert
                self.redis_client.delete(key)
        else:
            # Remove from in-memory storage
            self.in_memory_storage["alerts"] = [
                alert for alert in self.in_memory_storage["alerts"]
                if alert["id"] != alert_id
            ]
    
    def clear_expired_data(self) -> None:
        """Clean up expired data from memory"""
        # Redis handles expiration automatically
        if not self.is_redis_available():
            # Clean up expired alerts in memory
            current_time = datetime.now()
            self.in_memory_storage["alerts"] = [
                alert for alert in self.in_memory_storage["alerts"]
                if not alert.get("expiration") or 
                datetime.fromisoformat(alert["expiration"]) > current_time
            ]
