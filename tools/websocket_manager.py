"""
WebSocket Manager for Finance Analyst AI Agent

This module provides WebSocket streaming capabilities for real-time financial data.
It handles connections to various data providers and manages streaming subscriptions.
"""

import os
import json
import time
import threading
import websocket
from typing import Dict, List, Any, Callable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")


class WebSocketManager:
    """Manager for WebSocket connections to financial data providers"""
    
    def __init__(self):
        """Initialize WebSocket manager"""
        self.active_connections = {}
        self.callbacks = {}
        self.running = False
    
    def start_polygon_stock_stream(self, symbols: List[str], callback: Callable):
        """
        Start a WebSocket stream for real-time stock updates from Polygon.io
        
        Args:
            symbols: List of stock symbols to stream
            callback: Function to call with each update
            
        Returns:
            Connection ID for the stream
        """
        if not POLYGON_API_KEY:
            raise ValueError("Polygon.io API key not found")
        
        # Generate a unique connection ID
        connection_id = f"polygon_stocks_{int(time.time())}"
        
        def on_message(ws_client, message):
            """Handle incoming WebSocket messages"""
            try:
                # Parse the message
                msg_dict = json.loads(message)
                # Call the callback function with the parsed message
                callback(msg_dict)
            except Exception as e:
                print(f"Error processing WebSocket message: {str(e)}")
        
        def on_error(ws_client, error):
            """Handle WebSocket errors"""
            print(f"WebSocket error: {str(error)}")
        
        def on_close(ws_client, close_status_code, close_msg):
            """Handle WebSocket connection close"""
            print(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
            # Remove from active connections
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
        
        def on_open(ws_client):
            """Handle WebSocket connection open"""
            print(f"WebSocket connection opened for {connection_id}")
            # Authenticate
            auth_message = {"action": "auth", "params": POLYGON_API_KEY}
            ws_client.send(json.dumps(auth_message))
            
            # Subscribe to trades for the specified symbols
            symbols_str = ",".join([f"T.{symbol}" for symbol in symbols])
            subscribe_message = {"action": "subscribe", "params": symbols_str}
            ws_client.send(json.dumps(subscribe_message))
        
        # Create WebSocket client
        ws = websocket.WebSocketApp(
            "wss://socket.polygon.io/stocks",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Store the connection
        self.active_connections[connection_id] = ws
        self.callbacks[connection_id] = callback
        
        # Start the WebSocket connection in a separate thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        return connection_id
    
    def start_polygon_crypto_stream(self, symbols: List[str], callback: Callable):
        """
        Start a WebSocket stream for real-time cryptocurrency updates from Polygon.io
        
        Args:
            symbols: List of crypto symbols to stream (e.g., ["BTC-USD"])
            callback: Function to call with each update
            
        Returns:
            Connection ID for the stream
        """
        if not POLYGON_API_KEY:
            raise ValueError("Polygon.io API key not found")
        
        # Generate a unique connection ID
        connection_id = f"polygon_crypto_{int(time.time())}"
        
        def on_message(ws_client, message):
            """Handle incoming WebSocket messages"""
            try:
                # Parse the message
                msg_dict = json.loads(message)
                # Call the callback function with the parsed message
                callback(msg_dict)
            except Exception as e:
                print(f"Error processing WebSocket message: {str(e)}")
        
        def on_error(ws_client, error):
            """Handle WebSocket errors"""
            print(f"WebSocket error: {str(error)}")
        
        def on_close(ws_client, close_status_code, close_msg):
            """Handle WebSocket connection close"""
            print(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
            # Remove from active connections
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
        
        def on_open(ws_client):
            """Handle WebSocket connection open"""
            print(f"WebSocket connection opened for {connection_id}")
            # Authenticate
            auth_message = {"action": "auth", "params": POLYGON_API_KEY}
            ws_client.send(json.dumps(auth_message))
            
            # Subscribe to trades for the specified symbols
            symbols_str = ",".join([f"XT.{symbol}" for symbol in symbols])
            subscribe_message = {"action": "subscribe", "params": symbols_str}
            ws_client.send(json.dumps(subscribe_message))
        
        # Create WebSocket client
        ws = websocket.WebSocketApp(
            "wss://socket.polygon.io/crypto",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Store the connection
        self.active_connections[connection_id] = ws
        self.callbacks[connection_id] = callback
        
        # Start the WebSocket connection in a separate thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        return connection_id
    
    def start_polygon_forex_stream(self, pairs: List[str], callback: Callable):
        """
        Start a WebSocket stream for real-time forex updates from Polygon.io
        
        Args:
            pairs: List of forex pairs to stream (e.g., ["EUR-USD"])
            callback: Function to call with each update
            
        Returns:
            Connection ID for the stream
        """
        if not POLYGON_API_KEY:
            raise ValueError("Polygon.io API key not found")
        
        # Generate a unique connection ID
        connection_id = f"polygon_forex_{int(time.time())}"
        
        def on_message(ws_client, message):
            """Handle incoming WebSocket messages"""
            try:
                # Parse the message
                msg_dict = json.loads(message)
                # Call the callback function with the parsed message
                callback(msg_dict)
            except Exception as e:
                print(f"Error processing WebSocket message: {str(e)}")
        
        def on_error(ws_client, error):
            """Handle WebSocket errors"""
            print(f"WebSocket error: {str(error)}")
        
        def on_close(ws_client, close_status_code, close_msg):
            """Handle WebSocket connection close"""
            print(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
            # Remove from active connections
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
        
        def on_open(ws_client):
            """Handle WebSocket connection open"""
            print(f"WebSocket connection opened for {connection_id}")
            # Authenticate
            auth_message = {"action": "auth", "params": POLYGON_API_KEY}
            ws_client.send(json.dumps(auth_message))
            
            # Subscribe to trades for the specified pairs
            pairs_str = ",".join([f"C.{pair}" for pair in pairs])
            subscribe_message = {"action": "subscribe", "params": pairs_str}
            ws_client.send(json.dumps(subscribe_message))
        
        # Create WebSocket client
        ws = websocket.WebSocketApp(
            "wss://socket.polygon.io/forex",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Store the connection
        self.active_connections[connection_id] = ws
        self.callbacks[connection_id] = callback
        
        # Start the WebSocket connection in a separate thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        return connection_id
    
    def stop_stream(self, connection_id: str):
        """
        Stop a WebSocket stream
        
        Args:
            connection_id: ID of the connection to stop
            
        Returns:
            True if successful, False otherwise
        """
        if connection_id in self.active_connections:
            ws = self.active_connections[connection_id]
            ws.close()
            # Wait for the connection to close
            start_time = time.time()
            while connection_id in self.active_connections and time.time() - start_time < 5:
                time.sleep(0.1)
            return connection_id not in self.active_connections
        return False
    
    def stop_all_streams(self):
        """
        Stop all active WebSocket streams
        
        Returns:
            Number of streams stopped
        """
        count = 0
        connection_ids = list(self.active_connections.keys())
        for connection_id in connection_ids:
            if self.stop_stream(connection_id):
                count += 1
        return count
    
    def get_active_streams(self):
        """
        Get information about active streams
        
        Returns:
            Dictionary with active stream information
        """
        return {
            "active_streams": len(self.active_connections),
            "connection_ids": list(self.active_connections.keys())
        }


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


# Example usage
if __name__ == "__main__":
    # Example callback function
    def print_update(message):
        print(f"Received update: {json.dumps(message)}")
    
    # Start a stream
    try:
        print("Starting stock stream for AAPL, MSFT, GOOGL...")
        connection_id = websocket_manager.start_polygon_stock_stream(
            ["AAPL", "MSFT", "GOOGL"],
            print_update
        )
        
        print(f"Stream started with connection ID: {connection_id}")
        print("Waiting for updates (press Ctrl+C to stop)...")
        
        # Wait for updates
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping stream...")
            websocket_manager.stop_stream(connection_id)
            print("Stream stopped")
    
    except Exception as e:
        print(f"Error: {str(e)}")
