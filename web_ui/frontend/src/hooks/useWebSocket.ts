import { useState, useEffect, useRef, useCallback } from 'react';

interface WebSocketHookOptions {
  url: string;
  onOpen?: (event: WebSocketEventMap['open']) => void;
  onMessage?: (event: WebSocketEventMap['message']) => void;
  onClose?: (event: WebSocketEventMap['close']) => void;
  onError?: (event: WebSocketEventMap['error']) => void;
  reconnectInterval?: number;
  reconnectAttempts?: number;
  autoConnect?: boolean;
}

interface WebSocketHookReturn {
  sendMessage: (data: string | ArrayBufferLike | Blob | ArrayBufferView) => void;
  lastMessage: WebSocketEventMap['message'] | null;
  readyState: number;
  connectionStatus: 'connecting' | 'connected' | 'disconnected';
  connect: () => void;
  disconnect: () => void;
}

const useWebSocket = ({
  url,
  onOpen,
  onMessage,
  onClose,
  onError,
  reconnectInterval = 5000,
  reconnectAttempts = 5,
  autoConnect = true,
}: WebSocketHookOptions): WebSocketHookReturn => {
  const [lastMessage, setLastMessage] = useState<WebSocketEventMap['message'] | null>(null);
  const [readyState, setReadyState] = useState<number>(WebSocket.CLOSED);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);

  const connect = useCallback(() => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) return;
    
    try {
      setConnectionStatus('connecting');
      const ws = new WebSocket(url);
      
      ws.onopen = (event) => {
        setReadyState(WebSocket.OPEN);
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0;
        if (onOpen) onOpen(event);
      };
      
      ws.onmessage = (event) => {
        setLastMessage(event);
        if (onMessage) onMessage(event);
      };
      
      ws.onclose = (event) => {
        setReadyState(WebSocket.CLOSED);
        setConnectionStatus('disconnected');
        websocketRef.current = null;
        
        if (onClose) onClose(event);
        
        // Attempt to reconnect if not closed cleanly and we haven't exceeded max attempts
        if (!event.wasClean && reconnectAttemptsRef.current < reconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };
      
      ws.onerror = (event) => {
        setConnectionStatus('disconnected');
        if (onError) onError(event);
      };
      
      websocketRef.current = ws;
      
    } catch (error) {
      console.error('WebSocket connection error:', error);
      setConnectionStatus('disconnected');
    }
  }, [url, onOpen, onMessage, onClose, onError, reconnectInterval, reconnectAttempts]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
    
    setConnectionStatus('disconnected');
  }, []);

  const sendMessage = useCallback((data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      websocketRef.current.send(data);
      return true;
    }
    return false;
  }, []);

  // Connect on mount if autoConnect is true
  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    
    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Update readyState when websocket reference changes
  useEffect(() => {
    if (websocketRef.current) {
      setReadyState(websocketRef.current.readyState);
    } else {
      setReadyState(WebSocket.CLOSED);
    }
  }, [websocketRef.current]);

  return {
    sendMessage,
    lastMessage,
    readyState,
    connectionStatus,
    connect,
    disconnect,
  };
};

export default useWebSocket;
