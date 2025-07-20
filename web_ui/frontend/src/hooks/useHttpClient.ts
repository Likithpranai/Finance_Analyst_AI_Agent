import { useState, useCallback } from 'react';

interface UseHttpClientOptions {
  baseUrl: string;
  defaultHeaders?: Record<string, string>;
}

interface QueryRequest {
  query: string;
  session_id?: string;
}

interface QueryResponse {
  response: any;
  session_id: string;
  timestamp: string;
}

export const useHttpClient = (options: UseHttpClientOptions) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const sendQuery = useCallback(async (query: string, sessionId?: string): Promise<QueryResponse> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${options.baseUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...options.defaultHeaders
        },
        body: JSON.stringify({
          query,
          session_id: sessionId
        } as QueryRequest)
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Error ${response.status}: ${response.statusText}`);
      }
      
      const data: QueryResponse = await response.json();
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send query';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [options.baseUrl, options.defaultHeaders]);
  
  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch(`${options.baseUrl}/health`, {
        method: 'GET',
        headers: options.defaultHeaders
      });
      
      if (!response.ok) {
        return false;
      }
      
      const data = await response.json();
      return data.status === 'ok';
    } catch (err) {
      return false;
    }
  }, [options.baseUrl, options.defaultHeaders]);
  
  return {
    sendQuery,
    checkHealth,
    isLoading,
    error
  };
};
