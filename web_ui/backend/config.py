import os
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    reload: bool = True

class APIConfig(BaseModel):
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    alpha_vantage_api_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    polygon_api_key: Optional[str] = os.getenv("POLYGON_API_KEY")

class WebSocketConfig(BaseModel):
    ping_interval: int = 20  
    ping_timeout: int = 60   
    close_timeout: int = 10 
    max_message_size: int = 1024 * 1024 

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file: str = "logs/finance_agent_web_ui.log"

class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    api: APIConfig = APIConfig()
    websocket: WebSocketConfig = WebSocketConfig()
    logging: LoggingConfig = LoggingConfig()
    typing_speed_chars_per_second: float = 40.0
    thinking_time_seconds: float = 1.5
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
    ]
    
    enable_rate_limiting: bool = False
    rate_limit_requests: int = 60
    rate_limit_window: int = 60  
    enable_streaming: bool = True
    enable_tool_execution_updates: bool = True
    enable_conversation_history: bool = True
config = Config()

def get_config() -> Config:
    return config
