"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    comfyui_url: str = field(default_factory=lambda: os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"))
    comfyui_timeout: float = field(default_factory=lambda: float(os.environ.get("COMFYUI_TIMEOUT", "300")))
    comfyui_poll_interval: float = field(default_factory=lambda: float(os.environ.get("COMFYUI_POLL_INTERVAL", "1.0")))
    mcp_transport: str = field(default_factory=lambda: os.environ.get("COMFYUI_MCP_TRANSPORT", "stdio"))
    mcp_host: str = field(default_factory=lambda: os.environ.get("COMFYUI_MCP_HOST", "127.0.0.1"))
    mcp_port: int = field(default_factory=lambda: int(os.environ.get("COMFYUI_MCP_PORT", "8200")))
    comfy_api_key: str = field(default_factory=lambda: os.environ.get("COMFY_API_KEY", ""))


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
