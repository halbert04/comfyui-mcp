"""MCP resources exposing ComfyUI data."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from comfyui_mcp.client import ComfyUIClient


def register(mcp: FastMCP, get_client: Any) -> None:
    """Register MCP resources on the server."""

    @mcp.resource("comfyui://models")
    async def resource_models() -> list:
        """List available model folder types (checkpoints, loras, vae, etc.)."""
        client: ComfyUIClient = get_client()
        return await client.get_models()

    @mcp.resource("comfyui://models/{folder}")
    async def resource_models_folder(folder: str) -> list:
        """List model files in a specific folder."""
        client: ComfyUIClient = get_client()
        return await client.get_models(folder=folder)

    @mcp.resource("comfyui://nodes")
    async def resource_nodes() -> list:
        """List all available node class names."""
        client: ComfyUIClient = get_client()
        info = await client.get_object_info()
        return sorted(info.keys())

    @mcp.resource("comfyui://nodes/{node_class}")
    async def resource_node_info(node_class: str) -> dict:
        """Get the full schema for a specific node class."""
        client: ComfyUIClient = get_client()
        return await client.get_object_info(node_class=node_class)

    @mcp.resource("comfyui://system/stats")
    async def resource_system_stats() -> dict:
        """Get ComfyUI system stats (OS, RAM, VRAM, devices)."""
        client: ComfyUIClient = get_client()
        return await client.get_system_stats()

    @mcp.resource("comfyui://system/queue")
    async def resource_queue() -> dict:
        """Get current queue status."""
        client: ComfyUIClient = get_client()
        return await client.get_queue()

    @mcp.resource("comfyui://history")
    async def resource_history() -> dict:
        """Get recent execution history (last 20 items)."""
        client: ComfyUIClient = get_client()
        return await client.get_history(max_items=20)

    @mcp.resource("comfyui://history/{prompt_id}")
    async def resource_history_item(prompt_id: str) -> dict:
        """Get execution history for a specific prompt."""
        client: ComfyUIClient = get_client()
        return await client.get_history(prompt_id=prompt_id)
