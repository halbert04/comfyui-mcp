"""Model discovery and memory management tools."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from comfyui_mcp.client import ComfyUIClient


def register(mcp: FastMCP, get_client: Any) -> None:
    """Register model tools on the MCP server."""

    @mcp.tool()
    async def list_models(folder: str = "") -> list | dict:
        """List available models in ComfyUI.

        Args:
            folder: Model folder to list. Empty returns available folder names
                (e.g. "checkpoints", "loras", "vae"). Provide a folder name
                to list model files within it.
        """
        client: ComfyUIClient = get_client()
        if folder:
            return await client.get_models(folder=folder)
        return await client.get_models()

    @mcp.tool()
    async def list_samplers_and_schedulers() -> dict:
        """List all available sampler algorithms and noise schedulers.

        Returns a dict with 'samplers' and 'schedulers' lists.
        """
        client: ComfyUIClient = get_client()
        info = await client.get_object_info(node_class="KSampler")
        node = info.get("KSampler", info)
        required = node.get("input", {}).get("required", {})
        samplers = required.get("sampler_name", [[]])[0]
        schedulers = required.get("scheduler", [[]])[0]
        return {"samplers": list(samplers), "schedulers": list(schedulers)}

    @mcp.tool()
    async def free_memory(unload_models: bool = True, free_memory: bool = True) -> dict:
        """Free ComfyUI VRAM and RAM.

        Args:
            unload_models: Whether to unload loaded models from memory.
            free_memory: Whether to free cached memory.
        """
        client: ComfyUIClient = get_client()
        await client.free_memory(unload_models=unload_models, free_memory=free_memory)
        return {"status": "ok", "message": "Memory freed"}
