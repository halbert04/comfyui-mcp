"""Tests for MCP tools with mocked client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from comfyui_mcp.client import ComfyUIClient


def _make_mock_client(base_url: str = "http://127.0.0.1:8188") -> AsyncMock:
    """Create a mock ComfyUIClient."""
    mock = AsyncMock(spec=ComfyUIClient)
    mock.base_url = base_url

    def _view_url(filename, subfolder="", folder_type="output"):
        return f"{base_url}/view?filename={filename}&type={folder_type}"

    mock.view_url = _view_url
    return mock


class TestSystemTools:
    async def test_get_queue(self):
        """System tools register and delegate to client."""
        from comfyui_mcp.tools import system
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.get_queue.return_value = {"queue_running": [], "queue_pending": []}

        mcp = FastMCP("test")
        system.register(mcp, lambda: mock_client)

        # Use call_tool to invoke the registered tool
        result = await mcp.call_tool("get_queue", {})
        mock_client.get_queue.assert_awaited_once()

    async def test_cancel_job(self):
        from comfyui_mcp.tools import system
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.interrupt.return_value = {}

        mcp = FastMCP("test")
        system.register(mcp, lambda: mock_client)

        result = await mcp.call_tool("cancel_job", {})
        mock_client.interrupt.assert_awaited_once()

    async def test_clear_queue(self):
        from comfyui_mcp.tools import system
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.clear_queue.return_value = {}

        mcp = FastMCP("test")
        system.register(mcp, lambda: mock_client)

        result = await mcp.call_tool("clear_queue", {})
        mock_client.clear_queue.assert_awaited_once()

    async def test_get_system_stats(self):
        from comfyui_mcp.tools import system
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.get_system_stats.return_value = {"system": {"os": "posix"}}

        mcp = FastMCP("test")
        system.register(mcp, lambda: mock_client)

        result = await mcp.call_tool("get_system_stats", {})
        mock_client.get_system_stats.assert_awaited_once()

    async def test_get_job_status_completed(self):
        from comfyui_mcp.tools import system
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.get_history.return_value = {
            "abc": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {},
            }
        }

        mcp = FastMCP("test")
        system.register(mcp, lambda: mock_client)

        result = await mcp.call_tool("get_job_status", {"prompt_id": "abc"})
        mock_client.get_history.assert_awaited_once_with(prompt_id="abc")

    async def test_get_job_status_not_found(self):
        from comfyui_mcp.tools import system
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.get_history.return_value = {}
        mock_client.get_queue.return_value = {"queue_running": [], "queue_pending": []}

        mcp = FastMCP("test")
        system.register(mcp, lambda: mock_client)

        result = await mcp.call_tool("get_job_status", {"prompt_id": "xyz"})
        mock_client.get_history.assert_awaited_once()
        mock_client.get_queue.assert_awaited_once()

    async def test_get_result_images(self):
        from comfyui_mcp.tools import system
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.get_history.return_value = {
            "abc": {
                "outputs": {
                    "8": {
                        "images": [
                            {"filename": "out.png", "subfolder": "", "type": "output"}
                        ]
                    }
                }
            }
        }

        mcp = FastMCP("test")
        system.register(mcp, lambda: mock_client)

        result = await mcp.call_tool("get_result_images", {"prompt_id": "abc"})
        mock_client.get_history.assert_awaited_once_with(prompt_id="abc")


class TestModelTools:
    async def test_list_models_folders(self):
        from comfyui_mcp.tools import models
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.get_models.return_value = ["checkpoints", "loras", "vae"]

        mcp = FastMCP("test")
        models.register(mcp, lambda: mock_client)

        result = await mcp.call_tool("list_models", {"folder": ""})
        mock_client.get_models.assert_awaited_once()

    async def test_list_models_specific_folder(self):
        from comfyui_mcp.tools import models
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.get_models.return_value = ["model.safetensors"]

        mcp = FastMCP("test")
        models.register(mcp, lambda: mock_client)

        result = await mcp.call_tool("list_models", {"folder": "checkpoints"})
        mock_client.get_models.assert_awaited_once_with(folder="checkpoints")

    async def test_list_samplers_and_schedulers(self):
        from comfyui_mcp.tools import models
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.get_object_info.return_value = {
            "KSampler": {
                "input": {
                    "required": {
                        "sampler_name": [["euler", "dpmpp_2m"]],
                        "scheduler": [["normal", "karras"]],
                    }
                }
            }
        }

        mcp = FastMCP("test")
        models.register(mcp, lambda: mock_client)

        result = await mcp.call_tool("list_samplers_and_schedulers", {})
        mock_client.get_object_info.assert_awaited_once_with(node_class="KSampler")

    async def test_free_memory(self):
        from comfyui_mcp.tools import models
        from fastmcp import FastMCP

        mock_client = _make_mock_client()
        mock_client.free_memory.return_value = {}

        mcp = FastMCP("test")
        models.register(mcp, lambda: mock_client)

        result = await mcp.call_tool("free_memory", {})
        mock_client.free_memory.assert_awaited_once()
