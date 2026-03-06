"""Tests for the ComfyUI HTTP client."""

import pytest
from aioresponses import aioresponses

from comfyui_mcp.client import ComfyUIClient

BASE = "http://127.0.0.1:8188"


@pytest.fixture
def client():
    return ComfyUIClient(base_url=BASE)


@pytest.fixture
def mock_http():
    with aioresponses() as m:
        yield m


class TestQueuePrompt:
    async def test_sends_prompt(self, client: ComfyUIClient, mock_http):
        mock_http.post(
            f"{BASE}/prompt",
            payload={"prompt_id": "abc123", "number": 1, "node_errors": {}},
        )
        result = await client.queue_prompt({"1": {"class_type": "Foo", "inputs": {}}})
        assert result["prompt_id"] == "abc123"
        await client.close()


class TestGetHistory:
    async def test_get_all(self, client: ComfyUIClient, mock_http):
        mock_http.get(f"{BASE}/history?max_items=10", payload={"abc": {}})
        result = await client.get_history(max_items=10)
        assert "abc" in result
        await client.close()

    async def test_get_by_id(self, client: ComfyUIClient, mock_http):
        mock_http.get(f"{BASE}/history/abc123", payload={"abc123": {"outputs": {}}})
        result = await client.get_history(prompt_id="abc123")
        assert "abc123" in result
        await client.close()


class TestGetQueue:
    async def test_returns_queue(self, client: ComfyUIClient, mock_http):
        mock_http.get(f"{BASE}/queue", payload={"queue_running": [], "queue_pending": []})
        result = await client.get_queue()
        assert "queue_running" in result
        await client.close()


class TestGetObjectInfo:
    async def test_all_nodes(self, client: ComfyUIClient, mock_http):
        mock_http.get(f"{BASE}/object_info", payload={"KSampler": {}})
        result = await client.get_object_info()
        assert "KSampler" in result
        await client.close()

    async def test_specific_node(self, client: ComfyUIClient, mock_http):
        mock_http.get(
            f"{BASE}/object_info/KSampler",
            payload={"KSampler": {"input": {"required": {}}}},
        )
        result = await client.get_object_info(node_class="KSampler")
        assert "KSampler" in result
        await client.close()


class TestGetModels:
    async def test_list_folders(self, client: ComfyUIClient, mock_http):
        mock_http.get(f"{BASE}/models", payload=["checkpoints", "loras"])
        result = await client.get_models()
        assert "checkpoints" in result
        await client.close()

    async def test_list_folder_contents(self, client: ComfyUIClient, mock_http):
        mock_http.get(f"{BASE}/models/checkpoints", payload=["model.safetensors"])
        result = await client.get_models(folder="checkpoints")
        assert "model.safetensors" in result
        await client.close()


class TestGetSystemStats:
    async def test_returns_stats(self, client: ComfyUIClient, mock_http):
        mock_http.get(f"{BASE}/system_stats", payload={"system": {"os": "darwin"}})
        result = await client.get_system_stats()
        assert "system" in result
        await client.close()


class TestViewUrl:
    def test_basic_url(self, client: ComfyUIClient):
        url = client.view_url("test.png")
        assert url == f"{BASE}/view?filename=test.png&type=output"

    def test_with_subfolder(self, client: ComfyUIClient):
        url = client.view_url("test.png", subfolder="sub", folder_type="input")
        assert "subfolder=sub" in url
        assert "type=input" in url


class TestInterrupt:
    async def test_sends_interrupt(self, client: ComfyUIClient, mock_http):
        mock_http.post(f"{BASE}/interrupt", payload={})
        await client.interrupt()
        await client.close()


class TestClearQueue:
    async def test_clears(self, client: ComfyUIClient, mock_http):
        mock_http.post(f"{BASE}/queue", payload={})
        await client.clear_queue()
        await client.close()


class TestFreeMemory:
    async def test_free(self, client: ComfyUIClient, mock_http):
        mock_http.post(f"{BASE}/free", payload={})
        await client.free_memory()
        await client.close()
