"""Tests for discovery tools: search_nodes, get_node_schema, suggest_next."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from comfyui_mcp.client import ComfyUIClient
from comfyui_mcp.node_cache import NodeCache
from comfyui_mcp.tools import builder, discovery

from tests.test_node_cache import SAMPLE_NODES


class MockContext:
    """Mock FastMCP Context with in-memory state."""

    def __init__(self):
        self._state = {}

    async def set_state(self, key, value, **kwargs):
        self._state[key] = value

    async def get_state(self, key):
        return self._state.get(key)

    async def delete_state(self, key):
        self._state.pop(key, None)


def _make_deps():
    """Create mock client and node cache."""
    mock_client = AsyncMock(spec=ComfyUIClient)
    mock_client.base_url = "http://127.0.0.1:8188"
    mock_client.get_object_info.return_value = dict(SAMPLE_NODES)
    mock_client.get_models.return_value = ["model.safetensors"]

    def _view_url(filename, subfolder="", folder_type="output"):
        return f"http://127.0.0.1:8188/view?filename={filename}&type={folder_type}"

    mock_client.view_url = _view_url
    node_cache = NodeCache(mock_client, ttl=300.0)
    return mock_client, node_cache


class TestSearchNodes:
    async def test_search_by_query(self):
        _, node_cache = _make_deps()
        results = await discovery.search_nodes_impl(node_cache, query="checkpoint")
        names = [r["name"] for r in results]
        assert "CheckpointLoaderSimple" in names

    async def test_search_by_category(self):
        _, node_cache = _make_deps()
        results = await discovery.search_nodes_impl(node_cache, category="sampling")
        names = [r["name"] for r in results]
        assert "KSampler" in names

    async def test_search_by_input_type(self):
        _, node_cache = _make_deps()
        results = await discovery.search_nodes_impl(node_cache, input_type="MODEL")
        names = [r["name"] for r in results]
        assert "KSampler" in names
        assert "LoraLoader" in names

    async def test_search_by_output_type(self):
        _, node_cache = _make_deps()
        results = await discovery.search_nodes_impl(node_cache, output_type="IMAGE")
        names = [r["name"] for r in results]
        assert "VAEDecode" in names

    async def test_search_combined(self):
        _, node_cache = _make_deps()
        results = await discovery.search_nodes_impl(
            node_cache, query="load", category="loaders"
        )
        names = [r["name"] for r in results]
        assert "CheckpointLoaderSimple" in names
        assert "LoraLoader" in names


class TestGetNodeSchema:
    async def test_existing_node(self):
        _, node_cache = _make_deps()
        data = await discovery.get_node_schema_impl(node_cache, "KSampler")
        assert data["name"] == "KSampler"
        assert "inputs" in data
        assert "outputs" in data
        assert "required" in data["inputs"]
        assert "optional" in data["inputs"]
        # Check that model input is typed
        assert data["inputs"]["required"]["model"]["type"] == "MODEL"
        # Check COMBO type for sampler_name
        assert data["inputs"]["required"]["sampler_name"]["type"] == "COMBO"
        assert "options" in data["inputs"]["required"]["sampler_name"]
        # Check output structure
        assert len(data["outputs"]) == 1
        assert data["outputs"][0]["type"] == "LATENT"

    async def test_nonexistent_node(self):
        _, node_cache = _make_deps()
        data = await discovery.get_node_schema_impl(node_cache, "DoesNotExist")
        assert "error" in data

    async def test_checkpoint_schema(self):
        _, node_cache = _make_deps()
        data = await discovery.get_node_schema_impl(
            node_cache, "CheckpointLoaderSimple"
        )
        assert data["name"] == "CheckpointLoaderSimple"
        assert data["category"] == "loaders"
        assert len(data["outputs"]) == 3
        output_types = [o["type"] for o in data["outputs"]]
        assert output_types == ["MODEL", "CLIP", "VAE"]


class TestSuggestNext:
    async def test_empty_workflow(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        # Create an empty workflow using builder _impl
        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        data = await discovery.suggest_next_impl(
            lambda: node_cache, wf_id, ctx
        )
        assert data["missing_output_node"] is True
        assert data["ready_to_execute"] is False
        assert len(data["suggestions"]) > 0

    async def test_workflow_with_dangling_outputs(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        # Add only a checkpoint — its outputs are unused
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CheckpointLoaderSimple",
            json.dumps({"ckpt_name": "model.safetensors"}), ctx
        )

        data = await discovery.suggest_next_impl(
            lambda: node_cache, wf_id, ctx
        )
        assert data["missing_output_node"] is True
        assert len(data["unused_outputs"]) > 0
        # CheckpointLoaderSimple has 3 outputs (MODEL, CLIP, VAE)
        assert len(data["unused_outputs"]) == 3

    async def test_complete_workflow(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        # Build a complete workflow
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CheckpointLoaderSimple",
            json.dumps({"ckpt_name": "model.safetensors"}), ctx
        )
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CLIPTextEncode",
            json.dumps({"text": "a cat", "clip": ["1", 1]}), ctx
        )
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CLIPTextEncode",
            json.dumps({"text": "", "clip": ["1", 1]}), ctx
        )
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "EmptyLatentImage",
            json.dumps({"width": 512, "height": 512, "batch_size": 1}), ctx
        )
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "KSampler",
            json.dumps({
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["4", 0], "seed": 42, "steps": 20, "cfg": 8.0,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            }), ctx
        )
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "VAEDecode",
            json.dumps({"samples": ["5", 0], "vae": ["1", 2]}), ctx
        )
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "SaveImage",
            json.dumps({"images": ["6", 0], "filename_prefix": "test"}), ctx
        )

        data = await discovery.suggest_next_impl(
            lambda: node_cache, wf_id, ctx
        )
        assert data["missing_output_node"] is False
        assert data["ready_to_execute"] is True

    async def test_no_context(self):
        _, node_cache = _make_deps()
        data = await discovery.suggest_next_impl(
            lambda: node_cache, "wf_test"
        )
        assert "error" in data

    async def test_nonexistent_workflow(self):
        ctx = MockContext()
        _, node_cache = _make_deps()
        data = await discovery.suggest_next_impl(
            lambda: node_cache, "wf_nonexistent", ctx
        )
        assert "error" in data
