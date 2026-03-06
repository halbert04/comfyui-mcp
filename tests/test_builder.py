"""Tests for workflow builder tools — tested via _impl functions."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from comfyui_mcp.client import ComfyUIClient
from comfyui_mcp.node_cache import NodeCache
from comfyui_mcp.tools import builder

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


class TestAutoConnect:
    async def test_single_candidate_connects(self):
        _, node_cache = _make_deps()
        nodes = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model.safetensors"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a cat"},
            },
        }
        auto, unconnected = await builder._auto_connect(
            nodes, "2", {"text": "a cat"}, node_cache
        )
        assert "clip" in auto
        assert auto["clip"]["from_node"] == "1"
        assert auto["clip"]["from_output"] == 1

    async def test_ambiguous_not_connected(self):
        _, node_cache = _make_deps()
        nodes = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model.safetensors"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "positive", "clip": ["1", 1]},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "negative", "clip": ["1", 1]},
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1},
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {"seed": 42, "steps": 20, "cfg": 8.0, "denoise": 1.0},
            },
        }
        auto, unconnected = await builder._auto_connect(
            nodes, "5",
            {"seed": 42, "steps": 20, "cfg": 8.0, "denoise": 1.0},
            node_cache,
        )
        assert "model" in auto
        unconnected_names = [u["input"] for u in unconnected]
        assert "positive" in unconnected_names
        assert "negative" in unconnected_names
        assert "latent_image" in auto

    async def test_primitive_inputs_not_auto_connected(self):
        _, node_cache = _make_deps()
        nodes = {
            "1": {
                "class_type": "KSampler",
                "inputs": {},
            },
        }
        auto, unconnected = await builder._auto_connect(
            nodes, "1", {}, node_cache
        )
        assert len(auto) == 0
        unconnected_types = {u["type"] for u in unconnected}
        assert "MODEL" in unconnected_types or "CONDITIONING" in unconnected_types


class TestCreateWorkflow:
    async def test_create_empty(self):
        ctx = MockContext()
        mock_client, _ = _make_deps()

        result = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        assert "workflow_id" in result
        assert result["node_count"] == 0

    async def test_create_from_template(self):
        ctx = MockContext()
        mock_client, _ = _make_deps()

        result = await builder.create_workflow_impl(
            lambda: mock_client,
            template="txt2img",
            overrides=json.dumps({"prompt": "a cat", "checkpoint": "model.safetensors"}),
            ctx=ctx,
        )
        assert "workflow_id" in result
        assert result["node_count"] > 0

    async def test_unknown_template(self):
        ctx = MockContext()
        mock_client, _ = _make_deps()

        result = await builder.create_workflow_impl(
            lambda: mock_client, template="nonexistent", ctx=ctx
        )
        assert "error" in result

    async def test_no_context(self):
        mock_client, _ = _make_deps()
        result = await builder.create_workflow_impl(lambda: mock_client)
        assert "error" in result


class TestAddNode:
    async def test_basic_add(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        # Create workflow first
        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        result = await builder.add_node_impl(
            lambda: node_cache, wf_id, "CheckpointLoaderSimple",
            json.dumps({"ckpt_name": "model.safetensors"}), ctx
        )
        assert result["node_id"] == "1"
        assert result["class_type"] == "CheckpointLoaderSimple"
        assert result["outputs"] == ["MODEL", "CLIP", "VAE"]

    async def test_auto_connection(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        # Add checkpoint
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CheckpointLoaderSimple",
            json.dumps({"ckpt_name": "model.safetensors"}), ctx
        )

        # Add CLIPTextEncode — should auto-connect CLIP
        result = await builder.add_node_impl(
            lambda: node_cache, wf_id, "CLIPTextEncode",
            json.dumps({"text": "a cat"}), ctx
        )
        assert "clip" in result["auto_connected"]
        assert result["auto_connected"]["clip"]["from_node"] == "1"
        assert result["auto_connected"]["clip"]["from_output"] == 1

    async def test_unknown_class_type(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        result = await builder.add_node_impl(
            lambda: node_cache, wf_id, "NonExistentNode", "{}", ctx
        )
        assert "error" in result


class TestSetInputs:
    async def test_update_literal(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CLIPTextEncode",
            json.dumps({"text": "a cat"}), ctx
        )

        result = await builder.set_inputs_impl(
            wf_id, "1", json.dumps({"text": "a dog"}), ctx
        )
        assert result["inputs"]["text"] == "a dog"

    async def test_update_connection(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CLIPTextEncode",
            json.dumps({"text": "a cat"}), ctx
        )

        result = await builder.set_inputs_impl(
            wf_id, "1", json.dumps({"clip": ["2", 1]}), ctx
        )
        assert result["inputs"]["clip"] == ["2", 1]

    async def test_node_not_found(self):
        ctx = MockContext()
        mock_client, _ = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        result = await builder.set_inputs_impl(
            wf_id, "99", json.dumps({"text": "test"}), ctx
        )
        assert "error" in result


class TestRemoveNode:
    async def test_remove_and_clean_references(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        # Add checkpoint (node 1)
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CheckpointLoaderSimple",
            json.dumps({"ckpt_name": "model.safetensors"}), ctx
        )

        # Add CLIPTextEncode that links to checkpoint (node 2)
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CLIPTextEncode",
            json.dumps({"text": "test", "clip": ["1", 1]}), ctx
        )

        # Remove the checkpoint
        result = await builder.remove_node_impl(wf_id, "1", ctx)
        assert result["removed"] == "1"
        assert result["removed_class"] == "CheckpointLoaderSimple"
        assert len(result["broken_connections"]) == 1
        assert result["broken_connections"][0]["input"] == "clip"

    async def test_remove_nonexistent(self):
        ctx = MockContext()
        mock_client, _ = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        result = await builder.remove_node_impl(wf_id, "99", ctx)
        assert "error" in result


class TestGetWorkflow:
    async def test_enriched_output(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CheckpointLoaderSimple",
            json.dumps({"ckpt_name": "model.safetensors"}), ctx
        )

        result = await builder.get_workflow_impl(lambda: node_cache, wf_id, ctx)
        assert result["node_count"] == 1
        assert "1" in result["nodes"]
        assert result["nodes"]["1"]["outputs"] == ["MODEL", "CLIP", "VAE"]
        assert result["has_output_node"] is False

    async def test_nonexistent_workflow(self):
        ctx = MockContext()
        _, node_cache = _make_deps()

        result = await builder.get_workflow_impl(
            lambda: node_cache, "wf_nonexistent", ctx
        )
        assert "error" in result


class TestValidateWorkflow:
    async def _build_valid_workflow(self, ctx, mock_client, node_cache):
        """Helper to build a full valid workflow."""
        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

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
        return wf_id

    async def test_valid_workflow(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()
        wf_id = await self._build_valid_workflow(ctx, mock_client, node_cache)

        result = await builder.validate_workflow_impl(
            lambda: node_cache, wf_id, ctx
        )
        assert result["valid"] is True

    async def test_missing_output_node(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CheckpointLoaderSimple",
            json.dumps({"ckpt_name": "model.safetensors"}), ctx
        )

        result = await builder.validate_workflow_impl(
            lambda: node_cache, wf_id, ctx
        )
        assert result["valid"] is False
        assert any("output node" in e for e in result["errors"])

    async def test_missing_required_inputs(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        await builder.add_node_impl(
            lambda: node_cache, wf_id, "KSampler",
            json.dumps({"seed": 42}), ctx
        )
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "SaveImage",
            json.dumps({"filename_prefix": "test"}), ctx
        )

        result = await builder.validate_workflow_impl(
            lambda: node_cache, wf_id, ctx
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    async def test_type_mismatch(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        # EmptyLatentImage (node 1, outputs LATENT)
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "EmptyLatentImage",
            json.dumps({"width": 512, "height": 512, "batch_size": 1}), ctx
        )
        # VAEDecode with vae connected to LATENT (expects VAE)
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "VAEDecode",
            json.dumps({"samples": ["1", 0], "vae": ["1", 0]}), ctx
        )
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "SaveImage",
            json.dumps({"images": ["2", 0], "filename_prefix": "test"}), ctx
        )

        result = await builder.validate_workflow_impl(
            lambda: node_cache, wf_id, ctx
        )
        assert result["valid"] is False
        assert any("expects VAE" in e for e in result["errors"])

    async def test_reference_nonexistent_node(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        await builder.add_node_impl(
            lambda: node_cache, wf_id, "CLIPTextEncode",
            json.dumps({"text": "test", "clip": ["99", 0]}), ctx
        )
        await builder.add_node_impl(
            lambda: node_cache, wf_id, "SaveImage",
            json.dumps({"images": ["1", 0], "filename_prefix": "test"}), ctx
        )

        result = await builder.validate_workflow_impl(
            lambda: node_cache, wf_id, ctx
        )
        assert result["valid"] is False
        assert any("non-existent" in e for e in result["errors"])


class TestExecuteWorkflow:
    async def test_execute_and_wait(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        await builder.add_node_impl(
            lambda: node_cache, wf_id, "SaveImage",
            json.dumps({"images": ["2", 0], "filename_prefix": "test"}), ctx
        )

        mock_client.queue_prompt.return_value = {"prompt_id": "abc123"}
        mock_client.get_history.return_value = {
            "abc123": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "1": {
                        "images": [
                            {"filename": "out.png", "subfolder": "", "type": "output"}
                        ]
                    }
                },
            }
        }

        result = await builder.execute_workflow_impl(
            lambda: mock_client, wf_id, wait=True, ctx=ctx
        )
        assert result["status"] == "success"
        assert result["workflow_id"] == wf_id
        mock_client.queue_prompt.assert_awaited_once()

    async def test_execute_no_wait(self):
        ctx = MockContext()
        mock_client, node_cache = _make_deps()

        wf = await builder.create_workflow_impl(
            lambda: mock_client, name="test", ctx=ctx
        )
        wf_id = wf["workflow_id"]

        await builder.add_node_impl(
            lambda: node_cache, wf_id, "SaveImage",
            json.dumps({"images": ["2", 0], "filename_prefix": "test"}), ctx
        )

        mock_client.queue_prompt.return_value = {
            "prompt_id": "abc456",
            "number": 1,
        }

        result = await builder.execute_workflow_impl(
            lambda: mock_client, wf_id, wait=False, ctx=ctx
        )
        assert result["status"] == "queued"
        assert result["workflow_id"] == wf_id

    async def test_execute_nonexistent_workflow(self):
        ctx = MockContext()
        mock_client, _ = _make_deps()

        result = await builder.execute_workflow_impl(
            lambda: mock_client, "wf_nonexistent", ctx=ctx
        )
        assert "error" in result
