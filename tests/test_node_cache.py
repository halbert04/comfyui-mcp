"""Tests for NodeCache."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from comfyui_mcp.client import ComfyUIClient
from comfyui_mcp.node_cache import NodeCache

SAMPLE_NODES = {
    "CheckpointLoaderSimple": {
        "display_name": "Load Checkpoint",
        "category": "loaders",
        "description": "Loads a checkpoint model",
        "input": {
            "required": {
                "ckpt_name": [["model_a.safetensors", "model_b.safetensors"]],
            },
            "optional": {},
        },
        "output": ["MODEL", "CLIP", "VAE"],
        "output_name": ["MODEL", "CLIP", "VAE"],
        "output_node": False,
    },
    "CLIPTextEncode": {
        "display_name": "CLIP Text Encode (Prompt)",
        "category": "conditioning",
        "description": "Encodes text using CLIP",
        "input": {
            "required": {
                "text": ["STRING", {"multiline": True}],
                "clip": ["CLIP"],
            },
            "optional": {},
        },
        "output": ["CONDITIONING"],
        "output_name": ["CONDITIONING"],
        "output_node": False,
    },
    "KSampler": {
        "display_name": "KSampler",
        "category": "sampling",
        "description": "Samples from a model",
        "input": {
            "required": {
                "model": ["MODEL"],
                "seed": ["INT", {"default": 0, "min": 0, "max": 2**63}],
                "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                "cfg": ["FLOAT", {"default": 8.0}],
                "sampler_name": [["euler", "dpmpp_2m"]],
                "scheduler": [["normal", "karras"]],
                "denoise": ["FLOAT", {"default": 1.0}],
                "positive": ["CONDITIONING"],
                "negative": ["CONDITIONING"],
                "latent_image": ["LATENT"],
            },
            "optional": {},
        },
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "output_node": False,
    },
    "SaveImage": {
        "display_name": "Save Image",
        "category": "image",
        "description": "Saves an image",
        "input": {
            "required": {
                "images": ["IMAGE"],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
            },
            "optional": {},
        },
        "output": [],
        "output_name": [],
        "output_node": True,
    },
    "VAEDecode": {
        "display_name": "VAE Decode",
        "category": "latent",
        "description": "Decodes latent to image",
        "input": {
            "required": {
                "samples": ["LATENT"],
                "vae": ["VAE"],
            },
            "optional": {},
        },
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "output_node": False,
    },
    "LoraLoader": {
        "display_name": "Load LoRA",
        "category": "loaders",
        "description": "Loads a LoRA model",
        "input": {
            "required": {
                "model": ["MODEL"],
                "clip": ["CLIP"],
                "lora_name": [["lora_a.safetensors"]],
                "strength_model": ["FLOAT", {"default": 1.0}],
                "strength_clip": ["FLOAT", {"default": 1.0}],
            },
            "optional": {},
        },
        "output": ["MODEL", "CLIP"],
        "output_name": ["MODEL", "CLIP"],
        "output_node": False,
    },
    "DeprecatedNode": {
        "display_name": "Old Node",
        "category": "deprecated",
        "description": "Should be hidden",
        "deprecated": True,
        "input": {"required": {}, "optional": {}},
        "output": [],
        "output_name": [],
        "output_node": False,
    },
    "EmptyLatentImage": {
        "display_name": "Empty Latent Image",
        "category": "latent",
        "description": "Creates an empty latent image",
        "input": {
            "required": {
                "width": ["INT", {"default": 512}],
                "height": ["INT", {"default": 512}],
                "batch_size": ["INT", {"default": 1}],
            },
            "optional": {},
        },
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "output_node": False,
    },
    "ControlNetApplyAdvanced": {
        "display_name": "Apply ControlNet (Advanced)",
        "category": "conditioning/controlnet",
        "description": "Apply ControlNet with advanced options",
        "search_aliases": ["cnet"],
        "input": {
            "required": {
                "positive": ["CONDITIONING"],
                "negative": ["CONDITIONING"],
                "control_net": ["CONTROL_NET"],
                "image": ["IMAGE"],
                "strength": ["FLOAT", {"default": 1.0}],
            },
            "optional": {},
        },
        "output": ["CONDITIONING", "CONDITIONING"],
        "output_name": ["positive", "negative"],
        "output_node": False,
    },
}


def _make_cache() -> tuple[NodeCache, AsyncMock]:
    """Create a NodeCache with mocked client."""
    mock_client = AsyncMock(spec=ComfyUIClient)
    mock_client.get_object_info.return_value = dict(SAMPLE_NODES)
    cache = NodeCache(mock_client, ttl=300.0)
    return cache, mock_client


class TestCacheHitAndTTL:
    async def test_cache_hit(self):
        """Second call returns cached data without hitting client."""
        cache, mock_client = _make_cache()

        result1 = await cache.get_all()
        result2 = await cache.get_all()

        assert result1 is result2
        mock_client.get_object_info.assert_awaited_once()

    async def test_ttl_expiration(self):
        """After TTL, cache refetches from client."""
        cache, mock_client = _make_cache()
        cache._ttl = -1  # Always expired

        await cache.get_all()
        await cache.get_all()

        assert mock_client.get_object_info.await_count == 2

    async def test_invalidate(self):
        """invalidate() forces refetch."""
        cache, mock_client = _make_cache()

        await cache.get_all()
        cache.invalidate()
        await cache.get_all()

        assert mock_client.get_object_info.await_count == 2


class TestGetNode:
    async def test_existing_node(self):
        cache, _ = _make_cache()
        node = await cache.get_node("KSampler")
        assert node is not None
        assert node["category"] == "sampling"

    async def test_nonexistent_node(self):
        cache, _ = _make_cache()
        node = await cache.get_node("DoesNotExist")
        assert node is None


class TestGetOutputTypes:
    async def test_checkpoint_outputs(self):
        cache, _ = _make_cache()
        outputs = await cache.get_output_types("CheckpointLoaderSimple")
        assert outputs == ["MODEL", "CLIP", "VAE"]

    async def test_nonexistent_node(self):
        cache, _ = _make_cache()
        outputs = await cache.get_output_types("DoesNotExist")
        assert outputs == []


class TestGetRequiredInputs:
    async def test_ksampler_required(self):
        cache, _ = _make_cache()
        required = await cache.get_required_inputs("KSampler")
        assert "model" in required
        assert "seed" in required
        assert "positive" in required

    async def test_nonexistent_node(self):
        cache, _ = _make_cache()
        required = await cache.get_required_inputs("DoesNotExist")
        assert required == {}


class TestSearch:
    async def test_search_by_query(self):
        cache, _ = _make_cache()
        results = await cache.search(query="sampler")
        names = [r["name"] for r in results]
        assert "KSampler" in names

    async def test_search_by_alias(self):
        cache, _ = _make_cache()
        results = await cache.search(query="cnet")
        names = [r["name"] for r in results]
        assert "ControlNetApplyAdvanced" in names

    async def test_search_by_category(self):
        cache, _ = _make_cache()
        results = await cache.search(category="loaders")
        names = [r["name"] for r in results]
        assert "CheckpointLoaderSimple" in names
        assert "LoraLoader" in names

    async def test_search_by_output_type(self):
        cache, _ = _make_cache()
        results = await cache.search(output_type="MODEL")
        names = [r["name"] for r in results]
        assert "CheckpointLoaderSimple" in names
        assert "LoraLoader" in names
        assert "CLIPTextEncode" not in names

    async def test_search_by_input_type(self):
        cache, _ = _make_cache()
        results = await cache.search(input_type="MODEL")
        names = [r["name"] for r in results]
        assert "KSampler" in names
        assert "LoraLoader" in names

    async def test_search_excludes_deprecated(self):
        cache, _ = _make_cache()
        results = await cache.search()
        names = [r["name"] for r in results]
        assert "DeprecatedNode" not in names

    async def test_search_combined_filters(self):
        cache, _ = _make_cache()
        results = await cache.search(category="loaders", output_type="MODEL")
        names = [r["name"] for r in results]
        assert "CheckpointLoaderSimple" in names
        assert "LoraLoader" in names

    async def test_search_no_results(self):
        cache, _ = _make_cache()
        results = await cache.search(query="nonexistent_xyz")
        assert results == []

    async def test_search_result_structure(self):
        cache, _ = _make_cache()
        results = await cache.search(query="KSampler")
        assert len(results) >= 1
        result = results[0]
        assert "name" in result
        assert "display_name" in result
        assert "category" in result
        assert "description" in result
        assert "outputs" in result
        assert "output_names" in result
        assert "output_node" in result
