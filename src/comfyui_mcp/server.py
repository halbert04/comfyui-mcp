"""ComfyUI MCP Server — entry point."""

from __future__ import annotations

from fastmcp import FastMCP

from comfyui_mcp.client import ComfyUIClient
from comfyui_mcp.config import get_config
from comfyui_mcp.node_cache import NodeCache
from comfyui_mcp.resources import resources as resources_mod
from comfyui_mcp.tools import api_runner, builder, discovery, generate, models, system

mcp = FastMCP(
    name="ComfyUI",
    instructions="""MCP server for ComfyUI — image, video, audio, and 3D generation.

## Quick Generation (one-call tools)
All parameters have sensible defaults — only prompt is required. Change any
parameter to customize (steps, cfg, size, model, seed, etc.):
- text_to_image(prompt="...") — generate image (local SD/Flux model)
- image_to_image(prompt="...", input_image="...") — transform image
- text_to_video(prompt="...") — generate video (local LTX-V model)
- image_to_video(prompt="...", input_image="...") — animate image
- upscale_image(input_image="...") — AI upscale
- inpaint(prompt="...", input_image="...", mask_image="...") — inpaint
- dalle3_image(prompt="...") — DALL-E 3 (cloud API)
- gpt_image_generate(prompt="...") — GPT Image (cloud API)
- sora_video_generate(prompt="...") — Sora 2 video (cloud API)
- merge_videos(video_files=["a.mp4", "b.mp4"]) — merge videos into one

## Any API Node (200+ cloud providers)
Run any of 200+ API nodes (Kling, Runway, Luma, Stability, ElevenLabs,
Gemini, Veo, Recraft, Minimax, etc.) in a single call:
1. list_api_nodes(query="kling") — discover available API nodes
2. get_node_schema("KlingTextToVideoNode") — see inputs, types, defaults
3. run_api_node(node_class="KlingTextToVideoNode", inputs='{"prompt": "..."}')
Automatically inserts loaders for image/video inputs and save nodes for outputs.

## Custom Workflows (multi-node graphs)
For complex workflows (ControlNet, multi-LoRA, regional prompting, IP-Adapter):
1. search_nodes(query="controlnet") — find relevant nodes
2. get_node_schema("ControlNetApplyAdvanced") — understand inputs/outputs
3. create_workflow(template="txt2img") — start from a template
4. add_node(workflow_id, "ControlNetLoader", ...) — add nodes incrementally
5. suggest_next(workflow_id) — see what's missing
6. validate_workflow(workflow_id) — check for errors
7. execute_workflow(workflow_id) — generate results

## Parallel Execution
For independent jobs (e.g. generating multiple clips), queue them all at once
and wait for all results:
1. run_api_node(..., queue_only=true) × N — queue without waiting
2. wait_for_jobs("id1,id2,id3") — wait for all concurrently

## File Pipeline
Generated files live in the output directory. To use them as inputs:
- copy_output_to_input(filename="ComfyUI_00001_.png") → makes it available
  for LoadImage, image_to_image, run_api_node, etc.

## Key Nodes for Video Editing
- ImageBatch — concatenate/join IMAGE frame batches (for merging video frames)
- GetVideoComponents — decompose VIDEO → IMAGE frames + AUDIO + FPS
- CreateVideo — reassemble IMAGE frames + AUDIO → VIDEO
- Video Slice — trim a video by start_time and duration
- AudioConcat — sequence audio tracks, AudioMerge — overlay/mix audio

## Key Concepts
- Workflows are DAGs of nodes with typed inputs/outputs (MODEL, CLIP, VAE,
  IMAGE, CONDITIONING, LATENT, VIDEO, AUDIO, etc.)
- Connections link outputs to inputs — types must match
- Every workflow needs an output node (SaveImage, SaveVideo, SaveAudio)
- Use list_models() to discover installed models by folder
- Use list_samplers_and_schedulers() for valid sampler/scheduler names
""",
)

_client: ComfyUIClient | None = None
_node_cache: NodeCache | None = None


def get_client() -> ComfyUIClient:
    """Lazy-init the ComfyUI HTTP client."""
    global _client
    if _client is None:
        _client = ComfyUIClient(base_url=get_config().comfyui_url)
    return _client


def get_node_cache() -> NodeCache:
    """Lazy-init the node definition cache."""
    global _node_cache
    if _node_cache is None:
        _node_cache = NodeCache(get_client())
    return _node_cache


# Register all tools and resources
system.register(mcp, get_client)
models.register(mcp, get_client)
generate.register(mcp, get_client)
builder.register(mcp, get_client, get_node_cache)
discovery.register(mcp, get_node_cache)
api_runner.register(mcp, get_client, get_node_cache)
resources_mod.register(mcp, get_client)


def main() -> None:
    """Run the MCP server."""
    config = get_config()

    if config.mcp_transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=config.mcp_host,
            port=config.mcp_port,
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
