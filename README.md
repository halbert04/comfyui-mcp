# ComfyUI MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that connects AI assistants to [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for image, video, and audio generation. Works with Claude Desktop, Claude Code, and any MCP-compatible client.

## What It Does

This server gives AI agents full control over ComfyUI — from one-call image generation to building complex multi-node workflows. It exposes 40+ tools organized into three tiers:

### Quick Generation (one-call tools)

All parameters have sensible defaults. Only a prompt is required:

| Tool | Description |
|------|-------------|
| `text_to_image` | Generate images from text (local SD/Flux model) |
| `image_to_image` | Transform an existing image with a prompt |
| `text_to_video` | Generate video from text (local LTX-Video model) |
| `image_to_video` | Animate an image into video |
| `upscale_image` | AI upscale with an upscale model |
| `inpaint` | Fill masked regions with AI-generated content |
| `dalle3_image` | DALL-E 3 image generation (cloud API) |
| `gpt_image_generate` | GPT Image generation/editing (cloud API) |
| `sora_video_generate` | Sora 2 video generation (cloud API) |
| `merge_videos` | Merge multiple videos into one |

### Any API Node (200+ cloud providers)

Run any of ComfyUI's 200+ API nodes in a single call — Kling, Runway, Luma, Stability, ElevenLabs, Gemini, Veo, Recraft, Minimax, and more:

1. `list_api_nodes(query="kling")` — discover available API nodes
2. `get_node_schema("KlingTextToVideoNode")` — see inputs, types, defaults
3. `run_api_node(node_class="KlingTextToVideoNode", inputs='{"prompt": "..."}')` — execute

Automatically inserts loader nodes for image/video/audio inputs and save nodes for outputs. Validates COMBO values, INT/FLOAT ranges, and required inputs before submitting.

### Custom Workflows (multi-node graphs)

For complex workflows (ControlNet, multi-LoRA, regional prompting, IP-Adapter):

1. `search_nodes(query="controlnet")` — find relevant nodes
2. `get_node_schema("ControlNetApplyAdvanced")` — understand inputs/outputs
3. `create_workflow(template="txt2img")` — start from a template
4. `add_node(workflow_id, "ControlNetLoader", ...)` — add nodes incrementally
5. `suggest_next(workflow_id)` — see what's missing
6. `validate_workflow(workflow_id)` — check for errors
7. `execute_workflow(workflow_id)` — generate results

### System & Utility Tools

| Tool | Description |
|------|-------------|
| `list_models` | List installed models by folder (checkpoints, loras, etc.) |
| `list_samplers_and_schedulers` | List valid sampler/scheduler names |
| `get_system_stats` | ComfyUI system info (VRAM, version, etc.) |
| `get_queue_status` | Current queue state |
| `get_history` | Generation history |
| `get_logs` | Server logs |
| `upload_image` | Upload an image to ComfyUI's input directory |
| `upload_mask` | Upload a mask image for inpainting |
| `get_image_url` | Get URL to view/download generated files |
| `browse_files` | List files in ComfyUI directories |

## Prerequisites

### ComfyUI

You need a running ComfyUI instance. Install it following the [official instructions](https://github.com/comfyanonymous/ComfyUI#installing).

Make sure ComfyUI is running and accessible (default: `http://127.0.0.1:8188`).

### Models (for local generation)

For local generation tools (`text_to_image`, `text_to_video`, etc.), you need models installed in ComfyUI:

- **Image generation**: Any Stable Diffusion or Flux checkpoint in `models/checkpoints/`
- **Video generation**: [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) checkpoint + a T5-XXL text encoder in `models/text_encoders/`
- **Upscaling**: Any upscale model in `models/upscale_models/`

Use `list_models()` through the MCP to see what's currently installed.

### Comfy.org API Key (for cloud generation)

Cloud-based tools (`dalle3_image`, `gpt_image_generate`, `sora_video_generate`, `run_api_node`) require a [Comfy.org](https://comfy.org) API key with credits loaded. Set it via the `COMFY_API_KEY` environment variable.

## Installation

### From source

```bash
git clone https://github.com/halbert04/comfyui-mcp.git
cd comfyui-mcp
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

### Verify installation

```bash
comfyui-mcp --help
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI server URL |
| `COMFYUI_TIMEOUT` | `300` | Max wait time for job completion (seconds) |
| `COMFYUI_POLL_INTERVAL` | `1.0` | Polling interval for job status (seconds) |
| `COMFY_API_KEY` | (empty) | Comfy.org API key for cloud API nodes |
| `COMFYUI_MCP_TRANSPORT` | `stdio` | Transport mode: `stdio` or `streamable-http` |
| `COMFYUI_MCP_HOST` | `127.0.0.1` | Host for HTTP transport |
| `COMFYUI_MCP_PORT` | `8200` | Port for HTTP transport |

## Usage with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "comfyui": {
      "command": "comfyui-mcp",
      "env": {
        "COMFYUI_URL": "http://127.0.0.1:8188",
        "COMFYUI_TIMEOUT": "900",
        "COMFY_API_KEY": "your-comfy-api-key-here"
      }
    }
  }
}
```

Restart Claude Desktop after saving. The ComfyUI tools will appear in the tool list.

## Usage with Claude Code

Add the MCP server to your Claude Code configuration:

```bash
claude mcp add comfyui -- comfyui-mcp
```

Or with environment variables:

```bash
claude mcp add comfyui -e COMFYUI_URL=http://127.0.0.1:8188 -e COMFY_API_KEY=your-key -- comfyui-mcp
```

## Usage as HTTP Server

For non-stdio clients, run as an HTTP server:

```bash
COMFYUI_MCP_TRANSPORT=streamable-http comfyui-mcp
```

This starts an MCP server on `http://127.0.0.1:8200` that any MCP client can connect to.

## Project Structure

```
src/comfyui_mcp/
  server.py          # MCP server entry point and tool registration
  client.py          # HTTP client for ComfyUI API
  config.py          # Environment variable configuration
  workflows.py       # Workflow graph builder and templates
  node_cache.py      # Cached node schema from /object_info
  polling.py         # Job completion polling
  tools/
    generate.py      # One-call generation tools
    api_runner.py    # Generic API node runner (run_api_node, list_api_nodes)
    builder.py       # Workflow builder tools (create, add_node, execute, etc.)
    discovery.py     # Node discovery tools (search_nodes, get_node_schema)
    models.py        # Model listing tools
    system.py        # System info, queue, history, file browsing
  resources/
    resources.py     # MCP resources (workflow templates, system info)
```

## Development

Run tests:

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
