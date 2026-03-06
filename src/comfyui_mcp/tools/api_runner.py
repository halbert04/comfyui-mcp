"""Generic API node runner — execute any ComfyUI node in a single call."""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastmcp import Context, FastMCP

from comfyui_mcp.client import ComfyUIClient
from comfyui_mcp.config import get_config
from comfyui_mcp.node_cache import NodeCache
from comfyui_mcp.polling import wait_for_completion
from comfyui_mcp.workflow_export import to_ui_workflow
from comfyui_mcp.workflows import WorkflowBuilder


# Media types that can be auto-saved
_MEDIA_TYPES = {"IMAGE", "VIDEO", "AUDIO"}

# Map output type → (SaveNodeClass, input_key, default_save_inputs)
_SAVE_NODE_MAP: dict[str, tuple[str, str, dict[str, Any]]] = {
    "IMAGE": ("SaveImage", "images", {"filename_prefix": "ComfyUI_MCP_api"}),
    "VIDEO": ("SaveVideo", "video", {
        "filename_prefix": "video/ComfyUI_MCP_api",
        "format": "auto",
        "codec": "auto",
    }),
    "AUDIO": ("SaveAudio", "audio", {"filename_prefix": "audio/ComfyUI_MCP_api"}),
}

# Map input type → (LoaderNodeClass, loader_input_key, loader_output_index)
_LOADER_MAP: dict[str, tuple[str, str, int]] = {
    "IMAGE": ("LoadImage", "image", 0),
    "VIDEO": ("LoadVideo", "file", 0),
    "AUDIO": ("LoadAudio", "audio", 0),
}


async def _store_workflow(ctx: Any, workflow: dict, name: str) -> str:
    """Store a workflow in session state and return its ID."""
    wf_id = f"wf_{uuid.uuid4().hex[:8]}"
    node_counter = max((int(k) for k in workflow), default=0)
    await ctx.set_state(f"workflow:{wf_id}", {
        "id": wf_id,
        "name": name,
        "nodes": workflow,
        "node_counter": node_counter,
    })
    return wf_id


def register(mcp: FastMCP, get_client: Any, get_node_cache: Any) -> None:
    """Register API runner tools on the MCP server."""

    @mcp.tool()
    async def run_api_node(
        node_class: str,
        inputs: str = "{}",
        output_format: str = "auto",
        timeout: float = 0,
        queue_only: bool = False,
        ctx: Context | None = None,
    ) -> dict:
        """Run any ComfyUI node in a single call.

        Builds a minimal workflow around the node, auto-inserting loader and
        save nodes as needed. Works with all 200+ API nodes (Kling, Runway,
        Luma, Stability, ElevenLabs, Gemini, etc.) and any built-in node.

        Workflow: discover nodes with list_api_nodes() or search_nodes(),
        check inputs with get_node_schema(), then run with this tool.

        For IMAGE/VIDEO/AUDIO-typed inputs, pass a filename string (e.g.
        "photo.png") and a LoadImage/LoadVideo/LoadAudio node is auto-inserted.

        Args:
            node_class: The node class name (e.g. "KlingTextToVideoNode",
                "OpenAIDalle3", "ElevenLabsTextToSpeech"). Use list_api_nodes()
                or search_nodes() to discover available nodes, and
                get_node_schema() to see required/optional inputs.
            inputs: JSON string of node inputs. Default: "{}".
                Primitive values: {"prompt": "a cat", "seed": 42}.
                File inputs: {"image": "photo.png"} — auto-creates LoadImage.
                Use get_node_schema(node_class) to see all available inputs
                with their types, defaults, and valid ranges.
            output_format: How to save the output. Default: "auto".
                "auto" — detect from node outputs (IMAGE→SaveImage, etc.).
                "image", "video", "audio" — force a specific save type.
                "none" — skip save node (for nodes that save internally).
            timeout: Max seconds to wait for completion. Default: 0 (use
                server default, typically 300s). Set higher for slow API
                nodes like video generation (e.g. 600 for Veo 3).
            queue_only: If true, queue the job and return immediately with
                the prompt_id without waiting for completion. Default: false.
                Use get_job_status(prompt_id) to check progress later, or
                wait_for_jobs([prompt_id1, prompt_id2, ...]) to wait for
                multiple jobs at once.

        Returns:
            Dict with prompt_id, status, images, videos, audios, outputs,
            and workflow_id. If queue_only=true, returns only prompt_id
            and status="queued".
        """
        node_cache: NodeCache = get_node_cache()
        schema = await node_cache.get_node(node_class)
        if not schema:
            return {"error": f"Node class '{node_class}' not found. Use search_nodes() or list_api_nodes() to find available nodes."}

        try:
            user_inputs = json.loads(inputs) if inputs else {}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid inputs JSON: {e}"}

        # Get all input specs for the node
        all_input_specs = await node_cache.get_all_inputs(node_class)
        required_inputs = set(schema.get("input", {}).get("required", {}).keys())
        output_types = list(schema.get("output", []))
        is_output_node = schema.get("output_node", False)

        # Validate required inputs are provided (skip hidden ones like auth tokens)
        hidden = set(schema.get("input", {}).get("hidden", []))
        missing = []
        for req_name in required_inputs:
            if req_name in hidden:
                continue
            spec = all_input_specs.get(req_name)
            # Skip if it has a default value
            if spec and isinstance(spec, (list, tuple)) and len(spec) > 1:
                if isinstance(spec[1], dict) and "default" in spec[1]:
                    continue
            if req_name not in user_inputs:
                missing.append(req_name)
        if missing:
            return {
                "error": f"Missing required inputs: {missing}. "
                         f"Use get_node_schema('{node_class}') to see all inputs.",
            }

        # Validate COMBO inputs have valid values
        errors = []
        for input_name, input_value in user_inputs.items():
            spec = all_input_specs.get(input_name)
            if not spec or not isinstance(spec, (list, tuple)) or len(spec) < 2:
                continue
            spec_type = spec[0]
            spec_opts = spec[1] if isinstance(spec[1], dict) else {}

            # COMBO type with options list
            if spec_type == "COMBO" and "options" in spec_opts:
                valid = spec_opts["options"]
                if input_value not in valid:
                    errors.append(
                        f"Invalid value for '{input_name}': '{input_value}'. "
                        f"Valid options: {valid}"
                    )
            # Old-style combo: first element is a list of valid values
            elif isinstance(spec_type, list) and input_value not in spec_type:
                errors.append(
                    f"Invalid value for '{input_name}': '{input_value}'. "
                    f"Valid options: {spec_type}"
                )
            # INT range validation
            elif spec_type == "INT" and isinstance(input_value, (int, float)):
                mn = spec_opts.get("min")
                mx = spec_opts.get("max")
                if mn is not None and input_value < mn:
                    errors.append(
                        f"Value for '{input_name}' is {input_value}, "
                        f"minimum is {mn}"
                    )
                if mx is not None and input_value > mx:
                    errors.append(
                        f"Value for '{input_name}' is {input_value}, "
                        f"maximum is {mx}"
                    )
            # FLOAT range validation
            elif spec_type == "FLOAT" and isinstance(input_value, (int, float)):
                mn = spec_opts.get("min")
                mx = spec_opts.get("max")
                if mn is not None and input_value < mn:
                    errors.append(
                        f"Value for '{input_name}' is {input_value}, "
                        f"minimum is {mn}"
                    )
                if mx is not None and input_value > mx:
                    errors.append(
                        f"Value for '{input_name}' is {input_value}, "
                        f"maximum is {mx}"
                    )

        if errors:
            return {"error": "Input validation failed", "details": errors}

        wb = WorkflowBuilder()

        # Process inputs: auto-insert loaders for media-typed inputs
        processed_inputs: dict[str, Any] = {}
        for input_name, input_value in user_inputs.items():
            input_spec = all_input_specs.get(input_name)
            if (
                input_spec
                and isinstance(input_spec, (list, tuple))
                and len(input_spec) > 0
                and not isinstance(input_spec[0], list)
                and isinstance(input_value, str)
            ):
                input_type = str(input_spec[0]).upper()
                if input_type in _LOADER_MAP:
                    loader_class, loader_key, loader_idx = _LOADER_MAP[input_type]
                    loader_id = wb.add_node(loader_class, {loader_key: input_value})
                    processed_inputs[input_name] = wb.link(loader_id, loader_idx)
                    continue

            processed_inputs[input_name] = input_value

        # Add the main node
        main_node_id = wb.add_node(node_class, processed_inputs)

        # Determine output type and add save node
        if output_format != "none" and not is_output_node:
            # Determine target output type
            target_type = None
            target_idx = None

            if output_format == "auto":
                for idx, otype in enumerate(output_types):
                    if otype.upper() in _MEDIA_TYPES:
                        target_type = otype.upper()
                        target_idx = idx
                        break
            elif output_format.upper() in _MEDIA_TYPES:
                target_type = output_format.upper()
                # Find the matching output index
                for idx, otype in enumerate(output_types):
                    if otype.upper() == target_type:
                        target_idx = idx
                        break
                if target_idx is None:
                    target_idx = 0  # fallback to first output

            if target_type and target_type in _SAVE_NODE_MAP:
                save_class, save_key, save_defaults = _SAVE_NODE_MAP[target_type]
                save_inputs = {save_key: wb.link(main_node_id, target_idx)}
                save_inputs.update(save_defaults)
                wb.add_node(save_class, save_inputs)
            elif not is_output_node and not any(
                o.upper() in _MEDIA_TYPES for o in output_types
            ):
                return {
                    "error": f"Node '{node_class}' has no media outputs ({output_types}). "
                             f"Use builder tools for complex setups.",
                    "output_types": output_types,
                }

        # Build and execute
        workflow = wb.build()
        config = get_config()
        client: ComfyUIClient = get_client()

        # Convert to UI format for embedding in output PNGs
        extra_pnginfo = None
        try:
            ui_workflow = await to_ui_workflow(workflow, node_cache)
            extra_pnginfo = {"workflow": ui_workflow}
        except Exception:
            pass  # Non-fatal — workflow still executes without UI metadata

        result = await client.queue_prompt(
            workflow, api_key=config.comfy_api_key, extra_pnginfo=extra_pnginfo
        )
        prompt_id = result.get("prompt_id")

        if not prompt_id:
            return {"error": "No prompt_id returned", "details": result}

        if result.get("node_errors"):
            return {
                "error": "Workflow has node errors",
                "prompt_id": prompt_id,
                "node_errors": result["node_errors"],
            }

        # Queue-only mode: return immediately
        if queue_only:
            response: dict[str, Any] = {
                "prompt_id": prompt_id,
                "status": "queued",
            }
            if ctx is not None:
                wf_id = await _store_workflow(ctx, workflow, f"api_{node_class}")
                response["workflow_id"] = wf_id
            return response

        # Wait for completion with optional timeout override
        effective_timeout = timeout if timeout > 0 else config.comfyui_timeout
        completion = await wait_for_completion(
            client,
            prompt_id,
            timeout=effective_timeout,
            poll_interval=config.comfyui_poll_interval,
        )

        if ctx is not None:
            wf_id = await _store_workflow(ctx, workflow, f"api_{node_class}")
            completion["workflow_id"] = wf_id

        return completion

    @mcp.tool()
    async def list_api_nodes(
        query: str = "",
        output_type: str = "",
    ) -> list:
        """List available API nodes (Kling, Runway, Luma, Stability, ElevenLabs, etc.).

        These are cloud-based nodes that run via the Comfy.org API proxy.
        Use get_node_schema(name) to see full input/output details, then
        run_api_node(name, inputs) to execute.

        Args:
            query: Text search in node name/description. Default: "" (all).
            output_type: Filter by output type. Default: "" (all).
                Valid: "IMAGE", "VIDEO", "AUDIO".

        Returns:
            List of dicts with name, display_name, category, description,
            outputs, required_inputs, and optional_inputs.
        """
        node_cache: NodeCache = get_node_cache()
        all_nodes = await node_cache.get_all()
        results = []

        for name, info in all_nodes.items():
            if not info.get("api_node", False):
                continue
            if info.get("deprecated") or info.get("dev_only"):
                continue

            if query:
                q = query.lower()
                searchable = (
                    f"{name} {info.get('display_name', '')} "
                    f"{info.get('description', '')} "
                    f"{info.get('category', '')}"
                ).lower()
                if q not in searchable:
                    continue

            if output_type:
                outputs = [o.upper() for o in info.get("output", [])]
                if output_type.upper() not in outputs:
                    continue

            req = list(info.get("input", {}).get("required", {}).keys())
            opt = [
                k for k in info.get("input", {}).get("optional", {}).keys()
                if k not in ("seed",)  # seed is always optional, skip noise
            ]

            results.append({
                "name": name,
                "display_name": info.get("display_name", name),
                "category": info.get("category", ""),
                "description": info.get("description", ""),
                "outputs": list(info.get("output", [])),
                "required_inputs": req,
                "optional_inputs": opt,
            })

        results.sort(key=lambda r: r["category"])
        return results
