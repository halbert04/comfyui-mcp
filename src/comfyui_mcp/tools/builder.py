"""Stateful workflow builder tools."""

from __future__ import annotations

import json
import uuid
from typing import Any, Protocol

from fastmcp import Context, FastMCP

from comfyui_mcp import workflows
from comfyui_mcp.client import ComfyUIClient
from comfyui_mcp.config import get_config
from comfyui_mcp.node_cache import NodeCache
from comfyui_mcp.polling import wait_for_completion
from comfyui_mcp.workflow_export import to_ui_workflow


class StateStore(Protocol):
    """Minimal protocol for context state operations."""

    async def set_state(self, key: str, value: Any, **kwargs: Any) -> None: ...
    async def get_state(self, key: str) -> Any: ...


async def _get_workflow(ctx: StateStore, workflow_id: str) -> dict[str, Any] | None:
    """Retrieve a workflow from session state."""
    return await ctx.get_state(f"workflow:{workflow_id}")


async def _save_workflow(ctx: StateStore, wf_state: dict[str, Any]) -> None:
    """Save a workflow back to session state."""
    await ctx.set_state(f"workflow:{wf_state['id']}", wf_state)


async def _auto_connect(
    workflow_nodes: dict[str, Any],
    new_node_id: str,
    explicit_inputs: dict[str, Any],
    node_cache: NodeCache,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Auto-connect unset required inputs to matching outputs in the workflow."""
    class_type = workflow_nodes[new_node_id]["class_type"]
    required = await node_cache.get_required_inputs(class_type)
    auto_connected: dict[str, Any] = {}
    unconnected: list[dict[str, Any]] = []

    for input_name, input_spec in required.items():
        if input_name in explicit_inputs:
            continue

        if not isinstance(input_spec, (list, tuple)) or len(input_spec) == 0:
            continue

        expected_type = str(input_spec[0]).upper()

        # Skip primitive types and COMBO
        if expected_type in ("INT", "FLOAT", "STRING", "BOOLEAN") or isinstance(
            input_spec[0], list
        ):
            unconnected.append({"input": input_name, "type": expected_type})
            continue

        # Find nodes producing this type
        candidates: list[tuple[str, int, str]] = []
        for nid, node in workflow_nodes.items():
            if nid == new_node_id:
                continue
            outputs = await node_cache.get_output_types(node["class_type"])
            for out_idx, out_type in enumerate(outputs):
                if out_type.upper() == expected_type:
                    candidates.append((nid, out_idx, out_type))

        if len(candidates) == 1:
            nid, out_idx, out_type = candidates[0]
            workflow_nodes[new_node_id]["inputs"][input_name] = [nid, out_idx]
            auto_connected[input_name] = {
                "from_node": nid,
                "from_output": out_idx,
                "type": out_type,
            }
        else:
            unconnected.append({
                "input": input_name,
                "type": expected_type,
                "candidates": len(candidates),
            })

    return auto_connected, unconnected


# ── Module-level tool implementations ──────────────────────────────────


async def create_workflow_impl(
    get_client: Any,
    name: str = "",
    template: str = "",
    overrides: str = "{}",
    ctx: StateStore | None = None,
) -> dict[str, Any]:
    """Create a new workflow, optionally from a template."""
    if ctx is None:
        return {"error": "Context required for stateful workflow operations"}

    try:
        params = json.loads(overrides) if overrides else {}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid overrides JSON: {e}"}

    nodes: dict[str, Any] = {}
    if template:
        client: ComfyUIClient = get_client()
        template_builders = {
            "txt2img": workflows.txt2img,
            "img2img": workflows.img2img,
            "upscale": workflows.upscale,
            "inpaint": workflows.inpaint,
            "txt2video_ltxv": workflows.txt2video_ltxv,
            "img2video_ltxv": workflows.img2video_ltxv,
            "dalle3": workflows.dalle3,
            "gpt_image": workflows.gpt_image,
            "sora_video": workflows.sora_video,
            "merge_videos": workflows.merge_videos,
            "flux_txt2img": workflows.flux_txt2img,
            "wan_txt2video": workflows.wan_txt2video,
            "wan_img2video": workflows.wan_img2video,
        }
        builder_fn = template_builders.get(template)
        if not builder_fn:
            return {
                "error": f"Unknown template: {template}",
                "available": list(template_builders.keys()),
            }

        if template in ("txt2img", "img2img", "inpaint") and "checkpoint" not in params:
            models = await client.get_models(folder="checkpoints")
            if models:
                params["checkpoint"] = models[0]

        if template in ("txt2video_ltxv", "img2video_ltxv"):
            if "checkpoint" not in params:
                models = await client.get_models(folder="checkpoints")
                if models:
                    params["checkpoint"] = models[0]
            if "text_encoder" not in params:
                encoders = await client.get_models(folder="text_encoders")
                for enc in encoders:
                    if "t5xxl" in enc.lower():
                        params["text_encoder"] = enc
                        break
                if "text_encoder" not in params and encoders:
                    params["text_encoder"] = encoders[0]

        if template in ("txt2img",) and "prompt" not in params:
            params["prompt"] = ""
        if template == "img2img" and "input_image" not in params:
            params["input_image"] = ""
        if template == "inpaint":
            if "input_image" not in params:
                params["input_image"] = ""
            if "mask_image" not in params:
                params["mask_image"] = ""
            if "prompt" not in params:
                params["prompt"] = ""
        if template == "img2video_ltxv" and "input_image" not in params:
            params["input_image"] = ""
        if template == "upscale" and "input_image" not in params:
            params["input_image"] = ""

        # Merge videos template
        if template == "merge_videos" and "video_files" not in params:
            params["video_files"] = []

        # Flux template
        if template == "flux_txt2img":
            if "prompt" not in params:
                params["prompt"] = ""
            if "diffusion_model" not in params:
                # Try to auto-detect a Flux model
                for folder in ("diffusion_models", "unet"):
                    try:
                        dm_models = await client.get_models(folder=folder)
                    except Exception:
                        continue
                    for m in dm_models:
                        ml = m.lower()
                        if "flux" in ml or "turbo" in ml:
                            params["diffusion_model"] = m
                            params.setdefault("use_gguf", ml.endswith(".gguf"))
                            break
                    if "diffusion_model" in params:
                        break

        # Wan templates
        if template in ("wan_txt2video", "wan_img2video"):
            if "prompt" not in params:
                params["prompt"] = ""
            if template == "wan_img2video" and "input_image" not in params:
                params["input_image"] = ""
            if "diffusion_model" not in params:
                for folder in ("diffusion_models", "unet"):
                    try:
                        dm_models = await client.get_models(folder=folder)
                    except Exception:
                        continue
                    for m in dm_models:
                        if "wan" in m.lower():
                            params["diffusion_model"] = m
                            break
                    if "diffusion_model" in params:
                        break

        # API node templates — no local model resolution needed
        if template in ("dalle3", "gpt_image", "sora_video"):
            if "prompt" not in params:
                params["prompt"] = ""
        if template == "gpt_image" and "input_image" not in params:
            params.setdefault("input_image", "")
        if template == "sora_video" and "input_image" not in params:
            params.setdefault("input_image", "")

        try:
            nodes = builder_fn(**params)
        except TypeError as e:
            return {"error": f"Invalid template parameters: {e}"}

    wf_id = f"wf_{uuid.uuid4().hex[:8]}"
    node_counter = max((int(k) for k in nodes), default=0)

    wf_state = {
        "id": wf_id,
        "name": name or template or "custom",
        "nodes": nodes,
        "node_counter": node_counter,
    }
    await ctx.set_state(f"workflow:{wf_id}", wf_state)

    node_summary = {}
    for nid, node in nodes.items():
        node_summary[nid] = node["class_type"]

    return {
        "workflow_id": wf_id,
        "node_count": len(nodes),
        "nodes": node_summary,
    }


async def add_node_impl(
    get_node_cache: Any,
    workflow_id: str,
    class_type: str,
    inputs: str = "{}",
    ctx: StateStore | None = None,
) -> dict[str, Any]:
    """Add a node to a workflow with auto-connection."""
    if ctx is None:
        return {"error": "Context required for stateful workflow operations"}

    wf_state = await _get_workflow(ctx, workflow_id)
    if not wf_state:
        return {"error": f"Workflow '{workflow_id}' not found"}

    try:
        explicit_inputs = json.loads(inputs) if inputs else {}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid inputs JSON: {e}"}

    node_cache: NodeCache = get_node_cache()
    node_schema = await node_cache.get_node(class_type)
    if not node_schema:
        return {"error": f"Unknown node class: {class_type}"}

    wf_state["node_counter"] += 1
    new_id = str(wf_state["node_counter"])
    wf_state["nodes"][new_id] = {
        "class_type": class_type,
        "inputs": dict(explicit_inputs),
    }

    auto_connected, unconnected = await _auto_connect(
        wf_state["nodes"], new_id, explicit_inputs, node_cache
    )
    output_types = await node_cache.get_output_types(class_type)
    await _save_workflow(ctx, wf_state)

    return {
        "node_id": new_id,
        "class_type": class_type,
        "auto_connected": auto_connected,
        "unconnected_required": unconnected,
        "outputs": output_types,
    }


async def set_inputs_impl(
    workflow_id: str,
    node_id: str,
    inputs: str,
    ctx: StateStore | None = None,
) -> dict[str, Any]:
    """Update inputs on an existing workflow node."""
    if ctx is None:
        return {"error": "Context required for stateful workflow operations"}

    wf_state = await _get_workflow(ctx, workflow_id)
    if not wf_state:
        return {"error": f"Workflow '{workflow_id}' not found"}

    if node_id not in wf_state["nodes"]:
        return {"error": f"Node '{node_id}' not found in workflow"}

    try:
        new_inputs = json.loads(inputs)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid inputs JSON: {e}"}

    wf_state["nodes"][node_id]["inputs"].update(new_inputs)
    await _save_workflow(ctx, wf_state)

    return {
        "node_id": node_id,
        "class_type": wf_state["nodes"][node_id]["class_type"],
        "inputs": wf_state["nodes"][node_id]["inputs"],
    }


async def remove_node_impl(
    workflow_id: str,
    node_id: str,
    ctx: StateStore | None = None,
) -> dict[str, Any]:
    """Remove a node and clean up dangling connections."""
    if ctx is None:
        return {"error": "Context required for stateful workflow operations"}

    wf_state = await _get_workflow(ctx, workflow_id)
    if not wf_state:
        return {"error": f"Workflow '{workflow_id}' not found"}

    if node_id not in wf_state["nodes"]:
        return {"error": f"Node '{node_id}' not found in workflow"}

    removed_class = wf_state["nodes"].pop(node_id)["class_type"]

    broken: list[dict[str, str]] = []
    for nid, node in wf_state["nodes"].items():
        for input_name, input_val in list(node["inputs"].items()):
            if isinstance(input_val, (list, tuple)) and len(input_val) == 2:
                if str(input_val[0]) == str(node_id):
                    del node["inputs"][input_name]
                    broken.append({
                        "node_id": nid,
                        "node_type": node["class_type"],
                        "input": input_name,
                    })

    await _save_workflow(ctx, wf_state)

    return {
        "removed": node_id,
        "removed_class": removed_class,
        "broken_connections": broken,
    }


async def get_workflow_impl(
    get_node_cache: Any,
    workflow_id: str,
    ctx: StateStore | None = None,
) -> dict[str, Any]:
    """Get the full workflow graph state with type information."""
    if ctx is None:
        return {"error": "Context required for stateful workflow operations"}

    wf_state = await _get_workflow(ctx, workflow_id)
    if not wf_state:
        return {"error": f"Workflow '{workflow_id}' not found"}

    node_cache: NodeCache = get_node_cache()

    enriched_nodes: dict[str, Any] = {}
    unconnected_required: list[dict[str, Any]] = []
    has_output_node = False

    for nid, node in wf_state["nodes"].items():
        class_type = node["class_type"]
        output_types = await node_cache.get_output_types(class_type)
        schema = await node_cache.get_node(class_type)

        enriched_nodes[nid] = {
            "class_type": class_type,
            "inputs": node["inputs"],
            "outputs": output_types,
        }

        if schema and schema.get("output_node", False):
            has_output_node = True

        required = await node_cache.get_required_inputs(class_type)
        for input_name, input_spec in required.items():
            if input_name in node["inputs"]:
                continue
            if not isinstance(input_spec, (list, tuple)) or len(input_spec) == 0:
                continue
            input_type = str(input_spec[0]).upper()
            if isinstance(input_spec[0], list):
                continue
            if input_type in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                if len(input_spec) > 1 and isinstance(input_spec[1], dict):
                    if "default" in input_spec[1]:
                        continue
            unconnected_required.append({
                "node_id": nid,
                "node_type": class_type,
                "input": input_name,
                "type": input_type,
            })

    return {
        "id": wf_state["id"],
        "name": wf_state["name"],
        "node_count": len(wf_state["nodes"]),
        "nodes": enriched_nodes,
        "unconnected_required": unconnected_required,
        "has_output_node": has_output_node,
    }


async def validate_workflow_impl(
    get_node_cache: Any,
    workflow_id: str,
    ctx: StateStore | None = None,
) -> dict[str, Any]:
    """Validate a workflow for correctness without executing it."""
    if ctx is None:
        return {"error": "Context required for stateful workflow operations"}

    wf_state = await _get_workflow(ctx, workflow_id)
    if not wf_state:
        return {"error": f"Workflow '{workflow_id}' not found"}

    node_cache: NodeCache = get_node_cache()
    errors: list[str] = []
    has_output_node = False

    nodes = wf_state["nodes"]
    if not nodes:
        errors.append("Workflow has no nodes")
        return {"valid": False, "errors": errors}

    for nid, node in nodes.items():
        class_type = node["class_type"]
        schema = await node_cache.get_node(class_type)
        if not schema:
            errors.append(f"Node {nid}: unknown class_type '{class_type}'")
            continue

        if schema.get("output_node", False):
            has_output_node = True

        required = await node_cache.get_required_inputs(class_type)
        for input_name, input_spec in required.items():
            if input_name in node["inputs"]:
                val = node["inputs"][input_name]
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    ref_node_id = str(val[0])
                    ref_output_idx = int(val[1])
                    if ref_node_id not in nodes:
                        errors.append(
                            f"Node {nid} ({class_type}): input '{input_name}' "
                            f"references non-existent node '{ref_node_id}'"
                        )
                    else:
                        ref_class = nodes[ref_node_id]["class_type"]
                        ref_outputs = await node_cache.get_output_types(ref_class)
                        if ref_output_idx >= len(ref_outputs):
                            errors.append(
                                f"Node {nid} ({class_type}): input '{input_name}' "
                                f"references output index {ref_output_idx} of "
                                f"node {ref_node_id} ({ref_class}), "
                                f"but it only has {len(ref_outputs)} outputs"
                            )
                        elif (
                            isinstance(input_spec, (list, tuple))
                            and len(input_spec) > 0
                            and not isinstance(input_spec[0], list)
                        ):
                            expected_type = str(input_spec[0]).upper()
                            actual_type = ref_outputs[ref_output_idx].upper()
                            if (
                                expected_type != actual_type
                                and expected_type != "*"
                                and actual_type != "*"
                            ):
                                errors.append(
                                    f"Node {nid} ({class_type}): input '{input_name}' "
                                    f"expects {expected_type} but connected to "
                                    f"{actual_type} from node {ref_node_id}"
                                )
                continue

            if not isinstance(input_spec, (list, tuple)) or len(input_spec) == 0:
                continue
            if isinstance(input_spec[0], list):
                continue
            inp_type = str(input_spec[0]).upper()
            if inp_type in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                if len(input_spec) > 1 and isinstance(input_spec[1], dict):
                    if "default" in input_spec[1]:
                        continue
            errors.append(
                f"Node {nid} ({class_type}): required input '{input_name}' "
                f"({inp_type}) is not connected"
            )

    if not has_output_node:
        errors.append("Workflow has no output node (e.g. SaveImage, SaveAnimatedWEBP)")

    if errors:
        return {"valid": False, "errors": errors}
    return {"valid": True}


async def execute_workflow_impl(
    get_client: Any,
    get_node_cache: Any,
    workflow_id: str,
    wait: bool = True,
    ctx: StateStore | None = None,
) -> dict[str, Any]:
    """Execute a workflow built with builder tools."""
    if ctx is None:
        return {"error": "Context required for stateful workflow operations"}

    wf_state = await _get_workflow(ctx, workflow_id)
    if not wf_state:
        return {"error": f"Workflow '{workflow_id}' not found"}

    config = get_config()
    client: ComfyUIClient = get_client()

    # Convert to UI format for embedding in output PNGs
    extra_pnginfo = None
    try:
        node_cache: NodeCache = get_node_cache()
        ui_workflow = await to_ui_workflow(wf_state["nodes"], node_cache)
        extra_pnginfo = {"workflow": ui_workflow}
    except Exception:
        pass  # Non-fatal — workflow still executes without UI metadata

    result = await client.queue_prompt(
        wf_state["nodes"], api_key=config.comfy_api_key, extra_pnginfo=extra_pnginfo
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

    if not wait:
        return {
            "prompt_id": prompt_id,
            "status": "queued",
            "number": result.get("number"),
            "workflow_id": workflow_id,
        }

    completion = await wait_for_completion(
        client,
        prompt_id,
        timeout=config.comfyui_timeout,
        poll_interval=config.comfyui_poll_interval,
    )
    completion["workflow_id"] = workflow_id
    return completion


# ── FastMCP tool registration ──────────────────────────────────────────


def register(mcp: FastMCP, get_client: Any, get_node_cache: Any) -> None:
    """Register workflow builder tools on the MCP server."""

    @mcp.tool()
    async def create_workflow(
        name: str = "",
        template: str = "",
        overrides: str = "{}",
        ctx: Context | None = None,
    ) -> dict:
        """Create a new workflow, optionally from a template.

        Args:
            name: Human-readable name for the workflow.
            template: Template to start from: "txt2img", "img2img", "upscale",
                "inpaint", "txt2video_ltxv", "img2video_ltxv", "flux_txt2img",
                "wan_txt2video", "wan_img2video", "dalle3", "gpt_image",
                "sora_video", or "merge_videos". Empty creates an empty workflow.
            overrides: JSON string of parameter overrides for the template
                (e.g. '{"prompt": "a cat", "checkpoint": "sdxl.safetensors"}').

        Returns:
            Dict with workflow_id, node_count, and node summary.
        """
        return await create_workflow_impl(get_client, name, template, overrides, ctx)

    @mcp.tool()
    async def add_node(
        workflow_id: str,
        class_type: str,
        inputs: str = "{}",
        ctx: Context | None = None,
    ) -> dict:
        """Add a node to a workflow with auto-connection of typed inputs.

        Args:
            workflow_id: The workflow to modify.
            class_type: Node class name (e.g. "CLIPTextEncode", "KSampler").
            inputs: JSON string of input values/connections.
                Values: {"text": "a cat", "seed": 42}
                Connections: {"clip": ["1", 1]} (links to node "1", output index 1).

        Returns:
            Dict with node_id, class_type, auto_connected inputs,
            unconnected required inputs, and output types.
        """
        return await add_node_impl(get_node_cache, workflow_id, class_type, inputs, ctx)

    @mcp.tool()
    async def set_inputs(
        workflow_id: str,
        node_id: str,
        inputs: str,
        ctx: Context | None = None,
    ) -> dict:
        """Update inputs on an existing workflow node.

        Args:
            workflow_id: The workflow to modify.
            node_id: The node ID to update.
            inputs: JSON string of inputs to set/update.
                Values: {"text": "a cat"}
                Connections: {"clip": ["1", 1]}.

        Returns:
            Dict with the updated node inputs.
        """
        return await set_inputs_impl(workflow_id, node_id, inputs, ctx)

    @mcp.tool()
    async def remove_node(
        workflow_id: str,
        node_id: str,
        ctx: Context | None = None,
    ) -> dict:
        """Remove a node from a workflow and clean up dangling connections.

        Args:
            workflow_id: The workflow to modify.
            node_id: The node ID to remove.

        Returns:
            Dict with removed node ID and any broken connections.
        """
        return await remove_node_impl(workflow_id, node_id, ctx)

    @mcp.tool()
    async def get_workflow(
        workflow_id: str,
        ctx: Context | None = None,
    ) -> dict:
        """Get the full workflow graph state with type information.

        Args:
            workflow_id: The workflow to inspect.

        Returns:
            Full graph with node types, inputs, outputs, unconnected inputs,
            and whether an output node exists.
        """
        return await get_workflow_impl(get_node_cache, workflow_id, ctx)

    @mcp.tool()
    async def validate_workflow(
        workflow_id: str,
        ctx: Context | None = None,
    ) -> dict:
        """Validate a workflow for correctness without executing it.

        Checks: all node class_types exist, required inputs connected,
        connection type compatibility, output node exists.

        Args:
            workflow_id: The workflow to validate.

        Returns:
            Dict with valid=True or a list of errors.
        """
        return await validate_workflow_impl(get_node_cache, workflow_id, ctx)

    @mcp.tool()
    async def execute_workflow(
        workflow_id: str,
        wait: bool = True,
        ctx: Context | None = None,
    ) -> dict:
        """Execute a workflow that was built with builder tools.

        Args:
            workflow_id: The workflow to execute.
            wait: If True, poll until completion and return results with images.

        Returns:
            Dict with prompt_id, status, images, outputs, and workflow_id.
        """
        return await execute_workflow_impl(get_client, get_node_cache, workflow_id, wait, ctx)
