"""Node discovery tools: search, schema, and workflow suggestions."""

from __future__ import annotations

from typing import Any, Protocol

from fastmcp import Context, FastMCP

from comfyui_mcp.node_cache import NodeCache


class StateStore(Protocol):
    """Minimal protocol for context state operations."""

    async def set_state(self, key: str, value: Any, **kwargs: Any) -> None: ...
    async def get_state(self, key: str) -> Any: ...


# ── Module-level implementations ──────────────────────────────────────


async def search_nodes_impl(
    node_cache: NodeCache,
    query: str = "",
    category: str = "",
    input_type: str = "",
    output_type: str = "",
) -> list[dict[str, Any]]:
    """Search ComfyUI nodes."""
    return await node_cache.search(
        query=query,
        category=category,
        input_type=input_type,
        output_type=output_type,
    )


async def get_node_schema_impl(
    node_cache: NodeCache,
    class_type: str,
) -> dict[str, Any]:
    """Get the full input/output schema for a ComfyUI node class."""
    node = await node_cache.get_node(class_type)
    if not node:
        return {"error": f"Node class '{class_type}' not found"}

    formatted_inputs: dict[str, dict[str, Any]] = {"required": {}, "optional": {}}
    raw_input = node.get("input", {})

    for section in ("required", "optional"):
        for name, spec in raw_input.get(section, {}).items():
            if not isinstance(spec, (list, tuple)) or len(spec) == 0:
                formatted_inputs[section][name] = {"type": "UNKNOWN"}
                continue

            type_info = spec[0]
            constraints = spec[1] if len(spec) > 1 else {}

            if isinstance(type_info, list):
                formatted_inputs[section][name] = {
                    "type": "COMBO",
                    "options": type_info,
                }
                if isinstance(constraints, dict):
                    formatted_inputs[section][name].update(constraints)
            else:
                entry: dict[str, Any] = {"type": str(type_info)}
                if isinstance(constraints, dict):
                    entry.update(constraints)
                formatted_inputs[section][name] = entry

    output_types = list(node.get("output", []))
    output_names = list(node.get("output_name", []))
    formatted_outputs = []
    for i, otype in enumerate(output_types):
        oname = output_names[i] if i < len(output_names) else otype
        formatted_outputs.append({"index": i, "type": otype, "name": oname})

    return {
        "name": class_type,
        "display_name": node.get("display_name", class_type),
        "category": node.get("category", ""),
        "description": node.get("description", ""),
        "inputs": formatted_inputs,
        "outputs": formatted_outputs,
        "output_node": node.get("output_node", False),
    }


async def suggest_next_impl(
    get_node_cache: Any,
    workflow_id: str,
    ctx: StateStore | None = None,
) -> dict[str, Any]:
    """Analyze a workflow and suggest next steps."""
    if ctx is None:
        return {"error": "Context required for stateful workflow operations"}

    wf_state = await ctx.get_state(f"workflow:{workflow_id}")
    if not wf_state:
        return {"error": f"Workflow '{workflow_id}' not found"}

    node_cache: NodeCache = get_node_cache()
    nodes = wf_state["nodes"]

    if not nodes:
        return {
            "unconnected_inputs": [],
            "unused_outputs": [],
            "missing_output_node": True,
            "ready_to_execute": False,
            "suggestions": [
                "Workflow is empty. Start by adding a CheckpointLoaderSimple node."
            ],
        }

    consumed_outputs: set[tuple[str, int]] = set()
    for _nid, node in nodes.items():
        for _input_name, input_val in node["inputs"].items():
            if isinstance(input_val, (list, tuple)) and len(input_val) == 2:
                try:
                    consumed_outputs.add((str(input_val[0]), int(input_val[1])))
                except (ValueError, TypeError):
                    pass

    unconnected_inputs: list[dict[str, Any]] = []
    unused_outputs: list[dict[str, Any]] = []
    has_output_node = False

    for nid, node in nodes.items():
        class_type = node["class_type"]
        schema = await node_cache.get_node(class_type)
        if not schema:
            continue

        if schema.get("output_node", False):
            has_output_node = True

        required = await node_cache.get_required_inputs(class_type)
        for input_name, input_spec in required.items():
            if input_name in node["inputs"]:
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

            providers = await node_cache.search(output_type=inp_type)
            if providers:
                names = [p["name"] for p in providers[:3]]
                suggestion = f"Connect to a {inp_type} source (e.g. {', '.join(names)})"
            else:
                suggestion = f"Connect to a node that outputs {inp_type}"

            unconnected_inputs.append({
                "node_id": nid,
                "node_type": class_type,
                "input": input_name,
                "expected_type": inp_type,
                "suggestion": suggestion,
            })

        output_types = await node_cache.get_output_types(class_type)
        if not schema.get("output_node", False):
            for out_idx, out_type in enumerate(output_types):
                if (nid, out_idx) not in consumed_outputs:
                    consumers = await node_cache.search(input_type=out_type)
                    consumer_names = [c["name"] for c in consumers[:5]]

                    unused_outputs.append({
                        "node_id": nid,
                        "output_index": out_idx,
                        "type": out_type,
                        "suggested_consumers": consumer_names,
                    })

    suggestions: list[str] = []
    ready = True

    if unconnected_inputs:
        ready = False
        suggestions.append(
            f"{len(unconnected_inputs)} required input(s) are not connected."
        )

    if not has_output_node:
        ready = False
        suggestions.append(
            "No output node found. Add a SaveImage or SaveAnimatedWEBP node."
        )

    if ready:
        suggestions.append(
            "All required inputs are connected. Workflow is ready to execute."
        )

    return {
        "unconnected_inputs": unconnected_inputs,
        "unused_outputs": unused_outputs,
        "missing_output_node": not has_output_node,
        "ready_to_execute": ready,
        "suggestions": suggestions,
    }


# ── FastMCP tool registration ──────────────────────────────────────────


def register(mcp: FastMCP, get_node_cache: Any) -> None:
    """Register node discovery tools on the MCP server."""

    @mcp.tool()
    async def search_nodes(
        query: str = "",
        category: str = "",
        input_type: str = "",
        output_type: str = "",
    ) -> list:
        """Search ComfyUI nodes by name, category, or input/output type.

        Args:
            query: Text search in node name, display name, description, and aliases.
            category: Category prefix filter (e.g. "sampling", "loaders", "conditioning").
            input_type: Filter nodes that accept this input type (e.g. "MODEL", "IMAGE").
            output_type: Filter nodes that output this type (e.g. "MODEL", "CONDITIONING").

        Returns:
            List of matching node summaries with name, display_name, category,
            description, outputs, and whether it's an output node.
        """
        return await search_nodes_impl(
            get_node_cache(), query, category, input_type, output_type
        )

    @mcp.tool()
    async def get_node_schema(class_type: str) -> dict:
        """Get the full input/output schema for a ComfyUI node class.

        Args:
            class_type: The node class name (e.g. "KSampler", "CLIPTextEncode").

        Returns:
            Full schema with inputs (required + optional), outputs with types
            and names, category, description, and whether it's an output node.
        """
        return await get_node_schema_impl(get_node_cache(), class_type)

    @mcp.tool()
    async def suggest_next(
        workflow_id: str,
        ctx: Context | None = None,
    ) -> dict:
        """Analyze a workflow and suggest next steps.

        Checks for unconnected inputs, unused outputs, missing output nodes,
        and provides context-aware suggestions.

        Args:
            workflow_id: The workflow to analyze.

        Returns:
            Dict with unconnected_inputs, unused_outputs, missing_output_node,
            ready_to_execute flag, and text suggestions.
        """
        return await suggest_next_impl(get_node_cache, workflow_id, ctx)
