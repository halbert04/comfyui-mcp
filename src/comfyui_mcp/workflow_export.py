"""Convert execution-format workflows to ComfyUI UI format (litegraph).

The UI format embeds in output PNGs via extra_pnginfo, enabling drag-and-drop
back into ComfyUI's node editor with full node layout.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from comfyui_mcp.node_cache import NodeCache

# Layout constants
COL_WIDTH = 350  # horizontal spacing between columns
ROW_HEIGHT = 200  # vertical spacing between nodes
X_OFFSET = 100  # left margin
Y_OFFSET = 100  # top margin
DEFAULT_NODE_WIDTH = 300
DEFAULT_NODE_HEIGHT = 150


def _is_link(value: Any) -> bool:
    """Check if a value is a node link reference [node_id, output_index]."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
    )


def _topological_depth(prompt: dict[str, Any]) -> dict[str, int]:
    """Compute topological depth for each node (0 = no dependencies)."""
    depths: dict[str, int] = {}

    def get_depth(node_id: str) -> int:
        if node_id in depths:
            return depths[node_id]
        # Mark as visited to handle cycles
        depths[node_id] = 0
        node = prompt.get(node_id)
        if not node:
            return 0
        max_dep = 0
        for val in node.get("inputs", {}).values():
            if _is_link(val):
                dep_id = str(val[0])
                if dep_id in prompt:
                    max_dep = max(max_dep, get_depth(dep_id) + 1)
        depths[node_id] = max_dep
        return max_dep

    for nid in prompt:
        get_depth(nid)
    return depths


def _classify_input(input_spec: Any) -> str:
    """Classify an input spec as 'widget' or 'connection'.

    Widget inputs are: INT, FLOAT, STRING, BOOLEAN, COMBO (list of options).
    Connection inputs are typed links: MODEL, CLIP, VAE, IMAGE, etc.
    """
    if not isinstance(input_spec, (list, tuple)) or len(input_spec) == 0:
        return "widget"
    first = input_spec[0]
    # COMBO: first element is a list of options
    if isinstance(first, list):
        return "widget"
    type_name = str(first).upper()
    if type_name in ("INT", "FLOAT", "STRING", "BOOLEAN"):
        return "widget"
    return "connection"


async def to_ui_workflow(
    prompt: dict[str, Any],
    node_cache: NodeCache,
) -> dict[str, Any]:
    """Convert an execution-format prompt to ComfyUI UI workflow format.

    The returned dict can be embedded in extra_pnginfo["workflow"] so that
    output PNGs contain the full workflow for drag-and-drop into ComfyUI.

    Args:
        prompt: Execution-format workflow dict (node_id -> {class_type, inputs}).
        node_cache: NodeCache instance for looking up node schemas.

    Returns:
        ComfyUI UI-format workflow dict with nodes, links, positions.
    """
    if not prompt:
        return {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}

    # Step 1: Compute layout positions via topological depth
    depths = _topological_depth(prompt)

    # Group nodes by depth column
    columns: dict[int, list[str]] = defaultdict(list)
    for nid, depth in sorted(depths.items(), key=lambda x: (x[1], x[0])):
        columns[depth].append(nid)

    # Assign positions
    positions: dict[str, tuple[float, float]] = {}
    for col_idx in sorted(columns):
        for row_idx, nid in enumerate(columns[col_idx]):
            x = X_OFFSET + col_idx * COL_WIDTH
            y = Y_OFFSET + row_idx * ROW_HEIGHT
            positions[nid] = (x, y)

    # Step 2: First pass — collect all links from the execution format
    # Each link: (source_node_id, source_output_idx, target_node_id, target_input_name)
    raw_links: list[tuple[str, int, str, str]] = []
    for nid, node in prompt.items():
        for input_name, val in node.get("inputs", {}).items():
            if _is_link(val):
                raw_links.append((str(val[0]), int(val[1]), nid, input_name))

    # Step 3: Build UI nodes and links
    ui_nodes: list[dict[str, Any]] = []
    ui_links: list[dict[str, Any]] = []
    link_id_counter = 0

    # Pre-compute: for each (source_node, output_idx), which link IDs reference it
    # We need this to populate the "links" list on each output slot
    output_link_map: dict[tuple[str, int], list[int]] = defaultdict(list)
    # And for each (target_node, input_name), the link ID
    input_link_map: dict[tuple[str, str], int] = {}

    # Assign link IDs first
    for src_nid, src_out_idx, tgt_nid, tgt_input_name in raw_links:
        link_id_counter += 1
        output_link_map[(src_nid, src_out_idx)].append(link_id_counter)
        input_link_map[(tgt_nid, tgt_input_name)] = link_id_counter

    # Step 4: Build each UI node
    node_order = 0
    for nid in sorted(prompt.keys(), key=lambda x: (depths.get(x, 0), x)):
        node = prompt[nid]
        class_type = node["class_type"]
        node_inputs = node.get("inputs", {})

        schema = await node_cache.get_node(class_type)

        # Get all input specs from schema
        schema_required = {}
        schema_optional = {}
        if schema:
            schema_required = schema.get("input", {}).get("required", {})
            schema_optional = schema.get("input", {}).get("optional", {})

        # All schema inputs in order (required first, then optional)
        all_schema_inputs: list[tuple[str, Any]] = []
        all_schema_inputs.extend(schema_required.items())
        all_schema_inputs.extend(schema_optional.items())

        # Get output types from schema
        output_types: list[str] = []
        output_names: list[str] = []
        if schema:
            output_types = list(schema.get("output", []))
            output_names = list(schema.get("output_name", output_types))

        # Build UI inputs array and widgets_values
        ui_inputs: list[dict[str, Any]] = []
        widgets_values: list[Any] = []

        for inp_name, inp_spec in all_schema_inputs:
            kind = _classify_input(inp_spec)
            actual_value = node_inputs.get(inp_name)

            if kind == "connection":
                # Connection slot
                link = input_link_map.get((nid, inp_name))
                inp_type = str(inp_spec[0]) if isinstance(inp_spec, (list, tuple)) and inp_spec else "*"
                ui_input: dict[str, Any] = {
                    "name": inp_name,
                    "type": inp_type,
                    "link": link,
                }
                # If this input can also accept a widget value (converted widget)
                if actual_value is not None and not _is_link(actual_value):
                    ui_input["widget"] = {"name": inp_name}
                    widgets_values.append(actual_value)
                ui_inputs.append(ui_input)
            else:
                # Widget input — also add as an input slot (ComfyUI shows these)
                if isinstance(inp_spec, (list, tuple)) and inp_spec and isinstance(inp_spec[0], list):
                    # COMBO
                    inp_type = "COMBO"
                elif isinstance(inp_spec, (list, tuple)) and inp_spec:
                    inp_type = str(inp_spec[0])
                else:
                    inp_type = "STRING"

                link = input_link_map.get((nid, inp_name))
                if link is not None:
                    # This widget input is actually connected via a link
                    ui_inputs.append({
                        "name": inp_name,
                        "type": inp_type,
                        "widget": {"name": inp_name},
                        "link": link,
                    })
                else:
                    ui_inputs.append({
                        "name": inp_name,
                        "type": inp_type,
                        "widget": {"name": inp_name},
                        "link": None,
                    })

                # Add to widgets_values
                if actual_value is not None and not _is_link(actual_value):
                    widgets_values.append(actual_value)
                elif inp_name in node_inputs and _is_link(node_inputs[inp_name]):
                    # Connected widget — use default or None
                    widgets_values.append(None)
                else:
                    # Use default from schema if available
                    default = None
                    if isinstance(inp_spec, (list, tuple)) and len(inp_spec) > 1:
                        if isinstance(inp_spec[1], dict):
                            default = inp_spec[1].get("default")
                        elif isinstance(inp_spec[0], list) and inp_spec[0]:
                            default = inp_spec[0][0]  # First COMBO option
                    widgets_values.append(default)

        # Build UI outputs array
        ui_outputs: list[dict[str, Any]] = []
        for out_idx, out_type in enumerate(output_types):
            out_name = output_names[out_idx] if out_idx < len(output_names) else out_type
            links = output_link_map.get((nid, out_idx), [])
            ui_outputs.append({
                "name": out_name,
                "type": out_type,
                "links": links,
                "slot_index": out_idx,
            })

        pos = positions.get(nid, (X_OFFSET, Y_OFFSET))
        int_id = int(nid) if nid.isdigit() else hash(nid) % 100000

        ui_node = {
            "id": int_id,
            "type": class_type,
            "pos": [pos[0], pos[1]],
            "size": [DEFAULT_NODE_WIDTH, max(DEFAULT_NODE_HEIGHT, len(ui_inputs) * 26 + 60)],
            "flags": {},
            "order": node_order,
            "mode": 0,
            "inputs": ui_inputs,
            "outputs": ui_outputs,
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets_values,
        }
        ui_nodes.append(ui_node)
        node_order += 1

    # Step 5: Build links array
    link_id_counter = 0
    for src_nid, src_out_idx, tgt_nid, tgt_input_name in raw_links:
        link_id_counter += 1

        # Find the target input slot index
        tgt_slot = 0
        for node_data in ui_nodes:
            src_int_id = int(src_nid) if src_nid.isdigit() else hash(src_nid) % 100000
            tgt_int_id = int(tgt_nid) if tgt_nid.isdigit() else hash(tgt_nid) % 100000
            if node_data["id"] == tgt_int_id:
                for slot_idx, inp in enumerate(node_data["inputs"]):
                    if inp["name"] == tgt_input_name:
                        tgt_slot = slot_idx
                        break
                break

        # Get source output type
        src_int_id = int(src_nid) if src_nid.isdigit() else hash(src_nid) % 100000
        tgt_int_id = int(tgt_nid) if tgt_nid.isdigit() else hash(tgt_nid) % 100000
        link_type = "*"
        for node_data in ui_nodes:
            if node_data["id"] == src_int_id:
                if src_out_idx < len(node_data["outputs"]):
                    link_type = node_data["outputs"][src_out_idx]["type"]
                break

        ui_links.append({
            "id": link_id_counter,
            "origin_id": src_int_id,
            "origin_slot": src_out_idx,
            "target_id": tgt_int_id,
            "target_slot": tgt_slot,
            "type": link_type,
        })

    # Compute last_node_id
    max_node_id = max((n["id"] for n in ui_nodes), default=0)

    return {
        "last_node_id": max_node_id,
        "last_link_id": link_id_counter,
        "nodes": ui_nodes,
        "links": ui_links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }
