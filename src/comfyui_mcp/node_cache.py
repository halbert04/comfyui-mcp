"""Cache for ComfyUI /object_info node definitions."""

from __future__ import annotations

import time
from typing import Any

from comfyui_mcp.client import ComfyUIClient


class NodeCache:
    """Caches /object_info to avoid hitting ComfyUI on every lookup.

    Shared across all sessions — all clients see the same nodes.
    """

    def __init__(self, client: ComfyUIClient, ttl: float = 300.0) -> None:
        self._client = client
        self._cache: dict[str, Any] | None = None
        self._cache_time: float = 0
        self._ttl: float = ttl

    async def get_all(self) -> dict[str, Any]:
        """Fetch and cache all node schemas from /object_info."""
        if self._cache is None or (time.time() - self._cache_time) > self._ttl:
            self._cache = await self._client.get_object_info()
            self._cache_time = time.time()
        return self._cache

    async def get_node(self, class_type: str) -> dict[str, Any] | None:
        """Get schema for a single node class."""
        all_nodes = await self.get_all()
        return all_nodes.get(class_type)

    async def get_output_types(self, class_type: str) -> list[str]:
        """Return the output type list for a node class, e.g. ['MODEL','CLIP','VAE']."""
        node = await self.get_node(class_type)
        if not node:
            return []
        return list(node.get("output", []))

    async def get_required_inputs(self, class_type: str) -> dict[str, Any]:
        """Return required inputs dict: {name: (type, constraints)}."""
        node = await self.get_node(class_type)
        if not node:
            return {}
        return dict(node.get("input", {}).get("required", {}))

    async def get_all_inputs(self, class_type: str) -> dict[str, Any]:
        """Return all inputs (required + optional) dict: {name: (type, constraints)}."""
        node = await self.get_node(class_type)
        if not node:
            return {}
        result: dict[str, Any] = {}
        result.update(node.get("input", {}).get("required", {}))
        result.update(node.get("input", {}).get("optional", {}))
        return result

    async def search(
        self,
        query: str = "",
        category: str = "",
        input_type: str = "",
        output_type: str = "",
    ) -> list[dict[str, Any]]:
        """Search nodes by name/description, category prefix, input type, output type."""
        all_nodes = await self.get_all()
        results: list[dict[str, Any]] = []

        for name, info in all_nodes.items():
            if info.get("deprecated") or info.get("dev_only"):
                continue

            # Text search
            if query:
                q = query.lower()
                searchable = (
                    f"{name} {info.get('display_name', '')} "
                    f"{info.get('description', '')}"
                ).lower()
                aliases = " ".join(info.get("search_aliases") or []).lower()
                if q not in searchable and q not in aliases:
                    continue

            # Category filter
            if category and not info.get("category", "").lower().startswith(
                category.lower()
            ):
                continue

            # Output type filter
            if output_type:
                outputs = [o.upper() for o in info.get("output", [])]
                if output_type.upper() not in outputs:
                    continue

            # Input type filter
            if input_type:
                all_inputs: dict[str, Any] = {}
                all_inputs.update(info.get("input", {}).get("required", {}))
                all_inputs.update(info.get("input", {}).get("optional", {}))
                found = False
                for inp_val in all_inputs.values():
                    if isinstance(inp_val, (list, tuple)) and len(inp_val) > 0:
                        if input_type.upper() in str(inp_val[0]).upper():
                            found = True
                            break
                if not found:
                    continue

            results.append({
                "name": name,
                "display_name": info.get("display_name", name),
                "category": info.get("category", ""),
                "description": info.get("description", ""),
                "outputs": list(info.get("output", [])),
                "output_names": list(info.get("output_name", [])),
                "output_node": info.get("output_node", False),
            })

        return results

    def invalidate(self) -> None:
        """Force cache refresh on next access."""
        self._cache = None
