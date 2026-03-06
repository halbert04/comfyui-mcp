"""System management tools: queue, history, stats, files, logs, job status, result images."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from comfyui_mcp.client import ComfyUIClient


def register(mcp: FastMCP, get_client: Any) -> None:
    """Register system tools on the MCP server."""

    @mcp.tool()
    async def get_queue() -> dict:
        """Get the current ComfyUI queue with running and pending items."""
        client: ComfyUIClient = get_client()
        return await client.get_queue()

    @mcp.tool()
    async def clear_queue() -> dict:
        """Clear all pending items from the ComfyUI queue."""
        client: ComfyUIClient = get_client()
        await client.clear_queue()
        return {"status": "ok", "message": "Queue cleared"}

    @mcp.tool()
    async def cancel_job() -> dict:
        """Cancel the currently running ComfyUI job."""
        client: ComfyUIClient = get_client()
        await client.interrupt()
        return {"status": "ok", "message": "Interrupt signal sent"}

    @mcp.tool()
    async def get_history(max_items: int = 20, prompt_id: str = "") -> dict:
        """Get ComfyUI execution history.

        Args:
            max_items: Maximum number of history entries to return.
            prompt_id: If provided, get history for this specific prompt ID only.
        """
        client: ComfyUIClient = get_client()
        if prompt_id:
            return await client.get_history(prompt_id=prompt_id)
        return await client.get_history(max_items=max_items)

    @mcp.tool()
    async def get_system_stats() -> dict:
        """Get ComfyUI system information including OS, RAM, VRAM, and device info."""
        client: ComfyUIClient = get_client()
        return await client.get_system_stats()

    @mcp.tool()
    async def get_job_status(prompt_id: str) -> dict:
        """Get the status and outputs of a queued ComfyUI job.

        Args:
            prompt_id: The prompt ID returned when the job was queued.
        """
        client: ComfyUIClient = get_client()
        history = await client.get_history(prompt_id=prompt_id)

        if prompt_id not in history:
            # Check queue
            queue = await client.get_queue()
            running_ids = [item[1] for item in queue.get("queue_running", [])]
            pending_ids = [item[1] for item in queue.get("queue_pending", [])]

            if prompt_id in running_ids:
                return {"prompt_id": prompt_id, "status": "running"}
            if prompt_id in pending_ids:
                return {"prompt_id": prompt_id, "status": "pending"}
            return {"prompt_id": prompt_id, "status": "not_found"}

        entry = history[prompt_id]
        status_info = entry.get("status", {})
        return {
            "prompt_id": prompt_id,
            "status": status_info.get("status_str", "completed"),
            "completed": status_info.get("completed", True),
            "outputs": entry.get("outputs", {}),
        }

    @mcp.tool()
    async def get_result_images(prompt_id: str) -> list:
        """Get all output images from a completed ComfyUI job.

        Args:
            prompt_id: The prompt ID of the completed job.

        Returns:
            List of image dicts with filename, subfolder, type, and view_url.
        """
        client: ComfyUIClient = get_client()
        history = await client.get_history(prompt_id=prompt_id)

        if prompt_id not in history:
            return []

        entry = history[prompt_id]
        outputs = entry.get("outputs", {})
        images: list[dict[str, Any]] = []

        for _node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                images.append({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                    "view_url": client.view_url(
                        filename=img["filename"],
                        subfolder=img.get("subfolder", ""),
                        folder_type=img.get("type", "output"),
                    ),
                })

        return images

    @mcp.tool()
    async def list_files(
        directory_type: str = "output",
    ) -> list:
        """List files in a ComfyUI directory.

        Args:
            directory_type: Which directory to list - "output", "input", or "temp".

        Returns:
            List of filenames in the directory.
        """
        client: ComfyUIClient = get_client()
        return await client.list_files(directory_type)

    @mcp.tool()
    async def delete_history_items(prompt_ids: str) -> dict:
        """Delete specific items from ComfyUI execution history.

        Args:
            prompt_ids: Comma-separated list of prompt IDs to delete.
        """
        client: ComfyUIClient = get_client()
        ids = [pid.strip() for pid in prompt_ids.split(",") if pid.strip()]
        if not ids:
            return {"error": "No prompt IDs provided"}
        await client.delete_history(ids)
        return {"status": "ok", "deleted": ids}

    @mcp.tool()
    async def delete_queue_items(prompt_ids: str) -> dict:
        """Delete specific items from the ComfyUI queue.

        Args:
            prompt_ids: Comma-separated list of prompt IDs to remove from the queue.
        """
        client: ComfyUIClient = get_client()
        ids = [pid.strip() for pid in prompt_ids.split(",") if pid.strip()]
        if not ids:
            return {"error": "No prompt IDs provided"}
        await client.delete_queue_items(ids)
        return {"status": "ok", "deleted": ids}

    @mcp.tool()
    async def get_model_metadata(
        folder: str,
        filename: str,
    ) -> dict:
        """Get metadata from a safetensors model file.

        Useful for inspecting what a model was trained on, its configuration,
        resolution, license, etc.

        Args:
            folder: Model folder name (e.g. "checkpoints", "loras", "text_encoders").
            filename: Model filename within the folder.

        Returns:
            Dict of metadata from the safetensors file header.
        """
        client: ComfyUIClient = get_client()
        return await client.get_model_metadata(folder, filename)

    @mcp.tool()
    async def get_features() -> dict:
        """Get ComfyUI server feature flags and capabilities.

        Returns:
            Dict of supported features (preview metadata, max upload size,
            extensions, node replacements, etc.).
        """
        client: ComfyUIClient = get_client()
        return await client.get_features()

    @mcp.tool()
    async def get_logs(max_entries: int = 50) -> list:
        """Get recent ComfyUI server log entries for debugging.

        Args:
            max_entries: Maximum number of log entries to return (most recent first).

        Returns:
            List of log entry strings.
        """
        client: ComfyUIClient = get_client()
        entries = await client.get_logs()
        return entries[-max_entries:] if len(entries) > max_entries else entries
