"""System management tools: queue, history, stats, files, logs, job status, result images."""

from __future__ import annotations

from typing import Any

import aiohttp
from fastmcp import FastMCP

from comfyui_mcp.client import ComfyUIClient
from comfyui_mcp.config import get_config
from comfyui_mcp.polling import wait_for_completion


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
    async def wait_for_jobs(
        prompt_ids: str,
        timeout: float = 0,
    ) -> dict:
        """Wait for multiple queued jobs to complete and return all results.

        Use this after queueing multiple jobs with queue_only=true to wait
        for all of them at once, enabling parallel execution.

        Args:
            prompt_ids: Comma-separated list of prompt IDs to wait for.
                Example: "abc123,def456,ghi789"
            timeout: Max seconds to wait per job. Default: 0 (use server
                default). Set higher for slow jobs like video generation.

        Returns:
            Dict mapping each prompt_id to its result (status, images,
            videos, audios, outputs). Jobs are polled concurrently.
        """
        import asyncio

        client: ComfyUIClient = get_client()
        config = get_config()
        effective_timeout = timeout if timeout > 0 else config.comfyui_timeout
        ids = [pid.strip() for pid in prompt_ids.split(",") if pid.strip()]

        if not ids:
            return {"error": "No prompt IDs provided"}

        # Poll all jobs concurrently
        tasks = [
            wait_for_completion(
                client,
                pid,
                timeout=effective_timeout,
                poll_interval=config.comfyui_poll_interval,
            )
            for pid in ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, Any] = {}
        for pid, result in zip(ids, results):
            if isinstance(result, Exception):
                output[pid] = {"prompt_id": pid, "status": "error", "error": str(result)}
            else:
                output[pid] = result

        return output

    @mcp.tool()
    async def copy_output_to_input(
        filename: str,
        subfolder: str = "",
        new_name: str = "",
    ) -> dict:
        """Copy a file from ComfyUI's output directory to input directory.

        Makes generated files (images, videos, audio) available as inputs
        for subsequent workflows. For example, copy a generated image to
        use it in image_to_image, image_to_video, or run_api_node.

        Args:
            filename: The filename in the output directory (e.g.
                "ComfyUI_MCP_00001_.png", "video/ComfyUI_MCP_api_00001_.mp4").
            subfolder: Subfolder within output directory. Default: "" (root).
                Use "video" for video files.
            new_name: Rename the file in the input directory. Default: ""
                (keep original name).

        Returns:
            Dict with the input filename that can be used with LoadImage,
            LoadVideo, etc.
        """
        import io

        client: ComfyUIClient = get_client()
        session = await client._get_session()

        # Download from /view
        view_params: dict[str, str] = {
            "filename": filename,
            "type": "output",
        }
        if subfolder:
            view_params["subfolder"] = subfolder
        view_url = f"{client.base_url}/view"
        async with session.get(view_url, params=view_params) as resp:
            resp.raise_for_status()
            file_bytes = await resp.read()

        # Upload to input via /upload/image
        target_name = new_name or filename.split("/")[-1]
        upload_url = f"{client.base_url}/upload/image"
        form = aiohttp.FormData()
        form.add_field(
            "image",
            io.BytesIO(file_bytes),
            filename=target_name,
        )
        form.add_field("overwrite", "true")
        async with session.post(upload_url, data=form) as resp:
            resp.raise_for_status()
            result = await resp.json()

        return {
            "input_filename": result.get("name", target_name),
            "subfolder": result.get("subfolder", ""),
            "message": f"Copied to input directory as '{result.get('name', target_name)}'",
        }

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
