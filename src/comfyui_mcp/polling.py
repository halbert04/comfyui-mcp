"""Poll ComfyUI /history/{prompt_id} until job completion."""

from __future__ import annotations

import asyncio
from typing import Any

from comfyui_mcp.client import ComfyUIClient


async def wait_for_completion(
    client: ComfyUIClient,
    prompt_id: str,
    timeout: float = 300.0,
    poll_interval: float = 1.0,
) -> dict[str, Any]:
    """Poll until prompt_id appears in /history, then collect results.

    Returns dict with keys: prompt_id, status, images, outputs.
    """
    elapsed = 0.0

    while elapsed < timeout:
        history = await client.get_history(prompt_id=prompt_id)

        if prompt_id in history:
            entry = history[prompt_id]
            outputs = entry.get("outputs", {})
            status_info = entry.get("status", {})
            status_str = status_info.get("status_str", "completed")
            completed = status_info.get("completed", True)

            images: list[dict[str, Any]] = []
            videos: list[dict[str, Any]] = []
            audios: list[dict[str, Any]] = []
            for _node_id, node_output in outputs.items():
                for img in node_output.get("images", []):
                    entry = {
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                        "view_url": client.view_url(
                            filename=img["filename"],
                            subfolder=img.get("subfolder", ""),
                            folder_type=img.get("type", "output"),
                        ),
                    }
                    if img.get("type") == "output" and (
                        img["filename"].endswith((".mp4", ".webm"))
                        or node_output.get("animated")
                    ):
                        videos.append(entry)
                    else:
                        images.append(entry)
                for aud in node_output.get("audio", []):
                    audios.append({
                        "filename": aud["filename"],
                        "subfolder": aud.get("subfolder", ""),
                        "type": aud.get("type", "output"),
                        "view_url": client.view_url(
                            filename=aud["filename"],
                            subfolder=aud.get("subfolder", ""),
                            folder_type=aud.get("type", "output"),
                        ),
                    })
                for vid in node_output.get("videos", []):
                    videos.append({
                        "filename": vid["filename"],
                        "subfolder": vid.get("subfolder", ""),
                        "type": vid.get("type", "output"),
                        "view_url": client.view_url(
                            filename=vid["filename"],
                            subfolder=vid.get("subfolder", ""),
                            folder_type=vid.get("type", "output"),
                        ),
                    })

            return {
                "prompt_id": prompt_id,
                "status": status_str if completed else "error",
                "images": images,
                "videos": videos,
                "audios": audios,
                "outputs": outputs,
            }

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    # Timeout — check queue for status
    queue = await client.get_queue()
    running_ids = [item[1] for item in queue.get("queue_running", [])]
    pending_ids = [item[1] for item in queue.get("queue_pending", [])]

    if prompt_id in running_ids:
        queue_status = "still_running"
    elif prompt_id in pending_ids:
        queue_status = "still_pending"
    else:
        queue_status = "unknown"

    return {
        "prompt_id": prompt_id,
        "status": "timeout",
        "queue_status": queue_status,
        "images": [],
        "videos": [],
        "audios": [],
        "outputs": {},
        "error": f"Timed out after {timeout}s. Job is {queue_status}.",
    }
