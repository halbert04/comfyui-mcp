"""Async HTTP client wrapping the ComfyUI REST API."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import aiohttp


class ComfyUIClient:
    """Thin async wrapper around every ComfyUI REST endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(
        self,
        path: str,
        *,
        json: Any | None = None,
        data: Any | None = None,
    ) -> Any:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        async with session.post(url, json=json, data=data) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------
    # Prompt / execution
    # ------------------------------------------------------------------

    async def queue_prompt(
        self,
        prompt: dict,
        client_id: str | None = None,
        api_key: str = "",
    ) -> dict:
        """POST /prompt — queue a workflow for execution."""
        body: dict[str, Any] = {"prompt": prompt}
        if client_id:
            body["client_id"] = client_id
        if api_key:
            body["extra_data"] = {"api_key_comfy_org": api_key}
        return await self._post("/prompt", json=body)

    async def get_history(
        self, prompt_id: str | None = None, max_items: int | None = None
    ) -> dict:
        """GET /history or /history/{prompt_id}."""
        if prompt_id:
            return await self._get(f"/history/{prompt_id}")
        params: dict[str, Any] = {}
        if max_items is not None:
            params["max_items"] = max_items
        return await self._get("/history", params=params or None)

    async def get_queue(self) -> dict:
        """GET /queue — running and pending items."""
        return await self._get("/queue")

    async def interrupt(self) -> Any:
        """POST /interrupt — cancel current execution."""
        return await self._post("/interrupt", json={})

    async def clear_queue(self) -> Any:
        """POST /queue {clear: true} — remove all queued items."""
        return await self._post("/queue", json={"clear": True})

    async def clear_history(self) -> Any:
        """POST /history {clear: true} — remove all history."""
        return await self._post("/history", json={"clear": True})

    # ------------------------------------------------------------------
    # Node / model info
    # ------------------------------------------------------------------

    async def get_object_info(self, node_class: str | None = None) -> dict:
        """GET /object_info or /object_info/{node_class}."""
        if node_class:
            return await self._get(f"/object_info/{node_class}")
        return await self._get("/object_info")

    async def get_models(self, folder: str | None = None) -> list | dict:
        """GET /models or /models/{folder}."""
        if folder:
            return await self._get(f"/models/{folder}")
        return await self._get("/models")

    async def get_embeddings(self) -> list:
        """GET /embeddings."""
        return await self._get("/embeddings")

    async def get_system_stats(self) -> dict:
        """GET /system_stats."""
        return await self._get("/system_stats")

    # ------------------------------------------------------------------
    # File upload / view
    # ------------------------------------------------------------------

    async def upload_file(
        self,
        filepath: str,
        subfolder: str = "",
        overwrite: bool = False,
    ) -> dict:
        """POST /upload/image — upload a file to ComfyUI's input directory."""
        session = await self._get_session()
        url = f"{self.base_url}/upload/image"
        form = aiohttp.FormData()
        form.add_field("image", open(filepath, "rb"), filename=filepath.split("/")[-1])
        if subfolder:
            form.add_field("subfolder", subfolder)
        form.add_field("overwrite", str(overwrite).lower())
        async with session.post(url, data=form) as resp:
            resp.raise_for_status()
            return await resp.json()

    def view_url(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> str:
        """Construct a /view URL for an image."""
        params: dict[str, str] = {
            "filename": filename,
            "type": folder_type,
        }
        if subfolder:
            params["subfolder"] = subfolder
        return f"{self.base_url}/view?{urlencode(params)}"

    # ------------------------------------------------------------------
    # File browsing
    # ------------------------------------------------------------------

    async def list_files(self, directory_type: str = "output") -> list[str]:
        """GET /internal/files/{directory_type} — list files in output/input/temp."""
        return await self._get(f"/internal/files/{directory_type}")

    # ------------------------------------------------------------------
    # Mask upload
    # ------------------------------------------------------------------

    async def upload_mask(
        self,
        filepath: str,
        original_ref: str = "",
        subfolder: str = "",
        overwrite: bool = False,
    ) -> dict:
        """POST /upload/mask — upload a mask image with alpha handling."""
        session = await self._get_session()
        url = f"{self.base_url}/upload/mask"
        form = aiohttp.FormData()
        form.add_field("image", open(filepath, "rb"), filename=filepath.split("/")[-1])
        if original_ref:
            form.add_field("original_ref", original_ref)
        if subfolder:
            form.add_field("subfolder", subfolder)
        form.add_field("overwrite", str(overwrite).lower())
        async with session.post(url, data=form) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

    async def get_model_metadata(self, folder: str, filename: str) -> dict:
        """GET /view_metadata/{folder}?filename={filename} — safetensors metadata."""
        session = await self._get_session()
        url = f"{self.base_url}/view_metadata/{folder}"
        async with session.get(url, params={"filename": filename}) as resp:
            resp.raise_for_status()
            text = await resp.text()
            import json
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}

    # ------------------------------------------------------------------
    # History / queue management
    # ------------------------------------------------------------------

    async def delete_history(self, prompt_ids: list[str]) -> None:
        """POST /history {delete: [...]} — delete specific history entries."""
        session = await self._get_session()
        url = f"{self.base_url}/history"
        async with session.post(url, json={"delete": prompt_ids}) as resp:
            resp.raise_for_status()

    async def delete_queue_items(self, prompt_ids: list[str]) -> None:
        """POST /queue {delete: [...]} — delete specific queue items."""
        session = await self._get_session()
        url = f"{self.base_url}/queue"
        async with session.post(url, json={"delete": prompt_ids}) as resp:
            resp.raise_for_status()

    # ------------------------------------------------------------------
    # Server info
    # ------------------------------------------------------------------

    async def get_features(self) -> dict:
        """GET /features — server feature flags and capabilities."""
        return await self._get("/features")

    async def get_logs(self) -> list[str]:
        """GET /internal/logs — server log entries as lines."""
        session = await self._get_session()
        url = f"{self.base_url}/internal/logs"
        async with session.get(url) as resp:
            resp.raise_for_status()
            import json
            text = await resp.text()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = text
            if isinstance(data, str):
                return [line for line in data.split("\n") if line]
            if isinstance(data, dict):
                return data.get("entries", [])
            if isinstance(data, list):
                return data
            return []

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    async def free_memory(
        self, unload_models: bool = True, free_memory: bool = True
    ) -> Any:
        """POST /free — release VRAM / RAM."""
        return await self._post(
            "/free",
            json={"unload_models": unload_models, "free_memory": free_memory},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
