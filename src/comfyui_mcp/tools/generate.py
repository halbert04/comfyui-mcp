"""High-level image/video generation tools with workflow_id return."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastmcp import Context, FastMCP

from comfyui_mcp import workflows
from comfyui_mcp.client import ComfyUIClient
from comfyui_mcp.config import get_config
from comfyui_mcp.node_cache import NodeCache
from comfyui_mcp.polling import wait_for_completion
from comfyui_mcp.workflow_export import to_ui_workflow

logger = logging.getLogger(__name__)


async def _resolve_checkpoint(client: ComfyUIClient, checkpoint: str) -> str:
    """Auto-select first available checkpoint if none specified."""
    if checkpoint:
        return checkpoint
    models = await client.get_models(folder="checkpoints")
    if not models:
        raise ValueError("No checkpoints found in ComfyUI. Please install a checkpoint model.")
    return models[0]


async def _resolve_text_encoder(client: ComfyUIClient, text_encoder: str) -> str:
    """Auto-select a T5 text encoder for LTX-V if none specified."""
    if text_encoder:
        return text_encoder
    models = await client.get_models(folder="text_encoders")
    # Prefer t5xxl variants
    for m in models:
        if "t5xxl" in m.lower():
            return m
    if models:
        return models[0]
    raise ValueError("No text encoders found in ComfyUI. Please install a T5 XXL text encoder.")


async def _resolve_upscale_model(client: ComfyUIClient, model_name: str) -> str:
    """Auto-select first available upscale model if none specified."""
    if model_name:
        return model_name
    models = await client.get_models(folder="upscale_models")
    if not models:
        raise ValueError("No upscale models found in ComfyUI. Please install an upscale model.")
    return models[0]


async def _resolve_flux_model(client: ComfyUIClient) -> tuple[str, bool]:
    """Auto-detect a Flux diffusion model. Returns (model_name, use_gguf)."""
    for folder in ("diffusion_models", "unet"):
        try:
            models = await client.get_models(folder=folder)
        except Exception:
            continue
        for m in models:
            ml = m.lower()
            if "flux" in ml or "turbo" in ml:
                return m, ml.endswith(".gguf")
    raise ValueError(
        "No Flux model found. Install a Flux model in "
        "models/diffusion_models/ (safetensors or GGUF). "
        "Use list_models() to see installed models."
    )


async def _resolve_flux_clips(
    client: ComfyUIClient, clip1: str, clip2: str
) -> tuple[str, str]:
    """Auto-detect CLIP models for Flux (clip_l + t5xxl)."""
    if clip1 and clip2:
        return clip1, clip2
    encoders = await client.get_models(folder="text_encoders")
    resolved1, resolved2 = clip1, clip2
    if not resolved1:
        for e in encoders:
            if "clip_l" in e.lower():
                resolved1 = e
                break
        if not resolved1:
            raise ValueError(
                "No clip_l text encoder found. Install clip_l.safetensors "
                "in models/text_encoders/."
            )
    if not resolved2:
        for e in encoders:
            if "t5xxl" in e.lower():
                resolved2 = e
                break
        if not resolved2:
            raise ValueError(
                "No T5-XXL text encoder found. Install a t5xxl encoder "
                "in models/text_encoders/."
            )
    return resolved1, resolved2


async def _resolve_wan_model(client: ComfyUIClient) -> str:
    """Auto-detect a Wan 2.2 diffusion model (GGUF)."""
    for folder in ("diffusion_models", "unet"):
        try:
            models = await client.get_models(folder=folder)
        except Exception:
            continue
        for m in models:
            if "wan" in m.lower():
                return m
    raise ValueError(
        "No Wan 2.2 model found. Install a Wan 2.2 GGUF model "
        "in models/diffusion_models/. "
        "Use list_models() to see installed models."
    )


async def _resolve_wan_clip(client: ComfyUIClient) -> str:
    """Auto-detect UMT5-XXL text encoder for Wan 2.2."""
    encoders = await client.get_models(folder="text_encoders")
    for e in encoders:
        if "umt5" in e.lower():
            return e
    raise ValueError(
        "No UMT5-XXL text encoder found for Wan 2.2. Install "
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors in models/text_encoders/."
    )


async def _store_workflow(ctx: Context, workflow: dict, name: str) -> str:
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


async def _build_ui_workflow(
    workflow: dict, node_cache: NodeCache | None
) -> dict[str, Any] | None:
    """Convert execution workflow to UI format for embedding in PNGs.

    Returns None if conversion fails (non-fatal — workflow still executes).
    """
    if node_cache is None:
        return None
    try:
        return await to_ui_workflow(workflow, node_cache)
    except Exception:
        logger.debug("Failed to convert workflow to UI format", exc_info=True)
        return None


async def _queue_and_wait(
    client: ComfyUIClient,
    workflow: dict,
    node_cache: NodeCache | None = None,
) -> dict[str, Any]:
    """Queue a workflow and wait for results.

    If node_cache is provided, converts the workflow to UI format and embeds
    it in output PNGs via extra_pnginfo, enabling drag-and-drop back into
    ComfyUI's node editor.
    """
    config = get_config()

    # Convert to UI format for embedding in output PNGs
    extra_pnginfo = None
    ui_workflow = await _build_ui_workflow(workflow, node_cache)
    if ui_workflow is not None:
        extra_pnginfo = {"workflow": ui_workflow}

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

    return await wait_for_completion(
        client,
        prompt_id,
        timeout=config.comfyui_timeout,
        poll_interval=config.comfyui_poll_interval,
    )


def register(mcp: FastMCP, get_client: Any, get_node_cache: Any = None) -> None:
    """Register generate tools on the MCP server."""

    @mcp.tool()
    async def text_to_image(
        prompt: str,
        negative_prompt: str = "",
        checkpoint: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 8.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = -1,
        batch_size: int = 1,
        lora_name: str = "",
        lora_strength: float = 1.0,
        ctx: Context | None = None,
    ) -> dict:
        """Generate images from a text description.

        All parameters except prompt are optional with sensible defaults.
        Change any parameter to customize the generation.

        Args:
            prompt: Text description of the desired image.
            negative_prompt: Things to avoid. Default: "" (none).
            checkpoint: Checkpoint model name. Default: "" (auto-selects first
                available). Use list_models(folder="checkpoints") to see options.
            width: Image width in pixels. Default: 512. Must be multiple of 8.
            height: Image height in pixels. Default: 512. Must be multiple of 8.
            steps: Sampling steps. Default: 20. Higher = better quality, slower.
                Typical: 15-50.
            cfg: Classifier-free guidance scale. Default: 8.0. Higher = closer
                to prompt. Typical: 1.0-20.0.
            sampler: Sampler algorithm. Default: "euler". Valid: "euler",
                "dpmpp_2m", "dpmpp_sde", "ddim", "uni_pc", etc.
                Use list_samplers_and_schedulers() to see all.
            scheduler: Noise scheduler. Default: "normal". Valid: "normal",
                "karras", "sgm_uniform", "exponential", etc.
                Use list_samplers_and_schedulers() to see all.
            seed: Random seed for reproducibility. Default: -1 (random).
            batch_size: Number of images to generate. Default: 1.
            lora_name: LoRA model to apply. Default: "" (none).
                Use list_models(folder="loras") to see options.
            lora_strength: LoRA influence. Default: 1.0. Range: 0.0-2.0.

        Returns:
            Dict with prompt_id, status, images, outputs, and workflow_id.
            The workflow_id can be used with builder tools to modify and re-execute.
        """
        client: ComfyUIClient = get_client()
        checkpoint = await _resolve_checkpoint(client, checkpoint)

        wf = workflows.txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            checkpoint=checkpoint,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            batch_size=batch_size,
            lora_name=lora_name,
            lora_strength=lora_strength,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "text_to_image")
            result["workflow_id"] = wf_id

        return result

    @mcp.tool()
    async def image_to_image(
        prompt: str,
        input_image: str,
        negative_prompt: str = "",
        checkpoint: str = "",
        denoise: float = 0.75,
        steps: int = 20,
        cfg: float = 8.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = -1,
        lora_name: str = "",
        lora_strength: float = 1.0,
        ctx: Context | None = None,
    ) -> dict:
        """Transform an existing image guided by a text prompt.

        All parameters except prompt and input_image are optional with
        sensible defaults. Change any parameter to customize.

        Args:
            prompt: Text description guiding the transformation.
            input_image: Filename in ComfyUI's input directory. Upload first
                with upload_image if needed.
            negative_prompt: Things to avoid. Default: "" (none).
            checkpoint: Checkpoint model. Default: "" (auto-selects).
                Use list_models(folder="checkpoints") to see options.
            denoise: Denoising strength. Default: 0.75. Range: 0.0 (no change)
                to 1.0 (full regeneration). Typical: 0.5-0.8.
            steps: Sampling steps. Default: 20. Typical: 15-50.
            cfg: Guidance scale. Default: 8.0. Typical: 1.0-20.0.
            sampler: Sampler algorithm. Default: "euler".
                Use list_samplers_and_schedulers() to see all.
            scheduler: Noise scheduler. Default: "normal".
                Use list_samplers_and_schedulers() to see all.
            seed: Random seed. Default: -1 (random).
            lora_name: LoRA model. Default: "" (none).
                Use list_models(folder="loras") to see options.
            lora_strength: LoRA influence. Default: 1.0. Range: 0.0-2.0.

        Returns:
            Dict with prompt_id, status, images, outputs, and workflow_id.
        """
        client: ComfyUIClient = get_client()
        checkpoint = await _resolve_checkpoint(client, checkpoint)

        wf = workflows.img2img(
            prompt=prompt,
            input_image=input_image,
            negative_prompt=negative_prompt,
            checkpoint=checkpoint,
            denoise=denoise,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            lora_name=lora_name,
            lora_strength=lora_strength,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "image_to_image")
            result["workflow_id"] = wf_id

        return result

    @mcp.tool()
    async def upscale_image(
        input_image: str,
        upscale_model: str = "",
        ctx: Context | None = None,
    ) -> dict:
        """Upscale an image using an AI upscaling model.

        Args:
            input_image: Filename in ComfyUI's input directory. Upload first
                with upload_image if needed.
            upscale_model: Upscale model name. Default: "" (auto-selects first
                available). Use list_models(folder="upscale_models") to see options.

        Returns:
            Dict with prompt_id, status, images, outputs, and workflow_id.
        """
        client: ComfyUIClient = get_client()
        upscale_model = await _resolve_upscale_model(client, upscale_model)

        wf = workflows.upscale(
            input_image=input_image,
            upscale_model=upscale_model,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "upscale_image")
            result["workflow_id"] = wf_id

        return result

    @mcp.tool()
    async def inpaint(
        prompt: str,
        input_image: str,
        mask_image: str,
        negative_prompt: str = "",
        checkpoint: str = "",
        denoise: float = 1.0,
        steps: int = 20,
        cfg: float = 8.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = -1,
        grow_mask_by: int = 6,
        ctx: Context | None = None,
    ) -> dict:
        """Fill in masked regions of an image with AI-generated content.

        All parameters except prompt, input_image, and mask_image are
        optional with sensible defaults. Change any parameter to customize.

        Args:
            prompt: Text description of what to generate in the masked area.
            input_image: Filename of the source image in ComfyUI's input directory.
            mask_image: Filename of the mask image. White areas will be inpainted.
            negative_prompt: Things to avoid. Default: "" (none).
            checkpoint: Checkpoint model. Default: "" (auto-selects).
                Use list_models(folder="checkpoints") to see options.
            denoise: Denoising strength for inpainted area. Default: 1.0.
                Range: 0.0-1.0.
            steps: Sampling steps. Default: 20. Typical: 15-50.
            cfg: Guidance scale. Default: 8.0. Typical: 1.0-20.0.
            sampler: Sampler algorithm. Default: "euler".
                Use list_samplers_and_schedulers() to see all.
            scheduler: Noise scheduler. Default: "normal".
                Use list_samplers_and_schedulers() to see all.
            seed: Random seed. Default: -1 (random).
            grow_mask_by: Pixels to expand the mask boundary for smoother
                blending. Default: 6. Range: 0-64.

        Returns:
            Dict with prompt_id, status, images, outputs, and workflow_id.
        """
        client: ComfyUIClient = get_client()
        checkpoint = await _resolve_checkpoint(client, checkpoint)

        wf = workflows.inpaint(
            prompt=prompt,
            input_image=input_image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            checkpoint=checkpoint,
            denoise=denoise,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            grow_mask_by=grow_mask_by,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "inpaint")
            result["workflow_id"] = wf_id

        return result

    @mcp.tool()
    async def text_to_video(
        prompt: str,
        negative_prompt: str = "low quality, worst quality, deformed, distorted",
        checkpoint: str = "",
        text_encoder: str = "",
        width: int = 768,
        height: int = 512,
        length: int = 97,
        steps: int = 30,
        cfg: float = 3.0,
        seed: int = -1,
        frame_rate: float = 25.0,
        ctx: Context | None = None,
    ) -> dict:
        """Generate a video from a text description using LTX-Video (local model).

        All parameters except prompt are optional with sensible defaults.
        Change any parameter to customize. For cloud-based video generation,
        use sora_video_generate() or run_api_node() with Kling/Runway/Luma/etc.

        Args:
            prompt: Text description of the desired video.
            negative_prompt: Things to avoid. Default: "low quality, worst
                quality, deformed, distorted".
            checkpoint: LTX-Video checkpoint. Default: "" (auto-selects).
                Use list_models(folder="checkpoints") to see options.
            text_encoder: T5 text encoder filename. Default: "" (auto-selects
                T5-XXL). Use list_models(folder="text_encoders") to see options.
            width: Video width in pixels. Default: 768. Must be multiple of 32.
            height: Video height in pixels. Default: 512. Must be multiple of 32.
            length: Number of frames. Default: 97. Must be multiple of 8 + 1
                (e.g. 25, 33, 41, 49, 97). More frames = longer video.
            steps: Sampling steps. Default: 30. Typical: 20-50.
            cfg: Guidance scale. Default: 3.0. Typical: 1.0-7.0.
            seed: Random seed. Default: -1 (random).
            frame_rate: Output FPS. Default: 25.0.

        Returns:
            Dict with prompt_id, status, images, videos, outputs, and workflow_id.
        """
        client: ComfyUIClient = get_client()
        checkpoint = await _resolve_checkpoint(client, checkpoint)
        text_encoder = await _resolve_text_encoder(client, text_encoder)

        wf = workflows.txt2video_ltxv(
            prompt=prompt,
            negative_prompt=negative_prompt,
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            width=width,
            height=height,
            length=length,
            steps=steps,
            cfg=cfg,
            seed=seed,
            frame_rate=frame_rate,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "text_to_video")
            result["workflow_id"] = wf_id

        return result

    @mcp.tool()
    async def image_to_video(
        prompt: str,
        input_image: str,
        negative_prompt: str = "low quality, worst quality, deformed, distorted",
        checkpoint: str = "",
        text_encoder: str = "",
        width: int = 768,
        height: int = 512,
        length: int = 97,
        steps: int = 30,
        cfg: float = 3.0,
        seed: int = -1,
        frame_rate: float = 25.0,
        strength: float = 0.85,
        ctx: Context | None = None,
    ) -> dict:
        """Generate a video from an image using LTX-Video (local model).

        All parameters except prompt and input_image are optional with
        sensible defaults. Change any parameter to customize. For cloud-based
        I2V, use sora_video_generate(input_image=...) or run_api_node() with
        Kling/Runway/Luma/etc.

        Args:
            prompt: Text description guiding the video motion and style.
            input_image: Filename in ComfyUI's input directory. Upload first
                with upload_image if needed.
            negative_prompt: Things to avoid. Default: "low quality, worst
                quality, deformed, distorted".
            checkpoint: LTX-Video checkpoint. Default: "" (auto-selects).
                Use list_models(folder="checkpoints") to see options.
            text_encoder: T5 text encoder. Default: "" (auto-selects T5-XXL).
                Use list_models(folder="text_encoders") to see options.
            width: Video width in pixels. Default: 768. Must be multiple of 32.
            height: Video height in pixels. Default: 512. Must be multiple of 32.
            length: Number of frames. Default: 97. Must be multiple of 8 + 1
                (e.g. 25, 33, 41, 49, 97).
            steps: Sampling steps. Default: 30. Typical: 20-50.
            cfg: Guidance scale. Default: 3.0. Typical: 1.0-7.0.
            seed: Random seed. Default: -1 (random).
            frame_rate: Output FPS. Default: 25.0.
            strength: How much the video can deviate from the input image.
                Default: 0.85. Range: 0.0 (static) to 1.0 (full motion).

        Returns:
            Dict with prompt_id, status, images, videos, outputs, and workflow_id.
        """
        client: ComfyUIClient = get_client()
        checkpoint = await _resolve_checkpoint(client, checkpoint)
        text_encoder = await _resolve_text_encoder(client, text_encoder)

        wf = workflows.img2video_ltxv(
            prompt=prompt,
            input_image=input_image,
            negative_prompt=negative_prompt,
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            width=width,
            height=height,
            length=length,
            steps=steps,
            cfg=cfg,
            seed=seed,
            frame_rate=frame_rate,
            strength=strength,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "image_to_video")
            result["workflow_id"] = wf_id

        return result

    # ── API node tools (Comfy.org proxy) ────────────────────────────────

    @mcp.tool()
    async def dalle3_image(
        prompt: str,
        quality: str = "standard",
        style: str = "natural",
        size: str = "1024x1024",
        seed: int = -1,
        ctx: Context | None = None,
    ) -> dict:
        """Generate an image using DALL-E 3 via Comfy.org API.

        All parameters except prompt are optional. Requires COMFY_API_KEY.
        For other API providers, use run_api_node() with list_api_nodes().

        Args:
            prompt: Text description of the desired image.
            quality: Image quality. Default: "standard".
                Valid: "standard", "hd".
            style: Style preset. Default: "natural". Valid: "natural", "vivid".
            size: Image dimensions. Default: "1024x1024".
                Valid: "1024x1024", "1024x1792", "1792x1024".
            seed: Random seed. Default: -1 (random).

        Returns:
            Dict with prompt_id, status, images, outputs, and workflow_id.
        """
        client: ComfyUIClient = get_client()

        wf = workflows.dalle3(
            prompt=prompt,
            quality=quality,
            style=style,
            size=size,
            seed=seed,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "dalle3_image")
            result["workflow_id"] = wf_id

        return result

    @mcp.tool()
    async def gpt_image_generate(
        prompt: str,
        quality: str = "low",
        size: str = "auto",
        background: str = "auto",
        n: int = 1,
        model: str = "gpt-image-1.5",
        seed: int = -1,
        input_image: str = "",
        ctx: Context | None = None,
    ) -> dict:
        """Generate or edit an image using GPT Image via Comfy.org API.

        Can generate from text alone, or edit an existing image with a prompt.
        All parameters except prompt are optional. Requires COMFY_API_KEY.

        Args:
            prompt: Text description or edit instruction.
            quality: Image quality. Default: "low".
                Valid: "low", "medium", "high".
            size: Image dimensions. Default: "auto".
                Valid: "auto", "1024x1024", "1024x1536", "1536x1024".
            background: Background style. Default: "auto".
                Valid: "auto", "opaque", "transparent".
            n: Number of images. Default: 1. Range: 1-4.
            model: Model name. Default: "gpt-image-1.5".
                Valid: "gpt-image-1", "gpt-image-1.5".
            seed: Random seed. Default: -1 (random).
            input_image: Filename in ComfyUI input directory for editing.
                Default: "" (none, generate from scratch). Upload first
                with upload_image if needed.

        Returns:
            Dict with prompt_id, status, images, outputs, and workflow_id.
        """
        client: ComfyUIClient = get_client()

        wf = workflows.gpt_image(
            prompt=prompt,
            quality=quality,
            size=size,
            background=background,
            n=n,
            model=model,
            seed=seed,
            input_image=input_image,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "gpt_image_generate")
            result["workflow_id"] = wf_id

        return result

    @mcp.tool()
    async def sora_video_generate(
        prompt: str,
        model: str = "sora-2",
        size: str = "1280x720",
        duration: int = 8,
        seed: int = -1,
        input_image: str = "",
        ctx: Context | None = None,
    ) -> dict:
        """Generate a video using Sora 2 via Comfy.org API.

        Can generate from text alone, or animate an existing image.
        All parameters except prompt are optional. Requires COMFY_API_KEY.

        Args:
            prompt: Text description of the desired video.
            model: Sora model. Default: "sora-2".
                Valid: "sora-2", "sora-2-pro".
            size: Video dimensions. Default: "1280x720".
                Valid: "720x1280", "1280x720", "1024x1792", "1792x1024".
            duration: Video duration in seconds. Default: 8.
                Valid: 4, 8, 12.
            seed: Random seed. Default: -1 (random).
            input_image: Filename in ComfyUI input directory to animate.
                Default: "" (none, text-only). Upload first with upload_image
                if needed.

        Returns:
            Dict with prompt_id, status, videos, outputs, and workflow_id.
        """
        client: ComfyUIClient = get_client()

        wf = workflows.sora_video(
            prompt=prompt,
            model=model,
            size=size,
            duration=duration,
            seed=seed,
            input_image=input_image,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "sora_video_generate")
            result["workflow_id"] = wf_id

        return result

    # ── Video utilities ────────────────────────────────────────────────

    @mcp.tool()
    async def merge_videos(
        video_files: list[str],
        fps: float = 0.0,
        audio_mode: str = "concat",
        output_prefix: str = "ComfyUI_MCP_merged",
        ctx: Context | None = None,
    ) -> dict:
        """Merge multiple videos into a single video.

        Decomposes each video into frames and audio, batches the frames
        together, combines the audio tracks, and reassembles into one video.

        Args:
            video_files: List of video filenames in ComfyUI's input directory,
                in the order they should appear. Must have at least 2 files.
                Upload files first with upload_image if needed.
            fps: Output frames per second. Default: 0.0 (use FPS from the
                first video).
            audio_mode: How to combine audio tracks. Default: "concat".
                "concat" — sequence audio tracks one after another.
                "merge" — overlay/mix all audio tracks together.
                "none" — discard all audio.
            output_prefix: Filename prefix for the saved video. Default:
                "ComfyUI_MCP_merged".

        Returns:
            Dict with prompt_id, status, videos, outputs, and workflow_id.
        """
        if len(video_files) < 2:
            return {"error": "Need at least 2 video files to merge."}

        try:
            wf = workflows.merge_videos(
                video_files=video_files,
                fps=fps,
                audio_mode=audio_mode,
                output_prefix=output_prefix,
            )
        except ValueError as e:
            return {"error": str(e)}

        client: ComfyUIClient = get_client()
        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "merge_videos")
            result["workflow_id"] = wf_id

        return result

    # ── Flux (local model) ───────────────────────────────────────────────

    @mcp.tool()
    async def flux_text_to_image(
        prompt: str,
        diffusion_model: str = "",
        clip_name1: str = "",
        clip_name2: str = "",
        vae_name: str = "ae.safetensors",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        guidance: float = 3.5,
        seed: int = -1,
        batch_size: int = 1,
        ctx: Context | None = None,
    ) -> dict:
        """Generate images using Flux (local model).

        All parameters except prompt are optional with sensible defaults.
        Auto-detects installed Flux model (GGUF or safetensors) and CLIP
        encoders. Produces high-quality 1024x1024 images by default.

        Args:
            prompt: Text description of the desired image.
            diffusion_model: Flux model filename. Default: "" (auto-detects).
                Supports both safetensors and GGUF formats.
                Use list_models(folder="diffusion_models") to see options.
            clip_name1: First CLIP model (clip_l). Default: "" (auto-detects).
            clip_name2: Second CLIP model (t5xxl). Default: "" (auto-detects).
            vae_name: VAE model. Default: "ae.safetensors".
                Use list_models(folder="vae") to see options.
            width: Image width in pixels. Default: 1024. Must be multiple of 16.
            height: Image height in pixels. Default: 1024. Must be multiple of 16.
            steps: Sampling steps. Default: 20. Typical: 15-30.
            guidance: Prompt guidance strength. Default: 3.5. Typical: 2.0-5.0.
                Note: Flux embeds guidance in conditioning, not in CFG.
            seed: Random seed. Default: -1 (random).
            batch_size: Number of images to generate. Default: 1.

        Returns:
            Dict with prompt_id, status, images, outputs, and workflow_id.
        """
        client: ComfyUIClient = get_client()

        use_gguf = False
        if not diffusion_model:
            diffusion_model, use_gguf = await _resolve_flux_model(client)
        else:
            use_gguf = diffusion_model.lower().endswith(".gguf")

        clip_name1, clip_name2 = await _resolve_flux_clips(
            client, clip_name1, clip_name2
        )

        wf = workflows.flux_txt2img(
            prompt=prompt,
            diffusion_model=diffusion_model,
            clip_name1=clip_name1,
            clip_name2=clip_name2,
            vae_name=vae_name,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            batch_size=batch_size,
            use_gguf=use_gguf,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "flux_text_to_image")
            result["workflow_id"] = wf_id

        return result

    # ── Wan 2.2 (local model) ────────────────────────────────────────────

    @mcp.tool()
    async def wan_text_to_video(
        prompt: str,
        negative_prompt: str = "low quality, worst quality, deformed, distorted",
        diffusion_model: str = "",
        clip_name: str = "",
        vae_name: str = "wan2.2_vae.safetensors",
        width: int = 832,
        height: int = 480,
        length: int = 81,
        steps: int = 25,
        cfg: float = 6.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = -1,
        frame_rate: float = 16.0,
        ctx: Context | None = None,
    ) -> dict:
        """Generate video from text using Wan 2.2 (local model).

        All parameters except prompt are optional with sensible defaults.
        Auto-detects installed Wan model and UMT5-XXL text encoder.

        Args:
            prompt: Text description of the desired video.
            negative_prompt: Things to avoid. Default: "low quality, worst
                quality, deformed, distorted".
            diffusion_model: Wan 2.2 GGUF model. Default: "" (auto-detects).
                Use list_models(folder="diffusion_models") to see options.
            clip_name: UMT5-XXL text encoder. Default: "" (auto-detects).
                Use list_models(folder="text_encoders") to see options.
            vae_name: VAE model. Default: "wan2.2_vae.safetensors".
            width: Video width. Default: 832. Must be multiple of 16.
            height: Video height. Default: 480. Must be multiple of 16.
            length: Number of frames. Default: 81. Must be n*4+1
                (e.g. 21, 41, 61, 81). At 16fps, 81 frames ~ 5 seconds.
            steps: Sampling steps. Default: 25. Typical: 20-35.
            cfg: Guidance scale. Default: 6.0. Typical: 3.0-10.0.
            sampler: Sampler algorithm. Default: "euler".
                Use list_samplers_and_schedulers() to see all.
            scheduler: Noise scheduler. Default: "normal".
                Use list_samplers_and_schedulers() to see all.
            seed: Random seed. Default: -1 (random).
            frame_rate: Output FPS. Default: 16.0.

        Returns:
            Dict with prompt_id, status, images, videos, outputs, and
            workflow_id.
        """
        client: ComfyUIClient = get_client()

        if not diffusion_model:
            diffusion_model = await _resolve_wan_model(client)
        if not clip_name:
            clip_name = await _resolve_wan_clip(client)

        wf = workflows.wan_txt2video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            diffusion_model=diffusion_model,
            clip_name=clip_name,
            vae_name=vae_name,
            width=width,
            height=height,
            length=length,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            frame_rate=frame_rate,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "wan_text_to_video")
            result["workflow_id"] = wf_id

        return result

    @mcp.tool()
    async def wan_image_to_video(
        prompt: str,
        input_image: str,
        negative_prompt: str = "low quality, worst quality, deformed, distorted",
        diffusion_model: str = "",
        clip_name: str = "",
        vae_name: str = "wan2.2_vae.safetensors",
        width: int = 832,
        height: int = 480,
        length: int = 81,
        steps: int = 25,
        cfg: float = 6.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = -1,
        frame_rate: float = 16.0,
        ctx: Context | None = None,
    ) -> dict:
        """Generate video from an image using Wan 2.2 (local model).

        All parameters except prompt and input_image are optional with
        sensible defaults. Auto-detects installed Wan model and encoder.

        Args:
            prompt: Text description guiding the video motion and style.
            input_image: Filename in ComfyUI's input directory. Upload first
                with upload_image if needed.
            negative_prompt: Things to avoid. Default: "low quality, worst
                quality, deformed, distorted".
            diffusion_model: Wan 2.2 GGUF model. Default: "" (auto-detects).
            clip_name: UMT5-XXL text encoder. Default: "" (auto-detects).
            vae_name: VAE model. Default: "wan2.2_vae.safetensors".
            width: Video width. Default: 832. Must be multiple of 16.
            height: Video height. Default: 480. Must be multiple of 16.
            length: Number of frames. Default: 81. Must be n*4+1.
            steps: Sampling steps. Default: 25.
            cfg: Guidance scale. Default: 6.0.
            sampler: Sampler algorithm. Default: "euler".
            scheduler: Noise scheduler. Default: "normal".
            seed: Random seed. Default: -1 (random).
            frame_rate: Output FPS. Default: 16.0.

        Returns:
            Dict with prompt_id, status, images, videos, outputs, and
            workflow_id.
        """
        client: ComfyUIClient = get_client()

        if not diffusion_model:
            diffusion_model = await _resolve_wan_model(client)
        if not clip_name:
            clip_name = await _resolve_wan_clip(client)

        wf = workflows.wan_img2video(
            prompt=prompt,
            input_image=input_image,
            negative_prompt=negative_prompt,
            diffusion_model=diffusion_model,
            clip_name=clip_name,
            vae_name=vae_name,
            width=width,
            height=height,
            length=length,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            frame_rate=frame_rate,
        )

        node_cache = get_node_cache() if get_node_cache else None
        result = await _queue_and_wait(client, wf, node_cache)

        if ctx is not None:
            wf_id = await _store_workflow(ctx, wf, "wan_image_to_video")
            result["workflow_id"] = wf_id

        return result

    # ── File operations ─────────────────────────────────────────────────

    @mcp.tool()
    async def upload_image(
        image_path: str,
        subfolder: str = "",
        overwrite: bool = False,
    ) -> dict:
        """Upload a local image file to ComfyUI's input directory.

        Args:
            image_path: Absolute path to the image file on the local filesystem.
            subfolder: Subfolder within input directory. Default: "" (root).
            overwrite: Overwrite existing file. Default: false.

        Returns:
            Dict with name, subfolder, and type of the uploaded file.
        """
        client: ComfyUIClient = get_client()
        return await client.upload_file(
            filepath=image_path,
            subfolder=subfolder,
            overwrite=overwrite,
        )

    @mcp.tool()
    async def upload_mask(
        image_path: str,
        original_ref: str = "",
        subfolder: str = "",
        overwrite: bool = False,
    ) -> dict:
        """Upload a mask image to ComfyUI's input directory.

        The mask is processed to extract the alpha channel for use with
        inpainting workflows. White areas in the mask will be inpainted.

        Args:
            image_path: Absolute path to the mask image file.
            original_ref: Reference to the original image this mask belongs to.
                Default: "" (none).
            subfolder: Subfolder within input directory. Default: "" (root).
            overwrite: Overwrite existing file. Default: false.

        Returns:
            Dict with name, subfolder, and type of the uploaded mask.
        """
        client: ComfyUIClient = get_client()
        return await client.upload_mask(
            filepath=image_path,
            original_ref=original_ref,
            subfolder=subfolder,
            overwrite=overwrite,
        )

    @mcp.tool()
    async def get_image_url(
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> str:
        """Get the URL to view or download a ComfyUI image/video/audio.

        Args:
            filename: Filename (e.g. "ComfyUI_00001_.png").
            subfolder: Subfolder within the directory. Default: "" (root).
            folder_type: Directory type. Default: "output".
                Valid: "output", "input", "temp".

        Returns:
            Full URL to view/download the file.
        """
        client: ComfyUIClient = get_client()
        return client.view_url(
            filename=filename,
            subfolder=subfolder,
            folder_type=folder_type,
        )
