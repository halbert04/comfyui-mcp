"""Workflow builder for constructing ComfyUI prompt graphs."""

from __future__ import annotations

import random
from typing import Any


class WorkflowBuilder:
    """Incrementally build a ComfyUI workflow (prompt dict)."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def add_node(self, class_type: str, inputs: dict[str, Any]) -> str:
        """Add a node and return its string ID."""
        self._counter += 1
        node_id = str(self._counter)
        self._nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    @staticmethod
    def link(node_id: str, output_index: int = 0) -> list:
        """Return a link reference: [node_id, output_index]."""
        return [node_id, output_index]

    def build(self) -> dict[str, dict[str, Any]]:
        """Return the complete workflow dict."""
        return dict(self._nodes)


def _resolve_seed(seed: int) -> int:
    if seed < 0:
        return random.randint(0, 2**63 - 1)
    return seed


def _resolve_seed_api(seed: int) -> int:
    """Resolve seed for API nodes (32-bit int max: 2147483647)."""
    if seed < 0:
        return random.randint(0, 2**31 - 1)
    return min(seed, 2**31 - 1)


def txt2img(
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
) -> dict:
    """Build a text-to-image workflow."""
    wb = WorkflowBuilder()
    seed = _resolve_seed(seed)

    # 1: CheckpointLoaderSimple
    ckpt_id = wb.add_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint})

    # Model and CLIP sources default to checkpoint
    model_src = ckpt_id
    clip_src = ckpt_id

    # 2: Optional LoRA
    if lora_name:
        lora_id = wb.add_node("LoraLoader", {
            "lora_name": lora_name,
            "strength_model": lora_strength,
            "strength_clip": lora_strength,
            "model": wb.link(ckpt_id, 0),
            "clip": wb.link(ckpt_id, 1),
        })
        model_src = lora_id
        clip_src = lora_id

    # 3: Positive prompt
    pos_id = wb.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": wb.link(clip_src, 1 if clip_src == ckpt_id else 1),
    })

    # 4: Negative prompt
    neg_id = wb.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": wb.link(clip_src, 1 if clip_src == ckpt_id else 1),
    })

    # 5: Empty latent
    latent_id = wb.add_node("EmptyLatentImage", {
        "width": width,
        "height": height,
        "batch_size": batch_size,
    })

    # 6: KSampler
    sampler_id = wb.add_node("KSampler", {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": 1.0,
        "model": wb.link(model_src, 0),
        "positive": wb.link(pos_id, 0),
        "negative": wb.link(neg_id, 0),
        "latent_image": wb.link(latent_id, 0),
    })

    # 7: VAE Decode — VAE always from checkpoint (index 2)
    decode_id = wb.add_node("VAEDecode", {
        "samples": wb.link(sampler_id, 0),
        "vae": wb.link(ckpt_id, 2),
    })

    # 8: Save
    wb.add_node("SaveImage", {
        "images": wb.link(decode_id, 0),
        "filename_prefix": "ComfyUI_MCP",
    })

    return wb.build()


def img2img(
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
) -> dict:
    """Build an image-to-image workflow."""
    wb = WorkflowBuilder()
    seed = _resolve_seed(seed)

    # 1: Checkpoint
    ckpt_id = wb.add_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint})

    model_src = ckpt_id
    clip_src = ckpt_id

    # 2: Optional LoRA
    if lora_name:
        lora_id = wb.add_node("LoraLoader", {
            "lora_name": lora_name,
            "strength_model": lora_strength,
            "strength_clip": lora_strength,
            "model": wb.link(ckpt_id, 0),
            "clip": wb.link(ckpt_id, 1),
        })
        model_src = lora_id
        clip_src = lora_id

    # 3: Positive prompt
    pos_id = wb.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": wb.link(clip_src, 1 if clip_src == ckpt_id else 1),
    })

    # 4: Negative prompt
    neg_id = wb.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": wb.link(clip_src, 1 if clip_src == ckpt_id else 1),
    })

    # 5: Load input image
    load_id = wb.add_node("LoadImage", {"image": input_image})

    # 6: VAE Encode
    encode_id = wb.add_node("VAEEncode", {
        "pixels": wb.link(load_id, 0),
        "vae": wb.link(ckpt_id, 2),
    })

    # 7: KSampler
    sampler_id = wb.add_node("KSampler", {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": denoise,
        "model": wb.link(model_src, 0),
        "positive": wb.link(pos_id, 0),
        "negative": wb.link(neg_id, 0),
        "latent_image": wb.link(encode_id, 0),
    })

    # 8: VAE Decode
    decode_id = wb.add_node("VAEDecode", {
        "samples": wb.link(sampler_id, 0),
        "vae": wb.link(ckpt_id, 2),
    })

    # 9: Save
    wb.add_node("SaveImage", {
        "images": wb.link(decode_id, 0),
        "filename_prefix": "ComfyUI_MCP",
    })

    return wb.build()


def upscale(
    input_image: str,
    upscale_model: str = "",
) -> dict:
    """Build an upscale workflow."""
    wb = WorkflowBuilder()

    # 1: Upscale model loader
    model_id = wb.add_node("UpscaleModelLoader", {"model_name": upscale_model})

    # 2: Load image
    load_id = wb.add_node("LoadImage", {"image": input_image})

    # 3: Upscale
    up_id = wb.add_node("ImageUpscaleWithModel", {
        "upscale_model": wb.link(model_id, 0),
        "image": wb.link(load_id, 0),
    })

    # 4: Save
    wb.add_node("SaveImage", {
        "images": wb.link(up_id, 0),
        "filename_prefix": "ComfyUI_MCP_upscale",
    })

    return wb.build()


def txt2video_ltxv(
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
) -> dict:
    """Build an LTX-Video text-to-video workflow.

    Uses the advanced sampling pipeline: CLIPLoader (ltxv type),
    LTXVConditioning, LTXVScheduler, SamplerCustom.
    """
    wb = WorkflowBuilder()
    seed = _resolve_seed(seed)

    # 1: CheckpointLoaderSimple → MODEL (index 0) + VAE (index 2)
    #    Note: CLIP comes from separate CLIPLoader for LTXV
    ckpt_id = wb.add_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint})

    # 2: CLIPLoader with type "ltxv" → CLIP (index 0)
    clip_id = wb.add_node("CLIPLoader", {
        "clip_name": text_encoder,
        "type": "ltxv",
    })

    # 3: Positive prompt
    pos_id = wb.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": wb.link(clip_id, 0),
    })

    # 4: Negative prompt
    neg_id = wb.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": wb.link(clip_id, 0),
    })

    # 5: LTXVConditioning — sets frame_rate on conditioning
    cond_id = wb.add_node("LTXVConditioning", {
        "positive": wb.link(pos_id, 0),
        "negative": wb.link(neg_id, 0),
        "frame_rate": frame_rate,
    })

    # 6: EmptyLTXVLatentVideo
    latent_id = wb.add_node("EmptyLTXVLatentVideo", {
        "width": width,
        "height": height,
        "length": length,
        "batch_size": 1,
    })

    # 7: LTXVScheduler — generates sigmas from latent dimensions
    sched_id = wb.add_node("LTXVScheduler", {
        "steps": steps,
        "max_shift": 2.05,
        "base_shift": 0.95,
        "stretch": True,
        "terminal": 0.1,
        "latent": wb.link(latent_id, 0),
    })

    # 8: KSamplerSelect — pick sampler algorithm
    sampler_sel_id = wb.add_node("KSamplerSelect", {
        "sampler_name": "res_multistep",
    })

    # 9: SamplerCustom — advanced sampler using sigmas
    sample_id = wb.add_node("SamplerCustom", {
        "add_noise": True,
        "noise_seed": seed,
        "cfg": cfg,
        "model": wb.link(ckpt_id, 0),
        "positive": wb.link(cond_id, 0),   # positive from LTXVConditioning
        "negative": wb.link(cond_id, 1),   # negative from LTXVConditioning
        "sampler": wb.link(sampler_sel_id, 0),
        "sigmas": wb.link(sched_id, 0),
        "latent_image": wb.link(latent_id, 0),
    })

    # 10: VAE Decode — VAE from checkpoint (index 2)
    decode_id = wb.add_node("VAEDecode", {
        "samples": wb.link(sample_id, 0),
        "vae": wb.link(ckpt_id, 2),
    })

    # 11: SaveAnimatedWEBP — output as animated webp
    wb.add_node("SaveAnimatedWEBP", {
        "images": wb.link(decode_id, 0),
        "filename_prefix": "ComfyUI_MCP_video",
        "fps": frame_rate,
        "lossless": False,
        "quality": 90,
        "method": "default",
    })

    return wb.build()


def img2video_ltxv(
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
) -> dict:
    """Build an LTX-Video image-to-video workflow.

    Uses LTXVImgToVideo to condition video generation on an input image.
    """
    wb = WorkflowBuilder()
    seed = _resolve_seed(seed)

    # 1: CheckpointLoaderSimple → MODEL (0), CLIP (1), VAE (2)
    ckpt_id = wb.add_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint})

    # 2: CLIPLoader with type "ltxv" → CLIP (0)
    clip_id = wb.add_node("CLIPLoader", {
        "clip_name": text_encoder,
        "type": "ltxv",
    })

    # 3: Positive prompt
    pos_id = wb.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": wb.link(clip_id, 0),
    })

    # 4: Negative prompt
    neg_id = wb.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": wb.link(clip_id, 0),
    })

    # 5: LTXVConditioning — sets frame_rate on conditioning
    cond_id = wb.add_node("LTXVConditioning", {
        "positive": wb.link(pos_id, 0),
        "negative": wb.link(neg_id, 0),
        "frame_rate": frame_rate,
    })

    # 6: LoadImage
    img_id = wb.add_node("LoadImage", {"image": input_image})

    # 7: LTXVImgToVideo — conditions on image, produces modified conditioning + latent
    i2v_id = wb.add_node("LTXVImgToVideo", {
        "positive": wb.link(cond_id, 0),
        "negative": wb.link(cond_id, 1),
        "vae": wb.link(ckpt_id, 2),
        "image": wb.link(img_id, 0),
        "width": width,
        "height": height,
        "length": length,
        "batch_size": 1,
        "strength": strength,
    })
    # LTXVImgToVideo outputs: [CONDITIONING(0), CONDITIONING(1), LATENT(2)]

    # 8: LTXVScheduler — generates sigmas
    sched_id = wb.add_node("LTXVScheduler", {
        "steps": steps,
        "max_shift": 2.05,
        "base_shift": 0.95,
        "stretch": True,
        "terminal": 0.1,
        "latent": wb.link(i2v_id, 2),
    })

    # 9: KSamplerSelect
    sampler_sel_id = wb.add_node("KSamplerSelect", {
        "sampler_name": "res_multistep",
    })

    # 10: SamplerCustom — uses conditioned latent from LTXVImgToVideo
    sample_id = wb.add_node("SamplerCustom", {
        "add_noise": True,
        "noise_seed": seed,
        "cfg": cfg,
        "model": wb.link(ckpt_id, 0),
        "positive": wb.link(i2v_id, 0),
        "negative": wb.link(i2v_id, 1),
        "sampler": wb.link(sampler_sel_id, 0),
        "sigmas": wb.link(sched_id, 0),
        "latent_image": wb.link(i2v_id, 2),
    })

    # 11: VAE Decode
    decode_id = wb.add_node("VAEDecode", {
        "samples": wb.link(sample_id, 0),
        "vae": wb.link(ckpt_id, 2),
    })

    # 12: SaveAnimatedWEBP
    wb.add_node("SaveAnimatedWEBP", {
        "images": wb.link(decode_id, 0),
        "filename_prefix": "ComfyUI_MCP_i2v",
        "fps": frame_rate,
        "lossless": False,
        "quality": 90,
        "method": "default",
    })

    return wb.build()


def inpaint(
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
) -> dict:
    """Build an inpainting workflow."""
    wb = WorkflowBuilder()
    seed = _resolve_seed(seed)

    # 1: Checkpoint
    ckpt_id = wb.add_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint})

    # 2: Load input image
    img_id = wb.add_node("LoadImage", {"image": input_image})

    # 3: Load mask image
    mask_id = wb.add_node("LoadImage", {"image": mask_image})

    # 4: VAE Encode for inpaint
    encode_id = wb.add_node("VAEEncodeForInpaint", {
        "pixels": wb.link(img_id, 0),
        "vae": wb.link(ckpt_id, 2),
        "mask": wb.link(mask_id, 1),  # mask output at index 1
        "grow_mask_by": grow_mask_by,
    })

    # 5: Positive prompt
    pos_id = wb.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": wb.link(ckpt_id, 1),
    })

    # 6: Negative prompt
    neg_id = wb.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": wb.link(ckpt_id, 1),
    })

    # 7: KSampler
    sampler_id = wb.add_node("KSampler", {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": denoise,
        "model": wb.link(ckpt_id, 0),
        "positive": wb.link(pos_id, 0),
        "negative": wb.link(neg_id, 0),
        "latent_image": wb.link(encode_id, 0),
    })

    # 8: VAE Decode
    decode_id = wb.add_node("VAEDecode", {
        "samples": wb.link(sampler_id, 0),
        "vae": wb.link(ckpt_id, 2),
    })

    # 9: Save
    wb.add_node("SaveImage", {
        "images": wb.link(decode_id, 0),
        "filename_prefix": "ComfyUI_MCP_inpaint",
    })

    return wb.build()


# ── API node workflows ───────────────────────────────────────────────


def dalle3(
    prompt: str,
    quality: str = "standard",
    style: str = "natural",
    size: str = "1024x1024",
    seed: int = -1,
) -> dict:
    """Build a DALL-E 3 image generation workflow."""
    wb = WorkflowBuilder()
    seed = _resolve_seed_api(seed)

    dalle_id = wb.add_node("OpenAIDalle3", {
        "prompt": prompt,
        "seed": seed,
        "quality": quality,
        "style": style,
        "size": size,
    })

    wb.add_node("SaveImage", {
        "images": wb.link(dalle_id, 0),
        "filename_prefix": "ComfyUI_MCP_dalle3",
    })

    return wb.build()


def gpt_image(
    prompt: str,
    quality: str = "low",
    size: str = "auto",
    background: str = "auto",
    n: int = 1,
    model: str = "gpt-image-1.5",
    seed: int = -1,
    input_image: str = "",
) -> dict:
    """Build a GPT Image generation/editing workflow."""
    wb = WorkflowBuilder()
    seed = _resolve_seed_api(seed)

    inputs: dict[str, Any] = {
        "prompt": prompt,
        "seed": seed,
        "quality": quality,
        "size": size,
        "background": background,
        "n": n,
        "model": model,
    }

    if input_image:
        img_id = wb.add_node("LoadImage", {"image": input_image})
        inputs["image"] = wb.link(img_id, 0)

    gpt_id = wb.add_node("OpenAIGPTImage1", inputs)

    wb.add_node("SaveImage", {
        "images": wb.link(gpt_id, 0),
        "filename_prefix": "ComfyUI_MCP_gpt_image",
    })

    return wb.build()


def sora_video(
    prompt: str,
    model: str = "sora-2",
    size: str = "1280x720",
    duration: int = 8,  # 4, 8, or 12
    seed: int = -1,
    input_image: str = "",
) -> dict:
    """Build a Sora 2 video generation workflow."""
    wb = WorkflowBuilder()
    seed = _resolve_seed_api(seed)

    inputs: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "duration": duration,
        "seed": seed,
    }

    if input_image:
        img_id = wb.add_node("LoadImage", {"image": input_image})
        inputs["image"] = wb.link(img_id, 0)

    sora_id = wb.add_node("OpenAIVideoSora2", inputs)

    wb.add_node("SaveVideo", {
        "video": wb.link(sora_id, 0),
        "filename_prefix": "ComfyUI_MCP_sora",
        "format": "mp4",
        "codec": "h264",
    })

    return wb.build()


def merge_videos(
    video_files: list[str],
    fps: float = 0.0,
    audio_mode: str = "concat",
    output_prefix: str = "ComfyUI_MCP_merged",
) -> dict:
    """Build a workflow that merges multiple videos into one.

    Pipeline: LoadVideo → GetVideoComponents (per clip) → ImageBatch frames →
    AudioConcat/AudioMerge audio → CreateVideo → SaveVideo.

    Args:
        video_files: List of video filenames in ComfyUI's input directory, in order.
        fps: Output FPS. Default: 0.0 (use FPS from first video).
        audio_mode: How to combine audio tracks. "concat" = sequence them,
            "merge" = overlay/mix them, "none" = discard audio.
        output_prefix: Filename prefix for saved output.
    """
    if len(video_files) < 2:
        raise ValueError("Need at least 2 video files to merge.")

    wb = WorkflowBuilder()

    # Load and decompose each video
    frame_ids: list[str] = []  # GetVideoComponents node IDs (output 0 = IMAGE)
    audio_ids: list[str] = []  # GetVideoComponents node IDs (output 1 = AUDIO)
    fps_source_id: str | None = None

    for vf in video_files:
        load_id = wb.add_node("LoadVideo", {"file": vf})
        decomp_id = wb.add_node("GetVideoComponents", {
            "video": wb.link(load_id, 0),
        })
        frame_ids.append(decomp_id)
        audio_ids.append(decomp_id)
        if fps_source_id is None:
            fps_source_id = decomp_id  # first video's FPS (output index 2)

    # Batch all frames together using ImageBatch (pairwise chain)
    current_frames = frame_ids[0]  # output index 0 = IMAGE
    current_frames_idx = 0
    for i in range(1, len(frame_ids)):
        batch_id = wb.add_node("ImageBatch", {
            "image1": wb.link(current_frames, current_frames_idx),
            "image2": wb.link(frame_ids[i], 0),
        })
        current_frames = batch_id
        current_frames_idx = 0

    # Combine audio
    audio_output = None
    if audio_mode != "none" and len(audio_ids) >= 2:
        current_audio = audio_ids[0]
        current_audio_idx = 1  # GetVideoComponents output 1 = AUDIO
        for i in range(1, len(audio_ids)):
            if audio_mode == "merge":
                combine_id = wb.add_node("AudioMerge", {
                    "audio1": wb.link(current_audio, current_audio_idx),
                    "audio2": wb.link(audio_ids[i], 1),
                    "merge_method": "add",
                })
            else:  # concat
                combine_id = wb.add_node("AudioConcat", {
                    "audio1": wb.link(current_audio, current_audio_idx),
                    "audio2": wb.link(audio_ids[i], 1),
                    "direction": "after",
                })
            current_audio = combine_id
            current_audio_idx = 0
        audio_output = (current_audio, current_audio_idx)

    # CreateVideo from batched frames + combined audio
    create_inputs: dict[str, Any] = {
        "images": wb.link(current_frames, current_frames_idx),
    }
    # FPS: use explicit value or wire from first video's GetVideoComponents (index 2)
    if fps > 0:
        create_inputs["fps"] = fps
    else:
        create_inputs["fps"] = wb.link(fps_source_id, 2)

    if audio_output is not None:
        create_inputs["audio"] = wb.link(audio_output[0], audio_output[1])

    video_id = wb.add_node("CreateVideo", create_inputs)

    # Save
    wb.add_node("SaveVideo", {
        "video": wb.link(video_id, 0),
        "filename_prefix": output_prefix,
        "format": "auto",
        "codec": "auto",
    })

    return wb.build()


# ── Flux workflows ──────────────────────────────────────────────────


def flux_txt2img(
    prompt: str,
    diffusion_model: str = "",
    clip_name1: str = "clip_l.safetensors",
    clip_name2: str = "t5xxl_fp8_e4m3fn.safetensors",
    vae_name: str = "ae.safetensors",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    guidance: float = 3.5,
    seed: int = -1,
    batch_size: int = 1,
    use_gguf: bool = False,
) -> dict:
    """Build a Flux text-to-image workflow.

    Supports both safetensors (UNETLoader) and GGUF (UnetLoaderGGUF) models.
    Uses DualCLIPLoader, CLIPTextEncodeFlux, Flux2Scheduler, SamplerCustom.
    """
    wb = WorkflowBuilder()
    seed = _resolve_seed(seed)

    # 1: DualCLIPLoader → CLIP
    clip_id = wb.add_node("DualCLIPLoader", {
        "clip_name1": clip_name1,
        "clip_name2": clip_name2,
        "type": "flux",
    })

    # 2: CLIPTextEncodeFlux — positive prompt
    pos_id = wb.add_node("CLIPTextEncodeFlux", {
        "clip": wb.link(clip_id, 0),
        "clip_l": prompt,
        "t5xxl": prompt,
        "guidance": guidance,
    })

    # 3: CLIPTextEncodeFlux — negative (empty, Flux ignores negatives)
    neg_id = wb.add_node("CLIPTextEncodeFlux", {
        "clip": wb.link(clip_id, 0),
        "clip_l": "",
        "t5xxl": "",
        "guidance": guidance,
    })

    # 4: Load diffusion model
    if use_gguf:
        model_id = wb.add_node("UnetLoaderGGUF", {
            "unet_name": diffusion_model,
        })
    else:
        model_id = wb.add_node("UNETLoader", {
            "unet_name": diffusion_model,
            "weight_dtype": "default",
        })

    # 5: VAELoader
    vae_id = wb.add_node("VAELoader", {"vae_name": vae_name})

    # 6: EmptyFlux2LatentImage
    latent_id = wb.add_node("EmptyFlux2LatentImage", {
        "width": width,
        "height": height,
        "batch_size": batch_size,
    })

    # 7: Flux2Scheduler — resolution-aware sigma schedule
    sched_id = wb.add_node("Flux2Scheduler", {
        "steps": steps,
        "width": width,
        "height": height,
    })

    # 8: KSamplerSelect
    sampler_sel_id = wb.add_node("KSamplerSelect", {
        "sampler_name": "euler",
    })

    # 9: SamplerCustom — uses Flux2Scheduler sigmas
    sample_id = wb.add_node("SamplerCustom", {
        "add_noise": True,
        "noise_seed": seed,
        "cfg": 1.0,  # Flux uses guidance in conditioning, not CFG
        "model": wb.link(model_id, 0),
        "positive": wb.link(pos_id, 0),
        "negative": wb.link(neg_id, 0),
        "sampler": wb.link(sampler_sel_id, 0),
        "sigmas": wb.link(sched_id, 0),
        "latent_image": wb.link(latent_id, 0),
    })

    # 10: VAEDecode
    decode_id = wb.add_node("VAEDecode", {
        "samples": wb.link(sample_id, 0),
        "vae": wb.link(vae_id, 0),
    })

    # 11: SaveImage
    wb.add_node("SaveImage", {
        "images": wb.link(decode_id, 0),
        "filename_prefix": "ComfyUI_MCP_flux",
    })

    return wb.build()


# ── Wan 2.2 workflows ───────────────────────────────────────────────


def wan_txt2video(
    prompt: str,
    negative_prompt: str = "low quality, worst quality, deformed, distorted",
    diffusion_model: str = "",
    clip_name: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
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
) -> dict:
    """Build a Wan 2.2 text-to-video workflow.

    Pipeline: UnetLoaderGGUF → CLIPLoader(wan) → VAELoader →
    CLIPTextEncode → WanImageToVideo → KSampler → VAEDecode →
    SaveAnimatedWEBP.
    """
    wb = WorkflowBuilder()
    seed = _resolve_seed(seed)

    # 1: UnetLoaderGGUF — load Wan model
    model_id = wb.add_node("UnetLoaderGGUF", {
        "unet_name": diffusion_model,
    })

    # 2: CLIPLoader with type "wan"
    clip_id = wb.add_node("CLIPLoader", {
        "clip_name": clip_name,
        "type": "wan",
    })

    # 3: VAELoader
    vae_id = wb.add_node("VAELoader", {"vae_name": vae_name})

    # 4: Positive prompt
    pos_id = wb.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": wb.link(clip_id, 0),
    })

    # 5: Negative prompt
    neg_id = wb.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": wb.link(clip_id, 0),
    })

    # 6: WanImageToVideo (no start_image = text-only mode)
    # Outputs: [CONDITIONING(0), CONDITIONING(1), LATENT(2)]
    wan_id = wb.add_node("WanImageToVideo", {
        "positive": wb.link(pos_id, 0),
        "negative": wb.link(neg_id, 0),
        "vae": wb.link(vae_id, 0),
        "width": width,
        "height": height,
        "length": length,
        "batch_size": 1,
    })

    # 7: KSampler
    sampler_id = wb.add_node("KSampler", {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": 1.0,
        "model": wb.link(model_id, 0),
        "positive": wb.link(wan_id, 0),
        "negative": wb.link(wan_id, 1),
        "latent_image": wb.link(wan_id, 2),
    })

    # 8: VAEDecode
    decode_id = wb.add_node("VAEDecode", {
        "samples": wb.link(sampler_id, 0),
        "vae": wb.link(vae_id, 0),
    })

    # 9: SaveAnimatedWEBP
    wb.add_node("SaveAnimatedWEBP", {
        "images": wb.link(decode_id, 0),
        "filename_prefix": "ComfyUI_MCP_wan",
        "fps": frame_rate,
        "lossless": False,
        "quality": 90,
        "method": "default",
    })

    return wb.build()


def wan_img2video(
    prompt: str,
    input_image: str,
    negative_prompt: str = "low quality, worst quality, deformed, distorted",
    diffusion_model: str = "",
    clip_name: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
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
) -> dict:
    """Build a Wan 2.2 image-to-video workflow.

    Same as txt2video but passes a start_image to WanImageToVideo.
    """
    wb = WorkflowBuilder()
    seed = _resolve_seed(seed)

    # 1: UnetLoaderGGUF
    model_id = wb.add_node("UnetLoaderGGUF", {
        "unet_name": diffusion_model,
    })

    # 2: CLIPLoader
    clip_id = wb.add_node("CLIPLoader", {
        "clip_name": clip_name,
        "type": "wan",
    })

    # 3: VAELoader
    vae_id = wb.add_node("VAELoader", {"vae_name": vae_name})

    # 4: Positive prompt
    pos_id = wb.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": wb.link(clip_id, 0),
    })

    # 5: Negative prompt
    neg_id = wb.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": wb.link(clip_id, 0),
    })

    # 6: LoadImage
    img_id = wb.add_node("LoadImage", {"image": input_image})

    # 7: WanImageToVideo with start_image
    wan_id = wb.add_node("WanImageToVideo", {
        "positive": wb.link(pos_id, 0),
        "negative": wb.link(neg_id, 0),
        "vae": wb.link(vae_id, 0),
        "width": width,
        "height": height,
        "length": length,
        "batch_size": 1,
        "start_image": wb.link(img_id, 0),
    })

    # 8: KSampler
    sampler_id = wb.add_node("KSampler", {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": 1.0,
        "model": wb.link(model_id, 0),
        "positive": wb.link(wan_id, 0),
        "negative": wb.link(wan_id, 1),
        "latent_image": wb.link(wan_id, 2),
    })

    # 9: VAEDecode
    decode_id = wb.add_node("VAEDecode", {
        "samples": wb.link(sampler_id, 0),
        "vae": wb.link(vae_id, 0),
    })

    # 10: SaveAnimatedWEBP
    wb.add_node("SaveAnimatedWEBP", {
        "images": wb.link(decode_id, 0),
        "filename_prefix": "ComfyUI_MCP_wan_i2v",
        "fps": frame_rate,
        "lossless": False,
        "quality": 90,
        "method": "default",
    })

    return wb.build()
