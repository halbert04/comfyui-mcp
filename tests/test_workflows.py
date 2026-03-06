"""Tests for workflow builder."""

from comfyui_mcp.workflows import WorkflowBuilder, img2img, inpaint, txt2img, upscale


class TestWorkflowBuilder:
    def test_add_node_increments_id(self):
        wb = WorkflowBuilder()
        id1 = wb.add_node("Foo", {"a": 1})
        id2 = wb.add_node("Bar", {"b": 2})
        assert id1 == "1"
        assert id2 == "2"

    def test_link_format(self):
        assert WorkflowBuilder.link("3", 1) == ["3", 1]
        assert WorkflowBuilder.link("1") == ["1", 0]

    def test_build_returns_all_nodes(self):
        wb = WorkflowBuilder()
        wb.add_node("A", {})
        wb.add_node("B", {})
        result = wb.build()
        assert "1" in result
        assert "2" in result
        assert result["1"]["class_type"] == "A"


class TestTxt2Img:
    def test_basic_structure(self):
        wf = txt2img(prompt="a cat", checkpoint="model.safetensors", seed=42)
        # Should have 8 nodes without LoRA
        assert len(wf) == 7  # ckpt, pos, neg, latent, sampler, decode, save

    def test_has_checkpoint_loader(self):
        wf = txt2img(prompt="a cat", checkpoint="model.safetensors", seed=42)
        ckpt_nodes = [n for n in wf.values() if n["class_type"] == "CheckpointLoaderSimple"]
        assert len(ckpt_nodes) == 1
        assert ckpt_nodes[0]["inputs"]["ckpt_name"] == "model.safetensors"

    def test_has_sampler(self):
        wf = txt2img(prompt="a cat", checkpoint="model.safetensors", seed=42)
        sampler_nodes = [n for n in wf.values() if n["class_type"] == "KSampler"]
        assert len(sampler_nodes) == 1
        inputs = sampler_nodes[0]["inputs"]
        assert inputs["seed"] == 42
        assert inputs["denoise"] == 1.0

    def test_with_lora(self):
        wf = txt2img(
            prompt="a cat",
            checkpoint="model.safetensors",
            lora_name="detail.safetensors",
            lora_strength=0.8,
            seed=42,
        )
        # Should have 8 nodes with LoRA
        assert len(wf) == 8
        lora_nodes = [n for n in wf.values() if n["class_type"] == "LoraLoader"]
        assert len(lora_nodes) == 1
        assert lora_nodes[0]["inputs"]["lora_name"] == "detail.safetensors"
        assert lora_nodes[0]["inputs"]["strength_model"] == 0.8

    def test_vae_from_checkpoint(self):
        wf = txt2img(
            prompt="a cat",
            checkpoint="model.safetensors",
            lora_name="lora.safetensors",
            seed=42,
        )
        decode_nodes = [n for n in wf.values() if n["class_type"] == "VAEDecode"]
        assert len(decode_nodes) == 1
        # VAE link should point to checkpoint node (id "1"), output index 2
        vae_link = decode_nodes[0]["inputs"]["vae"]
        assert vae_link == ["1", 2]

    def test_save_image_prefix(self):
        wf = txt2img(prompt="a cat", checkpoint="m.safetensors", seed=42)
        save_nodes = [n for n in wf.values() if n["class_type"] == "SaveImage"]
        assert len(save_nodes) == 1
        assert save_nodes[0]["inputs"]["filename_prefix"] == "ComfyUI_MCP"

    def test_prompt_text_forwarded(self):
        wf = txt2img(prompt="hello", negative_prompt="bad", checkpoint="m.safetensors", seed=42)
        clip_nodes = [n for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
        texts = {n["inputs"]["text"] for n in clip_nodes}
        assert "hello" in texts
        assert "bad" in texts


class TestImg2Img:
    def test_basic_structure(self):
        wf = img2img(prompt="a cat", input_image="photo.png", checkpoint="m.safetensors", seed=42)
        # ckpt, pos, neg, loadimage, vaeencode, sampler, decode, save
        assert len(wf) == 8

    def test_has_load_image(self):
        wf = img2img(prompt="a cat", input_image="photo.png", checkpoint="m.safetensors", seed=42)
        load_nodes = [n for n in wf.values() if n["class_type"] == "LoadImage"]
        assert len(load_nodes) == 1
        assert load_nodes[0]["inputs"]["image"] == "photo.png"

    def test_denoise_value(self):
        wf = img2img(
            prompt="a cat", input_image="photo.png", checkpoint="m.safetensors",
            denoise=0.6, seed=42,
        )
        sampler_nodes = [n for n in wf.values() if n["class_type"] == "KSampler"]
        assert sampler_nodes[0]["inputs"]["denoise"] == 0.6


class TestUpscale:
    def test_basic_structure(self):
        wf = upscale(input_image="photo.png", upscale_model="RealESRGAN_x4.pth")
        # model loader, load image, upscale, save
        assert len(wf) == 4

    def test_model_name(self):
        wf = upscale(input_image="photo.png", upscale_model="RealESRGAN_x4.pth")
        loader = [n for n in wf.values() if n["class_type"] == "UpscaleModelLoader"]
        assert loader[0]["inputs"]["model_name"] == "RealESRGAN_x4.pth"


class TestInpaint:
    def test_basic_structure(self):
        wf = inpaint(
            prompt="fill sky",
            input_image="photo.png",
            mask_image="mask.png",
            checkpoint="m.safetensors",
            seed=42,
        )
        # ckpt, load_img, load_mask, vae_encode_inpaint, pos, neg, sampler, decode, save
        assert len(wf) == 9

    def test_mask_link(self):
        wf = inpaint(
            prompt="fill",
            input_image="photo.png",
            mask_image="mask.png",
            checkpoint="m.safetensors",
            seed=42,
        )
        encode_nodes = [n for n in wf.values() if n["class_type"] == "VAEEncodeForInpaint"]
        assert len(encode_nodes) == 1
        # mask comes from LoadImage node for mask, output index 1
        mask_link = encode_nodes[0]["inputs"]["mask"]
        assert mask_link[1] == 1  # MASK is output index 1 of LoadImage

    def test_grow_mask_by(self):
        wf = inpaint(
            prompt="fill",
            input_image="photo.png",
            mask_image="mask.png",
            checkpoint="m.safetensors",
            grow_mask_by=12,
            seed=42,
        )
        encode_nodes = [n for n in wf.values() if n["class_type"] == "VAEEncodeForInpaint"]
        assert encode_nodes[0]["inputs"]["grow_mask_by"] == 12
