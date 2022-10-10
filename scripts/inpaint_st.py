import sys
import numpy as np
import streamlit as st
from PIL import Image
from omegaconf import OmegaConf
from main import instantiate_from_config
from streamlit_drawable_canvas import st_canvas
import torch
from ldm.models.diffusion.ddim import DDIMSampler


@st.cache(allow_output_mutation=True)
def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "image": image.to(device=device),
            "txt": [txt],
            "mask": mask.to(device=device),
            "masked_image": masked_image.to(device=device),
            }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(1, 4, 64, 64)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device)

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [1, 4, 64, 64]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond={"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(1, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, 64, 64]
            samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    1,
                    shape,
                    cond,
                    verbose=False,
                    eta=0.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0,2,3,1)[0]*255
            result = Image.fromarray(result.astype(np.uint8))
    return result


def run():
    st.title("Stable Diffusion Inpainting")
    
    sampler = initialize_model(sys.argv[1], sys.argv[2])

    image = st.file_uploader("Image", ["jpg", "png"])
    if image:
        image = Image.open(image)
        image = image.resize((512, 512))
        width, height = image.size

        prompt = st.text_input("Prompt")

        seed = st.number_input("Seed", min_value=0, max_value=1000000, value=0)
        scale = st.slider("Scale", min_value=0.1, max_value=30.0, value=7.5, step=0.1)
        ddim_steps = st.slider("DDIM Steps", min_value=0, max_value=50, value=50, step=1)

        fill_color = "rgba(255, 255, 255, 0.0)"
        stroke_width = st.number_input("Brush Size",
                                       value=64,
                                       min_value=1,
                                       max_value=100)
        stroke_color = "rgba(255, 255, 255, 1.0)"
        bg_color = "rgba(0, 0, 0, 1.0)"
        drawing_mode = "freedraw"

        st.write("Canvas")
        st.caption("Draw a mask to inpaint, then click the 'Send to Streamlit' button (bottom left, with an arrow on it).")
        canvas_result = st_canvas(
            fill_color=fill_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=image,
            update_streamlit=False,
            height=height,
            width=width,
            drawing_mode=drawing_mode,
            key="canvas",
        )
        if canvas_result:
            mask = canvas_result.image_data
            mask = mask[:, :, -1] > 0
            if mask.sum() > 0:
                mask = Image.fromarray(mask)

                result = inpaint(
                    sampler=sampler,
                    image=image,
                    mask=mask,
                    prompt=prompt,
                    seed=seed,
                    scale=scale,
                    ddim_steps=ddim_steps,
                )
                st.write("Inpainted")
                st.image(result)


if __name__ == "__main__":
    run()
