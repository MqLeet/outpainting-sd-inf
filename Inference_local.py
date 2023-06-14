from diffusers import StableDiffusionInpaintPipeline

from PIL import Image, ImageOps, ImageDraw
import numpy as np
import torch
import os, glob
from diffusers import StableDiffusionInpaintPipeline
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def default_loader(path):
    return Image.open(path).convert('RGB')


def run_outpainting(pipe, path_image, path_output, 
                new_width=768, new_height=512):
    pipe.to("cuda")

    images = glob.glob(os.path.join(path_image, '*.png'))

    for image_name in images:
        image = default_loader(image_name)
            
        # load the image, extract the mask
        padding = (new_width - image.width) // 2

        # Create the new image for outpainting
        new_image = Image.new("RGB", (new_width, new_height), "white")
        new_image.paste(image, (padding, 0))

        # Create the mask for outpainting
        mask_np = np.ones((new_height, new_width))  # Start with white
        mask_np[:, padding:new_width-padding] = 0  # Draw black rectangle on the original part
        mask = Image.fromarray(mask_np * 255).convert('RGB')


        # run the pipeline
        prompt = "A person is driving."
        # image and mask_image should be PIL images.
        # The mask structure is white for outpainting and black for keeping as is
        outpainted_image = pipe(
            prompt=prompt,
            image=new_image,
            mask_image=mask,
            height=new_height,
            width=new_width
        ).images[0]

        image_name = "outpainted_" + image_name.split("/")[-1]
        outpainted_image.save(os.path.join(path_output, image_name))



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--new_width",
                    type=int,
                    default=768)
    
    parser.add_argument("--model",
                    type=str,
                    default="stable-diffusion-2-inpainting",
                    choices=["stable-diffusion-2-inpainting", "stable-diffusion-inpainting"])

    parser.add_argument("--path_image",
                    type=str,
                    default="/home/duanyuxuan/dms_gen/face_mask_cmp/iou_filtered_dreambooth_dmd_ids/images_0.7",
                    help="path of images which need to be outpainted")
    
    parser.add_argument("--path_output",
                    type=str,
                    default="./output")

    return parser.parse_args()
def main(args):

    if args.model == "stable-diffusion-2-inpainting":
        model_name = "stabilityai/stable-diffusion-2-inpainting"
    
    elif args.model == "stable-diffusion-inpainting":
        model_name = "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        requires_safety_checker= False
    )

    os.makedirs(args.path_output, exist_ok=True)
    run_outpainting(pipe, args.path_image, args.path_output, new_width=args.new_width)


if __name__ == "__main__":
    args = parse_args()
    main(args)