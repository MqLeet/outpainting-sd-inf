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

    images = glob.glob(os.path.join(path_image, '*.jpg'))

    for image_name in images:
        image = default_loader(image_name)
        
        mask_np = np.ones((new_height, new_width))  # Start with white


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


def get_inscribed_rect(path_image, path_output):

    images = glob.glob(os.path.join(path_image, '*_cropped.jpg'))
    for image_name in images:
        image = default_loader(image_name)
        img_np = np.array(image)
        non_black_pixels = np.any(img_np != [0, 0, 0], axis=2)
        first_nonblack_pixel = 0
        last_nonblack_pixel = image.width
        for idx, row in enumerate(non_black_pixels):
            if idx == 0 or idx == len(non_black_pixels) - 1:
                first_true_index = np.argmax(row)
                last_true_index =  np.where(row)[0][-1]

                    

                if first_true_index >= first_nonblack_pixel:
                    first_nonblack_pixel = first_true_index
                if last_true_index <= last_nonblack_pixel:
                    last_nonblack_pixel = last_true_index

        left = first_nonblack_pixel
        right = last_nonblack_pixel

        cropped_img_np = img_np[:, left+10:right-10, :]


        crop_img = Image.fromarray(cropped_img_np).convert('RGB')

        image_name = "remove_black_" + image_name.split("/")[-1]
        crop_img.save(os.path.join(path_output, image_name))





def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--new_width",
                    type=int,
                    default=1280)

    parser.add_argument("--new_height",
                    type=int,
                    default=720)
    
    parser.add_argument("--model",
                    type=str,
                    default="stable-diffusion-2-inpainting",
                    choices=["stable-diffusion-2-inpainting", "stable-diffusion-inpainting"])

    parser.add_argument("--path_image",
                    type=str,
                    default="/home/duanyuxuan/dms_gen/warp/result_outpainted",
                    help="path of images which need to be outpainted")
    
    parser.add_argument("--path_output",
                    type=str,
                    default="./output/black_remove")

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

    path_output = "./output/outpainted_black_remove"
    os.makedirs(path_output, exist_ok=True)
    path_image = "./output/black_remove"
    run_outpainting(pipe, path_image, path_output, new_width=args.new_width, new_height=args.new_height)

    # get_inscribed_rect(args.path_image, args.path_output)


if __name__ == "__main__":
    args = parse_args()
    main(args)