import os
import cv2
import numpy as np

def concat_image(origin_image_dir,
                      remove_black_image_dir,
                      outpainted_dir,
                      output_dir):
    
    cnt = 0
    for img in os.listdir(outpainted_dir):
        if cnt > 200:
            break
        
        cnt += 1
        img_name1 = "_".join(img.split("_")[3:])
        image_1 = cv2.imread(os.path.join(origin_image_dir, img_name1))
        img_name2 = "_".join(img.split("_")[1:])
        image_2 = cv2.imread(os.path.join(remove_black_image_dir, img_name2))
        image_3 = cv2.imread(os.path.join(outpainted_dir, img))

        concatenated_image = np.concatenate((image_1, image_2, image_3), axis=1)

        cv2.imwrite(os.path.join(output_dir, img), concatenated_image)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    origin_image_dir = "/home/duanyuxuan/dms_gen/warp/result_outpainted"
    remove_black_image_dir = "./output/black_remove"
    outpainted_dir = "./output/outpainted_black_remove"
    output_dir = "./output/grid0608"

    os.makedirs(output_dir, exist_ok=True)

    concat_image(origin_image_dir, remove_black_image_dir, outpainted_dir, output_dir)
