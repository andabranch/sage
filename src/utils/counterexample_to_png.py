import numpy as np
from PIL import Image
import re

def parse_counterexample(file_path):
    pixel_values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'X_\d+\s+([-\d.]+)', line)
            if match:
                value = float(match.group(1))
                pixel_values.append(value)

    return pixel_values

def pixels_to_image(pixel_values, width, height):
    try:
        image_array = np.array(pixel_values).reshape((height, width, 3)).astype(np.uint8)
        return Image.fromarray(image_array, 'RGB')
    except ValueError as e:
        print(f"Error reshaping pixel values into image: {e}")
        return None

def main():
    file_path = 'vnnlib/marabou/3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30_model_48_idx_11985_eps_15.00000.counterexample'
    width, height = 48, 48

    pixel_values = parse_counterexample(file_path)
    
    if pixel_values:
        image = pixels_to_image(pixel_values, width, height)
        if image:
            output_path = 'vnnlib/png/marabou/48x48/marabou_counterexample_image15.png'
            image.save(output_path)
            print(f"Image saved to {output_path}")
        else:
            print("Failed to create image from pixel values.")
    else:
        print("No pixel values were extracted. Ensure the file format is correct and matches the expected pattern.")

if __name__ == "__main__":
    main()
