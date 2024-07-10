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
    file_path = 'output/verification/counterexamples/new_alphabeta/30x30/model_30_idx_8258_eps_5.00000.txt'
    width, height = 30, 30

    pixel_values = parse_counterexample(file_path)
    
    if pixel_values:
        image = pixels_to_image(pixel_values, width, height)
        if image:
            output_path = 'output/verification/img_from_counterexamples/new_alphabeta/test/model_30_idx_8258_eps_5.00000.png'
            image.save(output_path)
            print(f"Image saved to {output_path}")
        else:
            print("Failed to create image from pixel values.")
    else:
        print("No pixel values were extracted. Ensure the file format is correct and matches the expected pattern.")

if __name__ == "__main__":
    main()
