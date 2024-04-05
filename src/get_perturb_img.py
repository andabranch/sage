from PIL import Image
import random
import numpy as np

dataset_path = "datasets/GTSRB_dataset/Test/"
result_path = "perturb_2/"
img_name = "00001.png"
start = 6010

def change_pixel_value(image_path, constant_value, nr_img):
    for k in range(start, start + nr_img):
        image = Image.open(dataset_path + image_path)
        # Get the pixel data
        pixels = image.load()
        # Iterate over each pixel
        for i in range(image.width):
            for j in range(image.height):
                r, g, b = pixels[i, j]

                # Change the RGB values by the constant value
                r += random.randint(-constant_value, constant_value)
                g += random.randint(-constant_value, constant_value)
                b += random.randint(-constant_value, constant_value)

                # Update the pixel with the new RGB values
                pixels[i, j] = (r, g, b)

        # Save the modified image
        modified_image_path = result_path + str(k) + "_" + image_path
        image.save(modified_image_path)

        if k % 10 == 0:
            print("Generated img = " + str(k))

    return True


def change_pixel_value_partial(image_path, constant_value, nr_img):
    for k in range(start, start+nr_img):
        image = Image.open(dataset_path + image_path)
        # Get the pixel data
        pixels = image.load()
        # Iterate over each pixel
        for i in range(image.width//4, image.width-image.width//4):
            for j in range(image.height//4, image.height-image.height//4):
                r, g, b = pixels[i, j]

                # Change the RGB values by the constant value
                r += constant_value if random.random() < 0.5 else -constant_value
                g += constant_value if random.random() < 0.5 else -constant_value
                b += constant_value if random.random() < 0.5 else -constant_value

                # Update the pixel with the new RGB values
                pixels[i, j] = (r, g, b)

        # Save the modified image
        modified_image_path = result_path + str(k) + "_" + image_path
        image.save(modified_image_path)

        if k % 10 == 0:
            print("Generated img = " + str(k))

    return True


def change_pixel_value_extreme(image_path, offset):
    # Open the image
    image = Image.open(dataset_path + image_path)

    # Convert the image to a numpy array for efficient computation
    image_array = np.array(image)

    # Generate all possible combinations of offset
    combinations = [(offset, offset, offset),
                    (-offset, offset, offset),
                    (offset, -offset, offset),
                    (offset, offset, -offset),
                    (-offset, -offset, offset),
                    (-offset, offset, -offset),
                    (offset, -offset, -offset),
                    (-offset, -offset, -offset)]

    new_images = []

    # Iterate over each combination
    idx = 1
    for combination in combinations:
        # Create a new image by adding/subtracting the offset from the RGB channels
        new_image_array = np.clip(image_array + combination, 0, 255)
        new_image = Image.fromarray(new_image_array.astype(np.uint8))

        modified_image_path = result_path + str(idx+start) + "_" + image_path
        new_image.save(modified_image_path)
        idx += 1

        # Append the new image to the list
        new_images.append(new_image)

    return new_images


def change_pixel_value_partial_extreme(image_path, constant_value, nr_img):
    for k in range(start, start+nr_img):
        image = Image.open(dataset_path + image_path)
        # Get the pixel data
        pixels = image.load()
        # Iterate over each pixel
        random_red = random.random()
        random_blue = random.random()
        random_green = random.random()
        for i in range(image.width//3, image.width-image.width//2):
            for j in range(0, image.height):
                r, g, b = pixels[i, j]

                # Change the RGB values by the constant value
                r += constant_value if random_red < 0.5 else -constant_value
                g += constant_value if random_green < 0.5 else -constant_value
                b += constant_value if random_blue < 0.5 else -constant_value

                # Update the pixel with the new RGB values
                pixels[i, j] = (r, g, b)

        # Save the modified image
        modified_image_path = result_path + str(k) + "_" + image_path
        image.save(modified_image_path)

        if k % 10 == 0:
            print("Generated img = " + str(k))

    return True


#change_pixel_value(img_name, 15, 5000)
change_pixel_value_extreme(img_name, 15)
#change_pixel_value_partial(img_name, 15, 1000)
change_pixel_value_partial_extreme(img_name, 15, 10)
