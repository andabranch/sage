import os
import numpy as np
from PIL import Image
from maraboupy import Marabou, MarabouNetworkONNX
import pandas as pd
import torch
VNN_PATH = ["/Users/jesss/Downloads/model_30_idx_11985_eps_1.00000.vnnlib"]
MODEL_PATHS = ["output/best/models/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx"]
IMAGE_IDS = ["11985"]
EPSILONS = [1.0]
IMG_RESIZE = (30, 30)
IMAGE_PATH = "datasets/GTSRB_dataset/Test"

def load_image(image_id):
    image_path = os.path.join(IMAGE_PATH, f"{image_id}.png")
    image = Image.open(image_path)
    image = image.resize(IMG_RESIZE)
    return np.array(image).astype(np.float32) / 255.0

def verify_model_robustness(model_path,vnn_path, image, epsilon):
    network = Marabou.read_onnx(model_path, vnn_path)
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]
    print(inputVars)
    flattened_image = image.flatten()

    for i in range(len(flattened_image)):
        network.setLowerBound(inputVars[i], max(0, flattened_image[i] - epsilon))
        network.setUpperBound(inputVars[i], min(1, flattened_image[i] + epsilon))

    adversarial_example, stats = network.solve(verbose=False)
    return adversarial_example == {}

def main():
    results = []
    for model_path in MODEL_PATHS:
        for vnn_path in VNN_PATH:
            for image_id in IMAGE_IDS:
                image = load_image(image_id)
                for epsilon in EPSILONS:
                    is_robust = verify_model_robustness(model_path,vnn_path, image, epsilon)
                    results.append((model_path, image_id, epsilon, is_robust))

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results, columns=["Model", "ImageID", "Epsilon", "IsRobust"])
    results_df.to_csv("robustness_verification_results.csv", index=False)
    print("Robustness verification completed. Results saved to 'robustness_verification_results.csv'.")

if __name__ == "__main__":
    main()
