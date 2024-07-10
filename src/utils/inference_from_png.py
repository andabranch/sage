import onnxruntime as ort
import numpy as np
from PIL import Image
import os

def load_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

def preprocess_image(image_path, model_input_size):
    image = Image.open(image_path).convert('RGB') 
    image = image.resize(model_input_size)

    image = np.array(image, dtype=np.float32)

    image = np.expand_dims(image, axis=0)
    return image


def predict(model, image):
    input_name = model.get_inputs()[0].name
    input_data = {input_name: image}

    result = model.run(None, input_data)
    return result

def load_specific_images(image_ids, image_dir, model_input_size):
    images = []
    for image_id in image_ids:
        image_path = os.path.join(image_dir, f"{image_id}.png") 
        preprocessed_image = preprocess_image(image_path, model_input_size)
        images.append(preprocessed_image)
    return images

def get_predicted_class_and_probability(prediction):
    predicted_class = np.argmax(prediction)
    probability = np.max(prediction)
    return predicted_class, probability


if __name__ == "__main__":
    model_path = 'models/onnx/3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx'
    #image_ids = ["07040", "08258", "11985"]
    image_ids = ["model_48_idx_11985_eps_15.00000"]

    image_dir = "output/verification/img_from_counterexamples/new_alphabeta/48x48"
    model_input_size = (48, 48)

    model = load_model(model_path)
    images = load_specific_images(image_ids, image_dir, model_input_size)

for image in images:
    prediction = predict(model, image)
    predicted_class, probability = get_predicted_class_and_probability(prediction[0])
    print(f"Predicted Class: {predicted_class}, Probability: {probability}")
