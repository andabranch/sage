import onnxruntime as ort
import numpy as np
from PIL import Image

model_path = "output/best/models/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

def preprocess_image(image_path, size=(30, 30), epsilon=10.0/255):
    image = Image.open(image_path)
    image = image.resize(size)
    image = np.array(image).astype(np.float32) / 255.0
    perturbation = np.random.uniform(-epsilon, epsilon, image.shape)
    image += perturbation
    image = np.clip(image, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image

def predict_class(image, session, input_name):
    result = session.run(None, {input_name: image})
    predicted_class = np.argmax(result[0], axis=1)
    return predicted_class

def visualize_perturbed_image(image_array):
    image_array = np.squeeze(image_array, axis=0)
    perturbed_image = (image_array * 255).astype(np.uint8)
    perturbed_image = Image.fromarray(perturbed_image)
    perturbed_image.show()
    #perturbed_image.save(save_path)
    
image_path = 'vnnlib/png/pyrat/30x30/pyrat_counterexample_image4.png'
image_array = preprocess_image(image_path)

predicted_class = predict_class(image_array, session, input_name)
print(f"Predicted class with epsilon perturbation: {predicted_class}")

save_path = 'datasets/GTSRB_dataset/Perturbed'
#visualize_perturbed_image(image_array)
