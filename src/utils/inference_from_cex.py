import onnxruntime as ort
import numpy as np
import re

def load_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

def load_counterexample(counterexample_path):
    with open(counterexample_path, 'r') as f:
        counterexample_data = f.read()
    return counterexample_data

def parse_smt_counterexample(smt_data):
    pattern = re.compile(r"\(X_\d+\s+([\d\.]+)\)")
    values = pattern.findall(smt_data)
    values = [float(value) for value in values]
    return values

def preprocess_counterexample(values, model_input_shape):
    input_data = np.array(values, dtype=np.float32)
    
    total_elements = np.prod(model_input_shape)
    if len(input_data) != total_elements:
        raise ValueError(f"Cannot reshape array of size {len(input_data)} into shape {model_input_shape}")

    input_data = input_data.reshape((1, *model_input_shape))
    return input_data

def predict(model, input_data):
    input_name = model.get_inputs()[0].name
    input_feed = {input_name: input_data}

    result = model.run(None, input_feed)
    return result

def get_predicted_class_and_probability(prediction):
    predicted_class = np.argmax(prediction)
    probability = np.max(prediction)
    return predicted_class, probability

if __name__ == "__main__":
    model_path = 'models/onnx/3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx'
    counterexample_path = 'output/verification/counterexamples/pyrat/3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30_model_48_idx_11985_eps_15.00000.counterexample'
    model_input_shape = (48, 48, 3)
    
    model = load_model(model_path)
    smt_data = load_counterexample(counterexample_path) 
    values = parse_smt_counterexample(smt_data)
    
    input_data = preprocess_counterexample(values, model_input_shape)
    prediction = predict(model, input_data)
    predicted_class, probability = get_predicted_class_and_probability(prediction[0])
    print(f"Predicted Class: {predicted_class}, Probability: {probability}")
