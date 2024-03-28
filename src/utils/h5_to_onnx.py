import tensorflow as tf
import tf2onnx
import onnx

model_path_h5 = "output/3QConv/models/48x48/3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.h5"
keras_model = tf.keras.models.load_model(model_path_h5)

onnx_model_path = "vnnlib/model/3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx"

spec = (tf.TensorSpec((None, 48, 48, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=13)

onnx.save(model_proto, onnx_model_path)

print(f"Model successfully converted to ONNX and saved at {onnx_model_path}")
