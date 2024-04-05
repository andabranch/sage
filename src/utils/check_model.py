import onnx

model_path = 'output/best/models/onnx/3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx3'
model = onnx.load(model_path)
onnx.checker.check_model(model)
print("The model is valid!")
