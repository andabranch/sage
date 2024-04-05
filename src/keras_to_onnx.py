from keras.models import load_model
import keras2onnx
import onnx
#protobuf 3.19.6
OUTPUT_PATH = 'output/best/models/'
NAME = '3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30'

model = load_model(OUTPUT_PATH + NAME + 'h5')
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, OUTPUT_PATH + 'onnx/' + NAME + '.onnx')