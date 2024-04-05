import larq as lq
import larq_compute_engine as lce
import tensorflow as tf

SIZE = 30
OUTPUT_PATH = 'output/best/models/'

# test_name = '3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.h5'
# test_name = '3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.h5'
test_name = '3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.h5'
model = tf.keras.models.load_model(
    OUTPUT_PATH + test_name)

# with lq.context.quantized_scope(True):
#     model.save(OUTPUT_PATH + "binary/" + test_name)

# Convert our Keras model to a TFLite flatbuffer file
with open(OUTPUT_PATH + "binary/" + test_name + ".tflite", "wb") as flatbuffer_file:
    flatbuffer_bytes = lce.convert_keras_model(model)
    flatbuffer_file.write(flatbuffer_bytes)
