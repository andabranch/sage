import tensorflow as tf
import larq as lq


def load_and_print_weights_and_config(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'QuantConv2D': lq.layers.QuantConv2D, 'QuantDense': lq.layers.QuantDense})

    for layer in model.layers:
        if isinstance(layer, (lq.layers.QuantDense, lq.layers.QuantConv2D)):
            weights = layer.get_weights()[0] 
            unique_values = tf.unique(tf.reshape(weights, [-1])).y.numpy()
            print(f'Layer {layer.name} unique weights: {unique_values}')
            print(f'Layer {layer.name} config: {layer.get_config()["kernel_quantizer"]}')

#+ de adaugat intr un fisier + de facut TOATE weights
model_path = '/Users/jesss/Documents/SAGE/BinarizedNeuralNetwork/output/German/models/30x30/3_30_30_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_43_ep_30.h5'
load_and_print_weights_and_config(model_path)
