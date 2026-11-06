import keras

def make_mlp(hidden_layers: list[dict] = None, output_layers: list[dict] = None):
    """
    Create a multi-layer perceptron (MLP) model.

    Args:
        hidden_layers (list of dict): A list of dictionaries specifying the configuration of hidden layers.
            Each dictionary should have the keys 'neurons' (int) and 'activation' (str).
        output_layers (list of dict): A list of dictionaries specifying the configuration of output layers.
            Each dictionary should have the keys 'neurons' (int) and 'activation' (str).

    Returns:
        keras.Sequential: A Keras Sequential model representing the MLP.

    """
    model = keras.Sequential()

    # Add hidden layers
    for layer_config in hidden_layers:
        neurons = layer_config.get('neurons')
        activation = layer_config.get('activation')
        model.add(keras.layers.Dense(neurons, activation=activation))

    # Add output layers
    for layer_config in output_layers:
        neurons = layer_config.get('neurons')
        activation = layer_config.get('activation')
        model.add(keras.layers.Dense(neurons, activation=activation))

    return model