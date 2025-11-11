import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import keras    
from keras import layers
import numpy as np 
import matplotlib.pyplot as plt
print("---------------------------------------------")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

def build_model(activation, learning_rate):
    model = keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(3, activation=activation, name='hidden_layer'),
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    loss = "mse"    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

activations = ['relu', 'sigmoid', 'linear']

histories = {}

epochs = 500
learning_rate = 0.01
all_preds = {}
for activation in activations:
    model : keras.Sequential = build_model(activation, learning_rate)
    
    history : keras.src.callbacks.history.History = model.fit(X, y, epochs=epochs, verbose=0)
    histories[activation] = history.history['loss']
    y_pred = model.predict(X)
    print(f"Activation: {activation}")
    print("Probability:")
    print(np.round(y_pred,2))
    Y_preds_bin = (y_pred >= 0.5).astype(int)
    print("Predictions:")    
    print(Y_preds_bin)
    all_preds[activation] = Y_preds_bin
    print("Weights:")
    for layer in model.layers:
        weights, biases = layer.get_weights()
        print(f"Weights: {weights}, Biases: {biases}")

for i in all_preds:
    print()
    print(i)
    print(all_preds[i])


plt.figure(figsize=(8,6))
for activation in activations:
    plt.plot(histories[activation], label=f'loss ({activation})')
plt.title('Loss during training for different activations')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
