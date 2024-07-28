import numpy as np
import tensorflow as tf
import pennylane as qml
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)

padded_sequences = np.load("padded_sequences.npy")
char_to_int = np.load("char_to_int.npy", allow_pickle=True).item()
int_to_char = np.load("int_to_char.npy", allow_pickle=True).item()
vocab_size = len(char_to_int)

# Reduce dataset size
max_samples = 1000
X = padded_sequences[:max_samples]
y = np.zeros((X.shape[0], vocab_size))
for i, seq in enumerate(X):
    y[i, seq[-1]] = 1

n_qubits = 4  # Reduced from 8 to 4
dev = qml.device("default.qubit", wires=n_qubits)

def strong_ent_layers(n_layers, n_wires):
    for layer in range(n_layers):
        for i in range(n_wires):
            qml.RY(np.random.uniform(0, 2*np.pi), wires=i)
            qml.RZ(np.random.uniform(0, 2*np.pi), wires=i)
        for i in range(n_wires - 1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_wires-1, 0])

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    strong_ent_layers(n_layers=2, n_wires=n_qubits)  # Reduced from 3 to 2
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.qlayer = tf.keras.layers.Lambda(lambda x: tf.stack(quantum_circuit(x), axis=-1))

    def call(self, inputs):
        quantum_output = self.qlayer(inputs)
        return tf.math.real(quantum_output)

def build_hybrid_model(vocab_size, max_length):
    inputs = tf.keras.Input(shape=(max_length,))
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length)(inputs)
    lstm = tf.keras.layers.LSTM(32)(embedding)
    dense = tf.keras.layers.Dense(n_qubits, activation='tanh', kernel_regularizer=l2(0.01))(lstm)
    dense = tf.keras.layers.Lambda(lambda x: x * np.pi)(dense)
    quantum_output = QuantumLayer(n_qubits)(dense)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(quantum_output)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("Building and training Hybrid Model...")
hybrid_model = build_hybrid_model(vocab_size, X.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = hybrid_model.fit(X, y, epochs=20, batch_size=64, validation_split=0.2, verbose=1, callbacks=[early_stopping])

print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

hybrid_model.save("hybrid_model.h5")
print("Model saved successfully.")