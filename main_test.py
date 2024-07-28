import numpy as np
import tensorflow as tf
import pennylane as qml

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Load your data
padded_sequences = np.load("padded_sequences.npy")
char_to_int = np.load("char_to_int.npy", allow_pickle=True).item()
int_to_char = np.load("int_to_char.npy", allow_pickle=True).item()
vocab_size = len(char_to_int)

X = padded_sequences
y = np.zeros((X.shape[0], vocab_size))
for i, seq in enumerate(padded_sequences):
    y[i, seq[-1]] = 1

# Define the quantum part
n_qubits = 8
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
    strong_ent_layers(n_layers=3, n_wires=n_qubits)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.qlayer = tf.keras.layers.Lambda(lambda x: tf.stack(quantum_circuit(x), axis=-1))

    def call(self, inputs):
        quantum_output = self.qlayer(inputs)
        return tf.math.real(quantum_output)  # Extract the real part

def build_hybrid_model(vocab_size, max_length):
    inputs = tf.keras.Input(shape=(max_length,))
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length)(inputs)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(embedding)
    lstm = tf.keras.layers.Dropout(0.5)(lstm)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(lstm)
    lstm = tf.keras.layers.Dropout(0.5)(lstm)
    dense = tf.keras.layers.Dense(n_qubits, activation='tanh')(lstm)
    dense = tf.keras.layers.Lambda(lambda x: x * np.pi)(dense)
    quantum_output = QuantumLayer(n_qubits)(dense)
    dense = tf.keras.layers.Dense(64, activation='relu')(quantum_output)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the hybrid model
print("Building and training Hybrid Model...")
hybrid_model = build_hybrid_model(vocab_size, X.shape[1])
history = hybrid_model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Print final accuracy
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Save the model
hybrid_model.save("hybrid_model.h5")
print("Model saved successfully.")