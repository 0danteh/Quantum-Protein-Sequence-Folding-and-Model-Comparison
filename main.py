import numpy as np
import requests
import time
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as npq
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
import matplotlib.pyplot as plt
import tracemalloc

# Set TensorFlow to eager execution
tf.config.run_functions_eagerly(True)

# Fetch and save sequences
def fetch_uniprot_sequences(query, format='fasta', limit=100, offset=0):
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {'query': query, 'format': format, 'size': limit, 'offset': offset}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        response.raise_for_status()

def save_sequences_to_file(sequences, filename):
    with open(filename, "a") as file:
        file.write(sequences)

def fetch_and_save_sequences(query, total_sequences, batch_size=100, filename="protein_sequences.fasta"):
    for start in range(0, total_sequences, batch_size):
        sequences = fetch_uniprot_sequences(query, limit=batch_size, offset=start)
        save_sequences_to_file(sequences, filename)
        print(f"Fetched and saved sequences {start + 1} to {start + batch_size}")

query = "reviewed:true"
total_sequences = 100
fetch_and_save_sequences(query, total_sequences)

# Extract sequences from FASTA file
def extract_sequences_from_fasta(fasta_file):
    with open(fasta_file, "r") as file:
        sequences = []
        sequence = ""
        for line in file:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return sequences

fasta_file = "protein_sequences.fasta"
sequences = extract_sequences_from_fasta(fasta_file)
print(f"Extracted {len(sequences)} sequences")

with open("protein_sequences_only.txt", "w") as file:
    for seq in sequences:
        file.write(seq + "\n")

# Load and preprocess sequences
def load_sequences(filename):
    with open(filename, "r") as file:
        sequences = file.readlines()
    return [seq.strip() for seq in sequences]

def tokenize_sequences(sequences):
    all_characters = set(''.join(sequences))
    char_to_int = {char: i for i, char in enumerate(all_characters)}
    int_to_char = {i: char for char, i in char_to_int.items()}
    tokenized_sequences = [[char_to_int[char] for char in seq] for seq in sequences]
    return tokenized_sequences, char_to_int, int_to_char

def preprocess_sequences(sequences, max_length):
    tokenized_sequences, char_to_int, int_to_char = tokenize_sequences(sequences)
    padded_sequences = pad_sequences(tokenized_sequences, maxlen=max_length, padding='post')
    return padded_sequences, char_to_int, int_to_char

filename = "protein_sequences_only.txt"
sequences = load_sequences(filename)
max_length = max(len(seq) for seq in sequences)
padded_sequences, char_to_int, int_to_char = preprocess_sequences(sequences, max_length)

np.save("padded_sequences.npy", padded_sequences)
np.save("char_to_int.npy", char_to_int)
np.save("int_to_char.npy", int_to_char)

# Load sequences and prepare data
padded_sequences = np.load("padded_sequences.npy")
char_to_int = np.load("char_to_int.npy", allow_pickle=True).item()
int_to_char = np.load("int_to_char.npy", allow_pickle=True).item()
vocab_size = len(char_to_int)

max_samples = 1000
X = padded_sequences[:max_samples]
y = np.zeros((X.shape[0], vocab_size))
for i, seq in enumerate(X):
    y[i, seq[-1]] = 1

# Quantum Layer and Hybrid Model
n_qubits = 4
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
    strong_ent_layers(n_layers=2, n_wires=n_qubits)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.qlayer = tf.keras.layers.Lambda(lambda x: tf.stack(quantum_circuit(x), axis=-1))

    def call(self, inputs):
        quantum_output = self.qlayer(inputs)
        return tf.cast(tf.math.real(quantum_output), dtype=tf.float32)

def build_hybrid_model(vocab_size, max_length):
    inputs = tf.keras.Input(shape=(max_length,))
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length)(inputs)
    embedding = tf.keras.layers.Dropout(0.2)(embedding)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(embedding)
    lstm = tf.keras.layers.Dropout(0.3)(lstm)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(lstm)
    lstm = tf.keras.layers.Dropout(0.3)(lstm)  
    dense = tf.keras.layers.Dense(n_qubits, activation='tanh', kernel_regularizer=l2(0.01))(lstm)
    dense = tf.keras.layers.Dropout(0.2)(dense)
    dense = tf.keras.layers.Lambda(lambda x: x * np.pi)(dense)
    quantum_output = QuantumLayer(n_qubits)(dense)
    quantum_output = tf.keras.layers.Dropout(0.2)(quantum_output)
    dense = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01))(quantum_output)
    dense = tf.keras.layers.Dropout(0.2)(dense)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(dense)
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

def build_classical_model(vocab_size, max_length):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def benchmark_training(model, X, y, epochs=10, batch_size=32):
    start_time = time.time()
    tracemalloc.start()
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    training_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return history, training_time, current, peak

# Train and evaluate classical model
classical_model = build_classical_model(vocab_size, X.shape[1])
history_classical, training_time_classical, mem_usage_classical, mem_peak_classical = benchmark_training(
    classical_model, X, y, epochs=10, batch_size=32)

print(f"Classical Model Training Time: {training_time_classical:.2f} seconds")
print(f"Classical Model Memory Usage: {mem_usage_classical / 1e6:.2f} MB")
print(f"Classical Model Peak Memory Usage: {mem_peak_classical / 1e6:.2f} MB")
