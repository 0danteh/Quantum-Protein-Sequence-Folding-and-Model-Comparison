import requests
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.callbacks import ModelCheckpoint
import pennylane as qml
from pennylane import numpy as npq
import tensorflow as tf
import tracemalloc

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

n_qubits = 8  # Increased number of qubits
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.weight_shapes = {"weights": (6, n_qubits, 3)}
        self.qlayer = qml.qnn.KerasLayer(quantum_circuit, self.weight_shapes, output_dim=n_qubits)

    def call(self, inputs):
        return tf.cast(self.qlayer(inputs), dtype=tf.float32)

def build_hybrid_model(vocab_size, max_length):
    inputs = tf.keras.Input(shape=(max_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length)(inputs)
    lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding)
    lstm = Dropout(0.5)(lstm)
    lstm = Bidirectional(LSTM(64))(lstm)
    lstm = Dropout(0.5)(lstm)
    dense = Dense(n_qubits, activation='tanh')(lstm)
    dense = tf.keras.layers.Lambda(lambda x: x * np.pi)(dense)
    quantum_output = QuantumLayer(n_qubits)(dense)
    dense = Dense(64, activation='relu')(quantum_output)
    outputs = Dense(vocab_size, activation='softmax')(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def benchmark_training(model, X, y, epochs=10, batch_size=32):
    start_time = time.time()
    tracemalloc.start()
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    training_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    return training_time, current / 10**6, peak / 10**6, accuracy, val_accuracy

padded_sequences = np.load("padded_sequences.npy")
char_to_int = np.load("char_to_int.npy", allow_pickle=True).item()
int_to_char = np.load("int_to_char.npy", allow_pickle=True).item()
vocab_size = len(char_to_int)

X = padded_sequences
y = np.zeros((X.shape[0], vocab_size))
for i, seq in enumerate(padded_sequences):
    y[i, seq[-1]] = 1

# Classical Model Training
print("Training Classical Model...")
model = build_classical_model(vocab_size, X.shape[1])
checkpoint = ModelCheckpoint("classical_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')
training_time, current_memory, peak_memory, accuracy, val_accuracy = benchmark_training(model, X, y, epochs=10, batch_size=32)

print(f"Classical Model Training Time: {training_time} seconds")
print(f"Classical Model Memory Usage: Current = {current_memory} MB, Peak = {peak_memory} MB")
print(f"Classical Model Accuracy: {accuracy * 100:.2f}%")
print(f"Classical Model Validation Accuracy: {val_accuracy * 100:.2f}%")

# Hybrid Model Training
print("\nTraining Hybrid Model...")
hybrid_model = build_hybrid_model(vocab_size, X.shape[1])
checkpoint_hybrid = ModelCheckpoint("hybrid_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')
training_time, current_memory, peak_memory, accuracy, val_accuracy = benchmark_training(hybrid_model, X, y, epochs=10, batch_size=32)

print(f"Hybrid Model Training Time: {training_time} seconds")
print(f"Hybrid Model Memory Usage: Current = {current_memory} MB, Peak = {peak_memory} MB")
print(f"Hybrid Model Accuracy: {accuracy * 100:.2f}%")
print(f"Hybrid Model Validation Accuracy: {val_accuracy * 100:.2f}%")

# Function to generate a sequence
def generate_sequence(model, seed_sequence, max_length):
    generated_sequence = seed_sequence.copy()
    for _ in range(max_length - len(seed_sequence)):
        x = pad_sequences([generated_sequence], maxlen=max_length, padding='post')
        pred = model.predict(x, verbose=0)[0]
        next_char_index = np.argmax(pred)
        generated_sequence.append(next_char_index)
    return generated_sequence

# Generate sequences using both models
print("\nGenerating sequences...")
seed_sequence = padded_sequences[0][:10].tolist()  # Use the first 10 characters of the first sequence as seed
print("Seed sequence:", ''.join([int_to_char[i] for i in seed_sequence]))

classical_generated = generate_sequence(model, seed_sequence, max_length)
hybrid_generated = generate_sequence(hybrid_model, seed_sequence, max_length)

print("Classical Model Generated Sequence:")
print(''.join([int_to_char[i] for i in classical_generated]))

print("\nHybrid Model Generated Sequence:")
print(''.join([int_to_char[i] for i in hybrid_generated]))

# Compare the generated sequences
print("\nComparing generated sequences:")
for i, (c, h) in enumerate(zip(classical_generated, hybrid_generated)):
    if c != h:
        print(f"Difference at position {i}: Classical '{int_to_char[c]}', Hybrid '{int_to_char[h]}'")

# Calculate and print the similarity between the generated sequences
similarity = sum(1 for c, h in zip(classical_generated, hybrid_generated) if c == h) / len(classical_generated)
print(f"\nSimilarity between generated sequences: {similarity * 100:.2f}%")

# Save the trained models
model.save("classical_model.h5")
hybrid_model.save("hybrid_model.h5")
print("\nModels saved successfully.")

# Function to evaluate model on test data
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return loss, accuracy

# Split data into train and test sets
test_split = 0.2
split_index = int(len(X) * (1 - test_split))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Evaluate both models on test data
print("\nEvaluating models on test data...")
classical_loss, classical_accuracy = evaluate_model(model, X_test, y_test)
hybrid_loss, hybrid_accuracy = evaluate_model(hybrid_model, X_test, y_test)

print(f"Classical Model - Test Loss: {classical_loss:.4f}, Test Accuracy: {classical_accuracy * 100:.2f}%")
print(f"Hybrid Model - Test Loss: {hybrid_loss:.4f}, Test Accuracy: {hybrid_accuracy * 100:.2f}%")

# Compare model sizes
classical_params = model.count_params()
hybrid_params = hybrid_model.count_params()

print(f"\nClassical Model Parameters: {classical_params}")
print(f"Hybrid Model Parameters: {hybrid_params}")
print(f"Difference in Parameters: {abs(classical_params - hybrid_params)}")

print("\nExperiment completed.")