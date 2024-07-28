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
from pennylane.optimize import AdamOptimizer
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

def build_hybrid_model(vocab_size, max_length):
    inputs = tf.keras.Input(shape=(max_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length)(inputs)
    lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    lstm = Dropout(0.5)(lstm)
    lstm = Bidirectional(LSTM(64))(lstm)
    lstm = Dropout(0.5)(lstm)
    quantum_output = quantum_layer(lstm)
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

model = build_classical_model(vocab_size, X.shape[1])
checkpoint = ModelCheckpoint("model.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')
training_time, current_memory, peak_memory, accuracy, val_accuracy = benchmark_training(model, X, y, epochs=10, batch_size=32)

print(f"Classical Model Training Time: {training_time} seconds")
print(f"Classical Model Memory Usage: Current = {current_memory} MB, Peak = {peak_memory} MB")
print(f"Classical Model Accuracy: {accuracy * 100:.2f}%")
print(f"Classical Model Validation Accuracy: {val_accuracy * 100:.2f}%")

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (6, n_qubits, 3)}
quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

hybrid_model = build_hybrid_model(vocab_size, X.shape[1])
checkpoint_hybrid = ModelCheckpoint("hybrid_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')
training_time, current_memory, peak_memory, accuracy, val_accuracy = benchmark_training(hybrid_model, X, y, epochs=10, batch_size=32)

print(f"Hybrid Model Training Time: {training_time} seconds")
print(f"Hybrid Model Memory Usage: Current = {current_memory} MB, Peak = {peak_memory} MB")
print(f"Hybrid Model Accuracy: {accuracy * 100:.2f}%")
print(f"Hybrid Model Validation Accuracy: {val_accuracy * 100:.2f}%")
