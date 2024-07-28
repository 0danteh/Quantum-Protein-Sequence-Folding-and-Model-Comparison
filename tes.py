import requests

def fetch_uniprot_sequences(query, format='fasta', limit=100, offset=0):
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        'query': query,
        'format': format,
        'size': limit,
        'offset': offset
    }
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

def extract_sequences_with_headers_from_fasta(fasta_file):
    with open(fasta_file, "r") as file:
        entries = []
        header = ""
        sequence = ""
        for line in file:
            if line.startswith(">"):
                if sequence:
                    entries.append((header, sequence))
                    sequence = ""
                header = line.strip()
            else:
                sequence += line.strip()
        if sequence:
            entries.append((header, sequence))
    return entries

fasta_file = "protein_sequences.fasta"
entries = extract_sequences_with_headers_from_fasta(fasta_file)
print(f"Extracted {len(entries)} entries")

with open("protein_sequences_with_headers.txt", "w") as file:
    for header, seq in entries:
        file.write(header + "\n" + seq + "\n")

query = "reviewed:true" 
total_sequences = 1000

fetch_and_save_sequences(query, total_sequences)
