import requests

def fetch_pdb_data(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    return response.text

pdb_id = "1TUP"  # Example PDB ID
pdb_data = fetch_pdb_data(pdb_id)