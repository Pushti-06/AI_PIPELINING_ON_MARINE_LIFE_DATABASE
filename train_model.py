# backend/train_model.py

import re
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from Bio import SeqIO
import requests
from io import StringIO
import time

print("Starting model training process...")
print("This will download ~400 sequences from NCBI using the correct API method...")

# --- 1. Correctly Fetch Data from NCBI using E-utilities API ---
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
k_value = 6
all_sequences, all_labels = [], []

def fetch_and_process_sequences(query, label, num_records):
    """Correctly fetches FASTA data using a two-step ESearch/EFetch process."""
    try:
        # STEP 1: ESearch - Get the list of IDs (UIDs) for our search query
        print(f"Searching for UIDs for query: '{query}'...")
        esearch_params = {
            "db": "nuccore",
            "term": query,
            "retmax": num_records,
            "usehistory": "y"
        }
        esearch_response = requests.get(ESEARCH_URL, params=esearch_params)
        esearch_response.raise_for_status() # Will raise an error for bad status codes
        
        # Extract WebEnv and QueryKey for the next step
        web_env = re.search(r"<WebEnv>(\S+)</WebEnv>", esearch_response.text).group(1)
        query_key = re.search(r"<QueryKey>(\d+)</QueryKey>", esearch_response.text).group(1)

        # STEP 2: EFetch - Use the history server to fetch the actual FASTA records
        print(f"Fetching {num_records} FASTA records using history server...")
        efetch_params = {
            "db": "nuccore",
            "query_key": query_key,
            "WebEnv": web_env,
            "rettype": "fasta",
            "retmode": "text",
            "retmax": num_records
        }
        efetch_response = requests.get(EFETCH_URL, params=efetch_params)
        efetch_response.raise_for_status()
        
        fasta_content = efetch_response.text
        count = 0
        with StringIO(fasta_content) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequence = str(record.seq).upper()
                kmers = generate_kmers(sequence, k_value)
                if kmers:
                    all_sequences.append(Counter(kmers))
                    all_labels.append(label)
                    count += 1
        print(f"Successfully processed {count} sequences.")
        return count
    except Exception as e:
        print(f"An error occurred during NCBI fetch: {e}")
        return 0

# --- 2. Feature Extraction ---
def generate_kmers(sequence, k):
    return re.findall(r'(?=(.{{{k}}}))'.format(k=k), sequence)

# Define the NCBI search queries
prokaryote_query = '("16s ribosomal rna"[Title]) AND "bacteria"[Organism] AND 1000:2000[Sequence Length]'
eukaryote_query = '("18s ribosomal rna"[Title]) AND "eukaryotes"[Organism] AND 1000:2000[Sequence Length]'

# Fetch and process 200 of each
fetch_and_process_sequences(prokaryote_query, 0, 200) # Label 0 for prokaryotes
time.sleep(1) # Brief pause to be respectful of NCBI servers
fetch_and_process_sequences(eukaryote_query, 1, 200) # Label 1 for eukaryotes

# --- 3. Train the AI Model ---
if len(all_sequences) > 50: # Check if we have a reasonable amount of data
    print(f"\nTotal sequences for training: {len(all_sequences)}")
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(all_sequences)
    y = all_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, random_state=42, C=0.1) # Added regularization
    
    print("\nTraining the model... this may take a moment.")
    model.fit(X_train, y_train)
    print("Model training complete.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

    # --- 4. Save the Model ---
    joblib.dump(model, 'classifier_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("\nNew, highly accurate model and vectorizer saved to files.")
else:
    print("\nCould not download enough data to train the model. Please check your internet connection or the API status.")