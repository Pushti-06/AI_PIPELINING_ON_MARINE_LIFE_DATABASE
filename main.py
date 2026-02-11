from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from Bio import SeqIO
from collections import Counter, defaultdict
import re
from io import StringIO
import uvicorn
import numpy as np

# Import machine learning and scientific libraries
from sklearn.feature_extraction import DictVectorizer
from hdbscan import HDBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import Biopython's BLAST libraries
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

# 1. Create the FastAPI app object
app = FastAPI(
    title="Advanced eDNA Analysis with Unsupervised Learning",
    description="An API that uses unsupervised learning to predict unknown organisms based on k-mer similarity."
)

# 2. Add CORS middleware - THIS WAS MISSING IN ORIGINAL CODE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define taxonomic classification helper functions
def extract_taxonomy_from_blast_title(blast_title):
    """Extracts basic taxonomic information from BLAST hit title."""
    taxonomy = {
        "kingdom": "Unknown",
        "genus": "Unknown", 
        "species": "Unknown"
    }
    
    title_lower = blast_title.lower()
    
    # Kingdom-level classification with more comprehensive patterns
    if any(word in title_lower for word in ['bacteria', 'bacterial', 'bacillus', 'streptococcus', 'escherichia', 'pseudomonas', 'staphylococcus', 'lactobacillus']):
        taxonomy["kingdom"] = "Bacteria"
    elif any(word in title_lower for word in ['fungi', 'fungal', 'yeast', 'candida', 'aspergillus', 'penicillium', 'saccharomyces']):
        taxonomy["kingdom"] = "Fungi"
    elif any(word in title_lower for word in ['plant', 'arabidopsis', 'oryza', 'triticum', 'chloroplast', 'plantae']):
        taxonomy["kingdom"] = "Plantae"
    elif any(word in title_lower for word in ['animal', 'homo', 'mus', 'drosophila', 'caenorhabditis', 'mitochondrion', 'animalia']):
        taxonomy["kingdom"] = "Animalia"
    elif any(word in title_lower for word in ['virus', 'viral', 'phage']):
        taxonomy["kingdom"] = "Viruses"
    elif any(word in title_lower for word in ['archaea', 'archaeal', 'methanococcus', 'thermococcus']):
        taxonomy["kingdom"] = "Archaea"
    
    # Extract genus and species
    words = blast_title.split()
    for i in range(len(words)-1):
        if (words[i][0].isupper() and len(words[i]) > 2 and 
            words[i+1][0].islower() and len(words[i+1]) > 2 and
            words[i].isalpha() and words[i+1].isalpha()):
            taxonomy["genus"] = words[i]
            taxonomy["species"] = f"{words[i]} {words[i+1]}"
            break
    
    return taxonomy

def calculate_kmer_similarity(kmer_dict1, kmer_dict2):
    """Calculate cosine similarity between two k-mer dictionaries."""
    # Get all unique k-mers from both dictionaries
    all_kmers = set(kmer_dict1.keys()) | set(kmer_dict2.keys())
    
    # Convert to vectors
    vec1 = np.array([kmer_dict1.get(kmer, 0) for kmer in all_kmers])
    vec2 = np.array([kmer_dict2.get(kmer, 0) for kmer in all_kmers])
    
    # Calculate cosine similarity
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def predict_unknown_taxonomy(unknown_kmer_data, known_sequences_data, min_confidence_threshold=0.1):
    """
    Uses unsupervised learning to predict taxonomy of unknown sequences.
    Always provides a best guess, even for very dissimilar sequences.
    """
    predictions = []
    
    for unknown_kmers in unknown_kmer_data:
        best_matches = []
        
        # Compare unknown sequence to all known sequences
        for seq_id, data in known_sequences_data.items():
            known_kmers = data['kmer_data']
            taxonomy = data['taxonomy']
            
            similarity = calculate_kmer_similarity(unknown_kmers, known_kmers)
            best_matches.append({
                'similarity': similarity,
                'kingdom': taxonomy['kingdom'],
                'genus': taxonomy['genus']
            })
        
        # Sort by similarity
        best_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        if best_matches and best_matches[0]['similarity'] > min_confidence_threshold:
            # Count kingdom predictions from top matches
            kingdom_votes = defaultdict(list)
            top_matches = best_matches[:min(10, len(best_matches))]
            
            for match in top_matches:
                if match['similarity'] > 0:
                    kingdom_votes[match['kingdom']].append(match['similarity'])
            
            # Calculate weighted average confidence for each kingdom
            kingdom_confidences = {}
            for kingdom, similarities in kingdom_votes.items():
                kingdom_confidences[kingdom] = np.mean(similarities)
            
            # Best prediction
            best_kingdom = max(kingdom_confidences, key=kingdom_confidences.get)
            best_confidence = kingdom_confidences[best_kingdom]
            
            # Add qualification based on confidence level
            if best_confidence < 0.3:
                qualified_prediction = f"Possibly {best_kingdom} (Low confidence)"
            elif best_confidence < 0.5:
                qualified_prediction = f"Likely {best_kingdom} (Moderate confidence)"
            else:
                qualified_prediction = f"Probably {best_kingdom} (High confidence)"
            
            predictions.append({
                'predicted_kingdom': qualified_prediction,
                'base_kingdom': best_kingdom,
                'confidence': best_confidence,
                'all_predictions': kingdom_confidences,
                'top_similarity': best_matches[0]['similarity']
            })
        else:
            # Even for very low similarity, make a best guess
            if best_matches:
                best_kingdom = best_matches[0]['kingdom']
                best_similarity = best_matches[0]['similarity']
                
                predictions.append({
                    'predicted_kingdom': f"Distantly related to {best_kingdom} (Very low confidence)",
                    'base_kingdom': best_kingdom,
                    'confidence': best_similarity,
                    'all_predictions': {best_kingdom: best_similarity},
                    'top_similarity': best_similarity,
                    'note': 'This may represent a novel organism or very distant relative'
                })
            else:
                predictions.append({
                    'predicted_kingdom': 'Completely Novel (No reference data)',
                    'base_kingdom': 'Unknown',
                    'confidence': 0.0,
                    'all_predictions': {'Unknown': 1.0},
                    'top_similarity': 0.0
                })
    
    return predictions

def cluster_unknown_sequences(unknown_kmer_data, min_cluster_size=2):
    """Clusters unknown sequences using unsupervised learning."""
    if len(unknown_kmer_data) < min_cluster_size:
        return [-1] * len(unknown_kmer_data)
    
    try:
        # Convert k-mer data to vectors
        vectorizer = DictVectorizer(sparse=False)
        X = vectorizer.fit_transform(unknown_kmer_data)
        
        # Apply PCA for dimensionality reduction if needed
        if X.shape[1] > 100:
            pca = PCA(n_components=min(100, X.shape[0] - 1))
            X = pca.fit_transform(X)
        
        # Calculate distance matrix
        distance_matrix = pairwise_distances(X, metric='cosine')
        
        # Cluster using HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=max(2, min(min_cluster_size, len(unknown_kmer_data) // 3)),
            metric='precomputed',
            cluster_selection_epsilon=0.1
        )
        
        cluster_labels = clusterer.fit_predict(distance_matrix)
        return cluster_labels
    except Exception as e:
        print(f"Clustering error: {e}")
        return [-1] * len(unknown_kmer_data)

def analyze_unknown_sequences_composition(unknown_sequences_data, known_sequences_data):
    """Main function to analyze unknown sequences using unsupervised learning."""
    unknown_kmer_data = [data['kmer_data'] for data in unknown_sequences_data.values()]
    unknown_seq_ids = list(unknown_sequences_data.keys())
    
    # Step 1: Cluster unknown sequences
    cluster_labels = cluster_unknown_sequences(unknown_kmer_data)
    
    # Step 2: For each cluster, predict taxonomy
    cluster_predictions = defaultdict(list)
    sequence_predictions = {}
    
    for i, (seq_id, cluster_label) in enumerate(zip(unknown_seq_ids, cluster_labels)):
        kmer_data = unknown_kmer_data[i]
        
        # Predict taxonomy for this sequence
        prediction = predict_unknown_taxonomy([kmer_data], known_sequences_data)[0]
        sequence_predictions[seq_id] = prediction
        
        cluster_name = f"Unknown_Cluster_{cluster_label}" if cluster_label != -1 else "Singleton"
        cluster_predictions[cluster_name].append(prediction)
    
    # Step 3: Aggregate predictions per cluster
    cluster_summary = {}
    total_unknown_sequences = len(unknown_seq_ids)
    kingdom_totals = defaultdict(int)
    
    for cluster_name, predictions in cluster_predictions.items():
        # Get consensus prediction for this cluster
        kingdom_votes = defaultdict(list)
        
        # Extract base kingdoms from qualified predictions
        base_kingdom_votes = defaultdict(list)
        
        for pred in predictions:
            base_kingdom = pred.get('base_kingdom', pred['predicted_kingdom'])
            if base_kingdom and base_kingdom != 'Unknown':
                base_kingdom_votes[base_kingdom].append(pred['confidence'])
        
        for kingdom, confidences in base_kingdom_votes.items():
            kingdom_votes[kingdom] = np.mean(confidences)
        
        # Best prediction for cluster
        if kingdom_votes:
            best_kingdom = max(kingdom_votes, key=kingdom_votes.get)
            best_confidence = kingdom_votes[best_kingdom]
            
            # Create qualified cluster prediction
            if best_confidence < 0.3:
                cluster_prediction = f"Possibly {best_kingdom}"
            elif best_confidence < 0.5:
                cluster_prediction = f"Likely {best_kingdom}"
            else:
                cluster_prediction = f"Probably {best_kingdom}"
        else:
            best_kingdom = "Completely Novel"
            best_confidence = 0.0
            cluster_prediction = "Completely Novel"
        
        cluster_summary[cluster_name] = {
            'predicted_kingdom': cluster_prediction,
            'base_kingdom': best_kingdom,
            'confidence': best_confidence,
            'num_sequences': len(predictions),
            'all_predictions': kingdom_votes,
            'individual_predictions': [p['predicted_kingdom'] for p in predictions]
        }
        
        # Add to total counts (use base kingdom for pie chart)
        kingdom_totals[best_kingdom] += len(predictions)
    
    # Step 4: Create pie chart data
    unknown_composition = {}
    for kingdom, count in kingdom_totals.items():
        unknown_composition[kingdom] = count  # Return actual counts, not percentages
    
    return {
        'cluster_summary': dict(cluster_summary),
        'sequence_predictions': sequence_predictions,
        'unknown_composition': unknown_composition,
        'total_unknown_sequences': total_unknown_sequences,
        'num_unknown_clusters': len([k for k in cluster_summary.keys() if 'Cluster' in k])
    }

# 4. Helper functions for file processing
def generate_kmers(sequence, k):
    """Generates all possible k-mers for a given DNA sequence."""
    return re.findall(r'(?=(.{{{k}}}))'.format(k=k), sequence)

def process_fasta_file(file_contents, k):
    """Parses a FASTA file string, generates k-mer counts, and stores original sequences."""
    kmer_data, sequence_ids, original_sequences = [], [], {}
    
    try:
        fasta_file = StringIO(file_contents)
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence_ids.append(record.id)
            sequence_str = str(record.seq).upper()
            original_sequences[record.id] = sequence_str
            kmers = generate_kmers(sequence_str, k)
            if kmers:
                kmer_counts = Counter(kmers)
                kmer_data.append(dict(kmer_counts))
    except Exception as e:
        print(f"Error processing FASTA file: {e}")
        
    return sequence_ids, kmer_data, original_sequences

def run_blast_search(sequence):
    """Runs an online BLAST search and returns the top hit's identity and name."""
    try:
        result_handle = NCBIWWW.qblast(program="blastn", database="nt", sequence=sequence)
        blast_record = NCBIXML.read(result_handle)
        if blast_record.alignments:
            top_alignment = blast_record.alignments[0]
            top_hsp = top_alignment.hsps[0]
            if top_hsp.expect < 0.001:
                percent_identity = (top_hsp.identities / top_hsp.align_length) * 100
                readable_name = top_alignment.title.split(' ', 1)[1] if ' ' in top_alignment.title else top_alignment.title
                return readable_name, f"{percent_identity:.2f}% Match"
        return "No significant match found.", "N/A"
    except Exception as e:
        print(f"BLAST search failed: {e}")
        return "BLAST search failed.", "Error"

# 5. API endpoints
@app.get("/")
async def root():
    return {"message": "eDNA Analysis API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze/")
async def analyze_edna(file: UploadFile = File(...), k: int = Form(...)):
    try:
        # Validate input
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith(('.fasta', '.fa', '.fna')):
            raise HTTPException(status_code=400, detail="File must be a FASTA file (.fasta, .fa, or .fna)")
        
        if k < 3 or k > 20:
            raise HTTPException(status_code=400, detail="K-mer size must be between 3 and 20")
        
        contents = await file.read()
        contents_str = contents.decode("utf-8")
        sequence_ids, kmer_data, original_sequences = process_fasta_file(contents_str, k)
        
        if not kmer_data:
            raise HTTPException(status_code=400, detail="No valid FASTA sequences found.")

        # Initial clustering of all sequences
        try:
            X = DictVectorizer(sparse=False).fit_transform(kmer_data)
            distance_matrix = pairwise_distances(X, metric='cosine')
            clusterer = HDBSCAN(min_cluster_size=2, metric='precomputed')
            cluster_labels = clusterer.fit_predict(distance_matrix)
        except Exception as e:
            print(f"Clustering error: {e}")
            cluster_labels = [-1] * len(kmer_data)  # All noise if clustering fails

        # Organize results
        clustered_results = {}
        for i, seq_id in enumerate(sequence_ids):
            label = int(cluster_labels[i])
            cluster_name = "Noise / Outliers" if label == -1 else f"Cluster {label}"
            if cluster_name not in clustered_results:
                clustered_results[cluster_name] = {"annotation": "N/A", "confidence": "N/A", "sequences": []}
            clustered_results[cluster_name]["sequences"].append(seq_id)

        # BLAST search for cluster representatives
        known_sequences_data = {}
        unknown_sequences_data = {}
        
        for cluster_name, cluster_data in clustered_results.items():
            if cluster_name != "Noise / Outliers" and cluster_data["sequences"]:
                rep_seq_id = cluster_data["sequences"][0]
                rep_sequence = original_sequences[rep_seq_id]
                annotation, confidence = run_blast_search(rep_sequence)
                clustered_results[cluster_name]["annotation"] = annotation
                clustered_results[cluster_name]["confidence"] = confidence
                
                # Categorize sequences as known or unknown
                seq_index = sequence_ids.index(rep_seq_id)
                
                if "No significant match" not in annotation and "BLAST search failed" not in annotation:
                    # This is a KNOWN sequence
                    taxonomy = extract_taxonomy_from_blast_title(annotation)
                    for seq_id in cluster_data["sequences"]:
                        seq_idx = sequence_ids.index(seq_id)
                        known_sequences_data[seq_id] = {
                            'kmer_data': kmer_data[seq_idx],
                            'taxonomy': taxonomy,
                            'annotation': annotation
                        }
                else:
                    # This is an UNKNOWN sequence
                    for seq_id in cluster_data["sequences"]:
                        seq_idx = sequence_ids.index(seq_id)
                        unknown_sequences_data[seq_id] = {
                            'kmer_data': kmer_data[seq_idx]
                        }
        
        # Handle noise/outliers
        if "Noise / Outliers" in clustered_results:
            for seq_id in clustered_results["Noise / Outliers"]["sequences"]:
                seq_idx = sequence_ids.index(seq_id)
                unknown_sequences_data[seq_id] = {
                    'kmer_data': kmer_data[seq_idx]
                }

        # Analyze unknown sequences using unsupervised learning
        unknown_analysis = {}
        if unknown_sequences_data:
            if known_sequences_data:
                unknown_analysis = analyze_unknown_sequences_composition(
                    unknown_sequences_data, known_sequences_data
                )
            else:
                # No known sequences to train on
                unknown_analysis = {
                    'unknown_composition': {'Completely Novel': len(unknown_sequences_data)},
                    'total_unknown_sequences': len(unknown_sequences_data),
                    'cluster_summary': {},
                    'message': 'No known sequences found for comparison - all sequences appear to be novel'
                }
        
        return {
            "filename": file.filename,
            "k_value": k,
            "num_clusters_found": len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            "cluster_results": clustered_results,
            "total_sequences_processed": len(sequence_ids),
            "known_sequences_count": len(known_sequences_data),
            "unknown_sequences_count": len(unknown_sequences_data),
            "unknown_organism_predictions": unknown_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict_unknowns/")
async def predict_unknowns_only(file: UploadFile = File(...), k: int = Form(...)):
    """Endpoint focused specifically on predicting unknown organisms."""
    try:
        result = await analyze_edna(file, k)
        
        unknown_predictions = result.get("unknown_organism_predictions", {})
        
        return {
            "filename": result["filename"],
            "total_sequences": result["total_sequences_processed"],
            "known_sequences": result["known_sequences_count"],
            "unknown_sequences": result["unknown_sequences_count"],
            "unknown_predictions_pie_data": unknown_predictions.get("unknown_composition", {}),
            "prediction_details": unknown_predictions,
            "message": f"Predicted taxonomy for {result['unknown_sequences_count']} unknown sequences using unsupervised learning"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. Run server
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)