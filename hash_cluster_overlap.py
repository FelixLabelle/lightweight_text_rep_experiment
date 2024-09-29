from collections import Counter
import json
from multiprocessing import Pool
import os
import random

from gensim import downloader as api
from gensim.models import FastText
from pickle import dump,load
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score,fowlkes_mallows_score
from tqdm.auto import tqdm

from utils import create_n_grams, calculate_n_gram_labels, set_seeds, weighted_overlap,tokenize_corpus,weighted_bidirectional_overlap

MAX_N_GRAMS = 1_000_000 # TODO: Try larger sizes 1_000_000 
MAX_N_GRAM_SIZE = 2
NUM_UNIQUE_SEEDS = 5_000 #500_000
EMBEDDING_SIZE = 50 # TODO: Try larger sizes
RAND_SEED = 0
MAX_DOCUMENTS = 300_000 # TODO: Try larger sizes
MIN_SIMILARITY = 0.98 #0.95 # THRESHOLD TO SELECT MIN SIMILARITY
CLUSTER_SIMILARITY_METRIC = "WEIGHTED_BIDIRECTIONAL_OVERLAP" #"WEIGHTED_OVERLAP" #"FOWLKES_MALLOWS" # "RAND_SCORE" # "FOWLKES_MALLOWS"
TAG = "v10"
MODEL_PATH = f"{MAX_DOCUMENTS}-docs_{EMBEDDING_SIZE}-dim.fstxt"
CLUSTER_MODEL_PATH = f"clustering__{MIN_SIMILARITY}_{MAX_N_GRAMS}_{MAX_DOCUMENTS}-docs_{EMBEDDING_SIZE}-dim.pkl"

def calculate_sim(args):
    n_grams, seed, kmeans_classes,hash_vector_size, weights = args
    hash_classes,signs = zip(*calculate_n_gram_labels(n_grams,seed,hash_vector_size))
    # Lets make sure this doesn't cause issues with built in functions
    # this makes sure similar items don't hurt negate other, but also doesn't count
    # collisions
    #hash_classes = [hash_class * sign for hash_class,sign in zip(hash_classes,signs)]
    
    if CLUSTER_SIMILARITY_METRIC == "FOWLKES_MALLOWS":
        similarity = fowlkes_mallows_score(kmeans_classes, hash_classes)
    elif CLUSTER_SIMILARITY_METRIC == "RAND_SCORE":
        similarity = adjusted_rand_score(kmeans_classes, hash_classes)
    elif CLUSTER_SIMILARITY_METRIC == "WEIGHTED_OVERLAP":
        similarity = weighted_overlap(kmeans_classes, hash_classes,weights)
    elif CLUSTER_SIMILARITY_METRIC == "WEIGHTED_BIDIRECTIONAL_OVERLAP":
        similarity = weighted_bidirectional_overlap(kmeans_classes, hash_classes,weights,beta=2)
    else:
        raise NotImplementedError()
    
    output = {"seed" : seed,  "hash_func" : "mmh3", "hash_vector_size" : hash_vector_size}
    if isinstance(similarity,dict):
        output.update(similarity)
    else:
        output["similarity"] = similarity
        
    return output
        
if __name__ == "__main__":
    set_seeds(RAND_SEED)
    
    # 1. load a dataset
    corpus = api.load('wiki-english-20171001')
    corpus = ["\n".join(item['section_texts']) for item, _ in tqdm(zip(corpus, range(MAX_DOCUMENTS)))]
    corpus = tokenize_corpus(corpus,verbose=True)

    # 2. Create or load embedding space (use word counts etc..)
    # NOTE: Might be worth including pretrained
    if not os.path.exists(CLUSTER_MODEL_PATH) and os.path.exists(MODEL_PATH):
        print("Loading embedding model")
        model = FastText.load(MODEL_PATH)
    elif not os.path.exists(MODEL_PATH):
        print("generating fasttext")
        model = FastText(vector_size=EMBEDDING_SIZE, window=5, min_count=1)  # instantiate
        model.build_vocab(corpus_iterable=corpus)
        model.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=10)
        model.save(MODEL_PATH)
    else:
        pass
    # 3. Generate and embed n-grams
    print("Generating n-grams")
    n_gram_counts = Counter([n_gram for tokens in tqdm(corpus) for n_gram in create_n_grams(tokens,MAX_N_GRAM_SIZE)])
    number_of_tokens = sum(n_gram_counts.values())
    print("Sorting n_grams")
    n_gram_counts = {key:value for (key,value), _  in zip(sorted(n_gram_counts.items(),key=lambda x:x[1], reverse=True),range(MAX_N_GRAMS))}
    list_n_grams = list(n_gram_counts.keys())
    del corpus    
    number_of_tokens_kept = sum(n_gram_counts.values())
    frequency_weights = [num_occs_token/number_of_tokens_kept for num_occs_token in n_gram_counts.values()]
    
    # Print report on coverage of n-grams and other info calculated
    print(f"{100*number_of_tokens_kept/number_of_tokens}% of tokens kept")
    print(f"Frequency weight total is {sum(frequency_weights)}")
    
    print(f"There are {len(list_n_grams)} n grams")

    # 4. calculate k-means of n-clusters (where n matches the number of hash features)
    
    if os.path.exists(CLUSTER_MODEL_PATH):
        print("Loading clustering model")
        with open(CLUSTER_MODEL_PATH, "rb") as fh:
            clustering_model = load(fh)
    else:
        print("Embedding")
        embeddings = model.wv[list_n_grams]
        print("Clustering")
        del model
        clustering_model = DBSCAN(1-MIN_SIMILARITY,min_samples=1,metric="cosine",n_jobs=-1).fit(embeddings)
        with open(CLUSTER_MODEL_PATH, "wb") as fh:
            dump(clustering_model, fh, protocol=5)
          
    print("Clustering data")
    clusters = clustering_model.labels_.tolist()
    del clustering_model
    print("Separating -1 label")
    num_clusters = max(clusters)
    print(num_clusters)
    for cluster_idx, cluster in enumerate(clusters):
        if cluster == -1:
            num_clusters += 1
            clusters[cluster_idx] = num_clusters
    
    # Report info on clusters
    print(f"There are {num_clusters} clusters")
    # TODO: Add cluster size summary stats, etc..
    
    seeds = []
    # 5 calculate similarity between k-means grouping across different seeds 
    seed_hash_size_combos = [(seed,hash_vector_size) for seed in tqdm(random.sample(range(1_000_000_000_000_000), NUM_UNIQUE_SEEDS)) for hash_vector_size in range(num_clusters,num_clusters*11,num_clusters)]
    args = [(list_n_grams, seed,clusters,hash_vector_size,frequency_weights) for seed, hash_vector_size in tqdm(seed_hash_size_combos)]
    with Pool() as pool:
        seeds = [seed for seed in tqdm(pool.imap(calculate_sim, args),total=len(args))]
        # NOTE: Consider adjusting for importance of words, more common words should be more heavily penalized for errors) (maybe I can upweight items by representing them in the input??)
    
    # 6. Save
    #import pdb;pdb.set_trace()
    experiment_dict = {"seeds" : seeds,
                       "max_n_gram_size" : MAX_N_GRAM_SIZE,
                       "n_gram_counts" : n_gram_counts,
                       "clusters" : clusters,
                       "embedding_size" : EMBEDDING_SIZE,
                       "rand_seed" : RAND_SEED,
                       "list_n_grams" : list_n_grams,
                       "max_n_grams" : MAX_N_GRAMS,
                       "min_similarity" : MIN_SIMILARITY,
                       "cluster_similarity_metric" : CLUSTER_SIMILARITY_METRIC,
                       "tag" : TAG,
                       "model_path" : MODEL_PATH,
                       "cluster_model_path" : CLUSTER_MODEL_PATH,
                       }
    json.dump(experiment_dict, open(f"wiki_eng_hash_similarity_{TAG}.json","w"))