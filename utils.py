from collections import defaultdict
import random
from typing import List

from gensim.utils import tokenize
import numpy as np
import mmh3
from scipy.sparse import csr_matrix
from scipy.stats import hmean
from tqdm.auto import tqdm

def set_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    
def create_n_grams(words,max_n_gram_size):
    return [" ".join(words[i:i+n_gram_size]) for n_gram_size in range(1,max_n_gram_size+1) for i in range(1+len(words)-n_gram_size)]

# NOTE: Signed = True effectively halves the space, but will
# center the vectors (on average) and help cancel out noise
def calculate_n_gram_labels(words, seed,hash_vector_size,signed=False):
    def compute_hash_and_sign(word):
        hash_val = mmh3.hash(word, signed=signed,seed=seed)
        return (abs(hash_val) % hash_vector_size,-1 if signed and hash_val < 0 else 1)
    return [compute_hash_and_sign(word) for word in words]

def weighted_overlap(gts : List[int],preds: List[int],weights: List[int]):
    assert(len(gts) == len(preds))
    assert(len(weights) == len(preds))
    score = 0.0
    def get_neighbors(items):
        neighborhoods = defaultdict(list)
        for word_idx, label in enumerate(items):
            neighborhoods[label].append(word_idx)
        return neighborhoods

    gt_groupings = get_neighbors(gts)
    pred_groupings = get_neighbors(preds)

    for word_idx in range(len(gts)):
        if set(gt_groupings[gts[word_idx]]) == set(pred_groupings[preds[word_idx]]):
            score += weights[word_idx]
    return score
    
# weighted f beta
# TODO: Consider using or atleast returning multiple values
# I think the correlation might be more complex than I think 
# Higher beta penalizes collisions more
def weighted_bidirectional_overlap(gts : List[int],preds: List[int],weights: List[int],beta=2):
    assert(len(gts) == len(preds))
    assert(len(weights) == len(preds))
    collision_score = 0.0
    overlap_score = 0.0
    def get_neighbors(items):
        neighborhoods = defaultdict(list)
        for word_idx, label in enumerate(items):
            neighborhoods[label].append(word_idx)
        return neighborhoods

    gt_groupings = get_neighbors(gts)
    pred_groupings = get_neighbors(preds)

    for word_idx in range(len(gts)):
        gt_grouping = set(gt_groupings[gts[word_idx]]) 
        pred_grouping = set(pred_groupings[preds[word_idx]])
        collisions = pred_grouping.difference(gt_grouping)
        dropped_values = gt_grouping.difference(pred_grouping)
        if len(collisions) == 0:
            collision_score += weights[word_idx]
           
        if len(dropped_values) == 0:
            overlap_score += weights[word_idx]
    
    # arithmetic mean of the two numbers
    # TODO: Add other
    score = hmean([collision_score,overlap_score],weights=[beta**2,1])
    return {"similarity" : score, "collision_score" : collision_score, "overlap_score" : overlap_score}
    
def text_to_csr_matrix(corpus,seed,max_n_gram_size,hash_vector_size,normalize=True):
    assert(isinstance(corpus,list))
    assert(len(corpus) > 0)
    assert(all([len(datum) > 0 for datum in corpus]))
    # Initialize the count vector
    idx_data = defaultdict(float)
    
    # Rather than count use a dictionary
    for doc_idx, document in enumerate(corpus):
        n_grams = create_n_grams(document,max_n_gram_size)
        num_n_grams = len(n_grams)
        for n_gram_idx, sign in calculate_n_gram_labels(n_grams,seed,hash_vector_size):
            idx_data[(doc_idx, n_gram_idx)] += sign*(1/num_n_grams if normalize else 1)

    # Convert the mapping to a CSR matrix
    data = list(idx_data.values())
    row_indices,col_indices = zip(*list(idx_data.keys()))
    csr_result = csr_matrix((data,(row_indices, col_indices)), shape=(len(corpus),hash_vector_size))

    return csr_result

def tokenize_corpus(corpus, verbose=False):
    if verbose:
        disp = tqdm
    else:
        disp = lambda x:x
    return [[token for token in tokenize(datum, lowercase=True)] for datum in disp(corpus)]
    
    
def uniform_subsample(values, num_bins, num_items_per_bin):
    if not values:
        return values
    min_sim = min(values)
    max_sim = max(values)
    adjusted_values = [(sim-min_sim)/(max_sim-min_sim) for sim in values]
    
    bins = np.linspace(0, 1, num_bins + 1)  # Bins from 0 to 1
    binned_sims = np.digitize(adjusted_values, bins)  # Bin indices
    selected_idxs = []

    for i in range(num_bins):
        # Find results that fall into the current bin
        bin_indices = [idx for idx, binned_sim in enumerate(binned_sims) if binned_sim == i]
        if len(bin_indices) > 0:
            selected_indices = random.sample(bin_indices, min(len(bin_indices), num_items_per_bin))
            selected_idxs += selected_indices
    return selected_idxs
    