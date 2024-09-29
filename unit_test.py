import unittest
from collections import defaultdict
from scipy.sparse import csr_matrix
import numpy as np
import random
from gensim.utils import tokenize

from utils import *

# Assuming the following functions are imported from your code
# from your_module import create_n_grams, weighted_overlap, text_to_csr_matrix, tokenize_corpus, uniform_subsample

def expected_n_grams(words, max_n_gram_size):
    return len(words) * max_n_gram_size - sum(range(1, max_n_gram_size))
class TestFunctions(unittest.TestCase):

    def test_create_n_grams(self):
        # a. Sanity check
        words = ["I", "love", "Python"]
        result = create_n_grams(words, 2)
        expected = ["I", "love", "Python", "I love", "love Python"]
        self.assertEqual(set(result), set(expected))

        # b. Empty list
        words = []
        result = create_n_grams(words, 3)
        self.assertEqual(result, [])  # Expect empty result for empty input

        # c. Input shorter than max n-gram size
        words = ["short"]
        result = create_n_grams(words, 2)
        self.assertEqual(result, ["short"])  # Single word, no bigrams can be created

        # d. Test that total number of n-grams is as expected
        words = ["This", "is", "a", "test"]
        max_n_gram_size = 3
        result = create_n_grams(words, max_n_gram_size)
        expected_length = expected_n_grams(words, max_n_gram_size)
        self.assertEqual(len(result), expected_length)

    def test_weighted_overlap(self):
        # a. Calculate overlap score correctly
        gts = [1, 2, 3]
        preds = [1, 2, 3]
        weights = [1, 1, 1]
        score = weighted_overlap(gts, preds, weights)
        self.assertEqual(score, 3)  # Complete overlap, score should be sum of weights

        # b. No correct items
        gts = [1, 2, 3]
        preds = [4, 4, 4]
        score = weighted_overlap(gts, preds, weights)
        self.assertEqual(score, 0)  # No overlap

        # c. One correct item
        gts = [1, 1, 3]
        preds = [1, 2, 3]
        score = weighted_overlap(gts, preds, weights)
        self.assertEqual(score, 1)  # Two overlaps (positions 1 and 3)

        # d. Mismatched lengths
        gts = [1, 2]
        preds = [1, 2, 3]
        with self.assertRaises(AssertionError):
            weighted_overlap(gts, preds, weights)

    def test_text_to_csr_matrix(self):
        # a. Basic test for conversion to CSR matrix
        corpus = [["this", "is", "a", "test"], ["another", "document"]]
        seed = 42
        max_n_gram_size = 2
        hash_vector_size = 100
        result = text_to_csr_matrix(corpus, seed, max_n_gram_size, hash_vector_size,normalize=False)
        self.assertIsInstance(result, csr_matrix)  # Ensure the result is a CSR matrix
        self.assertEqual(result.shape, (2, hash_vector_size))  # Shape matches corpus size

        # b. Check sparsity (not all elements are filled)
        self.assertGreater(result.nnz, 0)  # Ensure there are some non-zero values
        self.assertEqual(result.sum(), sum([expected_n_grams(words, max_n_gram_size) for words in corpus]))  # It should be sparse
        
        # c. Check that normalization logic gives almost 2
        normalized_result = text_to_csr_matrix(corpus, seed, max_n_gram_size, hash_vector_size)
        self.assertAlmostEqual(normalized_result.sum(),2)
        
        # d. Edge case: Empty corpus
        with self.assertRaises(AssertionError):
            result = text_to_csr_matrix([], seed, max_n_gram_size, hash_vector_size)

    def test_tokenize_corpus(self):
        # a. Tokenization of normal strings
        corpus = ["This is a test.", "Another document!"]
        result = tokenize_corpus(corpus)
        expected = [["this", "is", "a", "test"], ["another", "document"]]
        self.assertEqual(result, expected)

        # b. Empty string and special characters
        corpus = ["", "@#$$%^&*()"]
        result = tokenize_corpus(corpus)
        expected = [[], []]  # Should return empty tokens for both
        self.assertEqual(result, expected)

    def test_uniform_subsample(self):
        # a. Ensure uniform subsampling works correctly
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = uniform_subsample(values, num_bins=3, num_items_per_bin=1)
        self.assertLessEqual(len(result), 3)  # Should return at most 3 samples
        
        # TODO: Add back later
        '''
        # b. Edge cases
        values = [1] * 10  # All identical values
        result = uniform_subsample(values, num_bins=5, num_items_per_bin=2)
        self.assertLessEqual(len(result), 5)  # Subsample should return at most 5 items

        # c. More bins than items
        values = [0.1, 0.2]
        result = uniform_subsample(values, num_bins=5, num_items_per_bin=2)
        self.assertLessEqual(len(result), 2)  # Shouldn't return more items than exist
        '''
        
if __name__ == '__main__':
    unittest.main()
