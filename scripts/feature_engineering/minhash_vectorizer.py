import hashlib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MinHashVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=50, ngram_range=(3, 5), analyzer='char', prime=2147483647, random_state=42):
        self.n_components = n_components
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.prime = prime
        self.random_state = random_state

    def fit(self, X, y=None):
        # Initialize the random number generator
        rng = np.random.default_rng(self.random_state)
        # Generate random values for a and b in linear permutation hash: h(x) = (a * x + b) % prime
        # a should be positive integers less than prime
        self.a_ = rng.integers(1, self.prime, size=self.n_components)
        # b should be non-negative integers less than prime
        self.b_ = rng.integers(0, self.prime, size=self.n_components)
        return self

    def _get_shingles(self, text):
        if not text or not isinstance(text, str):
            return set()
        shingles = set()
        if self.analyzer == 'char':
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for i in range(len(text) - n + 1):
                    shingles.add(text[i:i+n])
        elif self.analyzer == 'word':
            words = text.split()
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for i in range(len(words) - n + 1):
                    shingles.add(" ".join(words[i:i+n]))
        return shingles

    def transform(self, X):
        signatures = []
        for text in X:
            shingles = self._get_shingles(text)
            if not shingles:
                # If no shingles, return maximum/infinity hash value (normalized to 1.0)
                signatures.append(np.full(self.n_components, 1.0, dtype=np.float64))
                continue
            
            # Map shingles to 32-bit integer hashes using a deterministic md5 snippet
            shingle_hashes = []
            for s in shingles:
                h_val = int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)
                shingle_hashes.append(h_val)
            
            shingle_hashes = np.array(shingle_hashes, dtype=np.int64) # (len_shingles,)
            
            # Compute parallel permutations
            # perms shape: (n_components, len_shingles)
            perms = (self.a_[:, np.newaxis] * shingle_hashes[np.newaxis, :] + self.b_[:, np.newaxis]) % self.prime
            
            # Find the minimum hash value for each permutation and normalize to [0, 1]
            min_hashes = perms.min(axis=1) / self.prime
            signatures.append(min_hashes)
            
        return np.array(signatures, dtype=np.float64)
