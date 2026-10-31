"""Conservative lexical duplicate detection and transitive grouping."""
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .data import normalize


def detect_duplicates(rows, threshold=0.9):
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    texts = [normalize(r["sentence"]) for r in rows]
    pairs = {}
    buckets = defaultdict(list)
    for i, text in enumerate(texts):
        if not text:
            raise ValueError("Empty normalized sentence")
        for j in buckets[text]:
            pairs[j, i] = {"left": j, "right": i, "kind": "exact", "similarity": 1.0}
        buckets[text].append(i)
    if len(rows) > 1:
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False).fit_transform(texts)
        neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1).fit(matrix)
        # Bound dense working memory by querying in batches.
        for start in range(0, len(rows), 256):
            distances, indices = neighbors.radius_neighbors(matrix[start:start+256], radius=1-threshold+1e-12)
            for offset, (ds, js) in enumerate(zip(distances, indices)):
                i = start + offset
                for distance, j in zip(ds, js):
                    j = int(j)
                    if i < j and (i, j) not in pairs and 1-distance+1e-12 >= threshold:
                        pairs[i, j] = {"left": i, "right": j, "kind": "near", "similarity": float(1-distance)}
    return [pairs[k] for k in sorted(pairs)]


def connected_groups(size, pairs):
    parent = list(range(size))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for p in pairs:
        a, b = find(p["left"]), find(p["right"])
        parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(size)]


def cross_split_pairs(pairs, assignments):
    return [p for p in pairs if assignments[p["left"]] != assignments[p["right"]]]
