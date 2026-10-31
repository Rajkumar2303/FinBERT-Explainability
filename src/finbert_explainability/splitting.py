from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

SPLITS = ("train", "validation", "test")


def stratified_split(rows, fractions, seed):
    labels = np.array([r["label"] for r in rows])
    train, rest = train_test_split(np.arange(len(rows)), test_size=fractions[1]+fractions[2],
                                   stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=fractions[2]/(fractions[1]+fractions[2]),
                                 stratify=labels[rest], random_state=seed)
    assignments = [None] * len(rows)
    for name, indices in zip(SPLITS, (train, val, test)):
        for i in indices:
            assignments[i] = name
    return assignments


def grouped_stratified_split(rows, groups, fractions, seed):
    """Greedily minimize per-class target error, keeping components indivisible.

    Largest components first, seeded tie order. Ratios are approximate when
    group sizes prevent exact quotas. No rows are dropped or copied.
    """
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    ordered = list(members.values())
    np.random.default_rng(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    labels = np.array([r["label"] for r in rows])
    target = np.outer(fractions, np.bincount(labels, minlength=3))
    counts = np.zeros((3, 3))
    result = [None] * len(rows)
    for indices in ordered:
        addition = np.bincount(labels[indices], minlength=3)
        costs = (((counts+addition-target)**2 - (counts-target)**2) / np.maximum(target, 1)).sum(axis=1)
        chosen = int(np.argmin(costs))
        counts[chosen] += addition
        for i in indices:
            result[i] = SPLITS[chosen]
    if any(not np.any(np.array(result) == name) for name in SPLITS):
        raise ValueError("Duplicate groups cannot form three nonempty splits")
    return result
