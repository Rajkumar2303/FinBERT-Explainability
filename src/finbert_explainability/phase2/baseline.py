import itertools
import warnings

import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..pipeline import write_json
from .common import choose_candidate, export_predictions


def fit_baseline(train, params, cfg, seed):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=tuple(params["ngram_range"]),
                                 max_features=cfg.baseline_max_features, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=params["C"], solver="lbfgs",
                                         max_iter=cfg.baseline_max_iter, random_state=seed))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit([r["sentence"] for r in train], [r["label"] for r in train])
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("All three classes must occur in training")
    return model


def tune_baseline(train, validation, cfg, out, logger):
    records = []
    for index, (c, ngram) in enumerate(itertools.product(cfg.baseline_c, cfg.baseline_ngram_ranges)):
        params = {"C": c, "ngram_range": ngram}
        path = out / f"candidate-{index}"
        path.mkdir()
        model = fit_baseline(train, params, cfg, cfg.search_seed)
        report = export_predictions(path, "validation", validation,
                                    model.predict_proba([r["sentence"] for r in validation]))
        records.append({"candidate": index, "params": params, "validation_macro_f1": report["macro_f1"]})
        write_json(out / "search_records.json", records)
        logger.info("Candidate %s validation macro-F1=%s", index, report["macro_f1"])
    winner = choose_candidate(records)
    # Independent fits with identical training data; lbfgs is normally deterministic.
    for seed in cfg.seeds:
        path = out / f"seed-{seed}"
        path.mkdir()
        model = fit_baseline(train, winner["params"], cfg, seed)
        joblib.dump(model, path / "model.joblib")
        export_predictions(path, "validation", validation, model.predict_proba([r["sentence"] for r in validation]))
    return winner
