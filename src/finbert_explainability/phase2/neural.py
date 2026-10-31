import gc

from datasets import Dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig, Trainer, TrainingArguments

from ..data import LABELS
from ..pipeline import seed_everything, write_json
from .common import choose_candidate, export_predictions
from .metrics import canonical_probabilities, trainer_metrics

CANONICAL = dict(enumerate(LABELS))


def load_bert_config(model_spec, kwargs, *, classification=False):
    # finbert-pretrain's original config omits model_type. Explicit BERT loading
    # avoids Transformers 5 AutoConfig failure without altering the source model.
    config = BertConfig.from_pretrained(model_spec["id"], **kwargs)
    if classification:
        config.num_labels = 3
        config.id2label = CANONICAL.copy()
        config.label2id = {v: k for k, v in CANONICAL.items()}
    return config


def tokenize(rows, tokenizer, max_length):
    tokens = tokenizer([r["sentence"] for r in rows], truncation=True,
                       padding="max_length", max_length=max_length)
    return Dataset.from_dict({**tokens, "labels": [r["label"] for r in rows]})


def arguments(path, cfg, params, seed, *, cpu=False, smoke=False):
    return TrainingArguments(
        output_dir=str(path), learning_rate=params["learning_rate"], num_train_epochs=params["epochs"],
        per_device_train_batch_size=2 if smoke else cfg.per_device_batch_size,
        per_device_eval_batch_size=2 if smoke else cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        fp16=bool(cfg.fp16 and not cpu), bf16=False, use_cpu=cpu,
        weight_decay=cfg.weight_decay, optim="adamw_torch", lr_scheduler_type="linear",
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, save_only_model=True, full_determinism=True,
        seed=seed, data_seed=seed, dataloader_num_workers=0, dataloader_pin_memory=not cpu,
        logging_steps=1 if smoke else 25, report_to="none", disable_tqdm=True,
        tf32=False, eval_accumulation_steps=1)


def train_once(train, validation, cfg, model_spec, params, seed, path, *, cpu=False, smoke=False):
    seed_everything(seed)  # Must precede random classifier initialization.
    kwargs = ({"local_files_only": True} if smoke else
              {"revision": model_spec["revision"], "cache_dir": cfg.cache_dir})
    config = load_bert_config(model_spec, kwargs, classification=True)
    tokenizer = AutoTokenizer.from_pretrained(model_spec["id"], config=config, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_spec["id"], config=config, attn_implementation="eager", **kwargs)
    # FinBERT-pretrain is an MLM checkpoint: its new classifier is deliberately seeded.
    train_data = tokenize(train, tokenizer, cfg.max_length)
    val_data = tokenize(validation, tokenizer, cfg.max_length)
    trainer = Trainer(model=model, args=arguments(path / "checkpoints", cfg, params, seed, cpu=cpu, smoke=smoke),
        train_dataset=train_data, eval_dataset=val_data, processing_class=tokenizer,
        compute_metrics=trainer_metrics)
    trainer.train()
    prediction = trainer.predict(val_data)
    report = export_predictions(path, "validation", validation,
        canonical_probabilities(prediction.predictions, trainer.model.config.id2label))
    trainer.save_model(str(path / "model"))
    tokenizer.save_pretrained(path / "model")
    write_json(path / "training_record.json", {"seed": seed, "params": params,
        "model": model_spec, "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "log_history": trainer.state.log_history, "training_arguments": trainer.args.to_dict(),
        "selection_split": "validation", "smoke_only": smoke})
    del trainer, model, tokenizer
    gc.collect()
    if not cpu:
        torch.cuda.empty_cache()
    return report


def tune_neural(train, validation, cfg, model_spec, out, logger, *, cpu=False, smoke=False):
    records = []
    candidates = [{"learning_rate": 2e-5, "epochs": 1}] if smoke else cfg.candidates()
    for index, params in enumerate(candidates):
        path = out / f"candidate-{index}"
        path.mkdir()
        report = train_once(train, validation, cfg, model_spec, params, cfg.search_seed, path, cpu=cpu, smoke=smoke)
        records.append({"candidate": index, "params": params, "validation_macro_f1": report["macro_f1"]})
        write_json(out / "search_records.json", records)
        logger.info("Candidate %s validation macro-F1=%s", index, report["macro_f1"])
    winner = choose_candidate(records)
    for seed in cfg.seeds:
        path = out / f"seed-{seed}"
        path.mkdir()
        train_once(train, validation, cfg, model_spec, winner["params"], seed, path, cpu=cpu, smoke=smoke)
    return winner


def predict_model(rows, model_spec, cfg, *, cpu=False, local=False):
    kwargs = {"local_files_only": True} if local else {"revision": model_spec["revision"], "cache_dir": cfg.cache_dir}
    config = load_bert_config(model_spec, kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_spec["id"], config=config, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_spec["id"], config=config, attn_implementation="eager", **kwargs)
    # Validate id2label before inference, particularly for the external reference.
    canonical_probabilities([[0, 0, 0]], model.config.id2label)
    mapping = dict(model.config.id2label)
    device = torch.device("cpu" if cpu else "cuda")
    model.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), cfg.per_device_batch_size):
            batch = tokenizer([r["sentence"] for r in rows[start:start+cfg.per_device_batch_size]],
                padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt").to(device)
            # FP32 evaluation for all saved models and the external reference.
            logits = model(**batch).logits.float().cpu().numpy()
            predictions.extend(canonical_probabilities(logits, mapping).tolist())
    del model, tokenizer
    gc.collect()
    if not cpu:
        torch.cuda.empty_cache()
    return predictions, mapping
