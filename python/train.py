"""Step 2: train the Table III 1D-CNN for each outer fold.

Outputs (models/fold<k>/): model.keras, norm.json, history.csv, train_info.json
"""
import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

import config as C
import data as D


def build_model(n_classes):
    """Table III. 'same' padding keeps the 128-step length, matching the reported output shapes."""
    return tf.keras.Sequential([
        tf.keras.layers.Input((C.WINDOW, len(C.AXES))),
        tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])


def class_weights(y, n_classes, mode):
    """'balanced': inverse frequency; 'sqrt': inverse square root of frequency; 'none': no weights."""
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(float)
    present = counts > 0
    raw = np.zeros(n_classes)
    raw[present] = 1.0 / (counts[present] if mode == "balanced" else np.sqrt(counts[present]))
    raw *= counts.sum() / (raw * counts).sum()   # average weight per training window = 1
    return {i: float(w) for i, w in enumerate(raw)}


def fit(fold, weighting, stride, verbose=2):
    """Train one model; returns (model, mean, std, history, seconds, n_train, n_val)."""
    tf.keras.utils.set_random_seed(C.SEED)
    mean, std = D.fit_norm(fold["train"])
    x_tr, y_tr = D.labelled_windows(fold["train"], stride)
    x_va, y_va = D.labelled_windows(fold["val"], C.EVAL_STRIDE)
    x_tr, x_va = D.normalize(x_tr, mean, std), D.normalize(x_va, mean, std)

    n_classes = len(C.CLASSES)
    model = build_model(n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=C.LR_PLATEAU_PATIENCE),
    ]
    start = time.time()
    hist = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=C.MAX_EPOCHS, batch_size=C.BATCH_SIZE,
                     class_weight=class_weights(y_tr, n_classes, weighting), callbacks=callbacks, verbose=verbose)
    return model, mean, std, hist, time.time() - start, len(y_tr), (x_va, y_va)


def val_macro_f1(model, val):
    x_va, y_va = val
    pred = np.argmax(model.predict(x_va, verbose=0), axis=1)
    return float(f1_score(y_va, pred, average="macro", labels=np.unique(y_va), zero_division=0))


def train_fold(fold):
    out = D.fold_dir(fold["fold"])
    model, mean, std, hist, seconds, n_train, val = fit(fold, C.CLASS_WEIGHTING, C.TRAIN_STRIDE)

    model.save(out / "model.keras")
    pd.DataFrame(hist.history).rename_axis("epoch").to_csv(out / "history.csv")
    D.save_json(out / "norm.json", {"mean": mean.tolist(), "std": std.tolist()})
    D.save_json(out / "train_info.json", {
        "fold": fold["fold"],
        "train_participants": fold["train"],
        "val_participants": fold["val"],
        "classes": list(C.CLASSES),
        "class_weighting": C.CLASS_WEIGHTING,
        "train_stride": C.TRAIN_STRIDE,
        "train_windows": int(n_train),
        "val_windows": int(len(val[1])),
        "val_macro_f1_float": val_macro_f1(model, val),
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
        "training_seconds": round(seconds, 1),
        "parameters": int(model.count_params()),
        "float_model_bytes": (out / "model.keras").stat().st_size,
    })
    print(f"fold {fold['fold']}: {model.count_params()} parameters, {n_train} training windows, {seconds:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, help="train one fold (default: all)")
    args = ap.parse_args()
    for fold in D.load_folds():
        if args.fold is None or fold["fold"] == args.fold:
            train_fold(fold)


if __name__ == "__main__":
    main()
