"""Step 2: train the Table III 1D-CNN for each outer fold.

Outputs (models/fold<k>/): model.keras, norm.json, history.csv, train_info.json
"""
import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf

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


def train_fold(fold):
    out = D.fold_dir(fold["fold"])
    tf.keras.utils.set_random_seed(C.SEED)

    mean, std = D.fit_norm(fold["train"])
    x_tr, y_tr = D.labelled_windows(fold["train"], C.TRAIN_STRIDE)
    x_va, y_va = D.labelled_windows(fold["val"], C.TRAIN_STRIDE)
    x_tr, x_va = D.normalize(x_tr, mean, std), D.normalize(x_va, mean, std)
    print(f"fold {fold['fold']}: {len(y_tr)} training windows, {len(y_va)} validation windows")

    n_classes = len(C.CLASSES)
    class_weight = None
    if C.CLASS_WEIGHTING:
        counts = np.bincount(y_tr, minlength=n_classes).astype(float)
        present = counts > 0
        weights = np.zeros(n_classes)
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
        class_weight = {i: float(w) for i, w in enumerate(weights)}

    model = build_model(n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                                            restore_best_weights=True)
    start = time.time()
    hist = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=C.MAX_EPOCHS,
                     batch_size=C.BATCH_SIZE, class_weight=class_weight, callbacks=[stop], verbose=2)
    seconds = time.time() - start

    model.save(out / "model.keras")
    pd.DataFrame(hist.history).rename_axis("epoch").to_csv(out / "history.csv")
    D.save_json(out / "norm.json", {"mean": mean.tolist(), "std": std.tolist()})
    D.save_json(out / "train_info.json", {
        "fold": fold["fold"],
        "train_participants": fold["train"],
        "val_participants": fold["val"],
        "train_windows": int(len(y_tr)),
        "val_windows": int(len(y_va)),
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
        "training_seconds": round(seconds, 1),
        "parameters": int(model.count_params()),
        "float_model_bytes": (out / "model.keras").stat().st_size,
        "class_weight": class_weight,
    })
    print(f"fold {fold['fold']}: {model.count_params()} parameters, {seconds:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, help="train one fold (default: all)")
    args = ap.parse_args()
    for fold in D.load_folds():
        if args.fold is None or fold["fold"] == args.fold:
            train_fold(fold)


if __name__ == "__main__":
    main()
