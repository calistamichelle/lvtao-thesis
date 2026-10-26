"""Run the INT8 .tflite model exactly the way the ESP32 firmware does."""
import numpy as np
import tensorflow as tf

import config as C
import data as D


class Int8Model:
    def __init__(self, fold):
        d = D.fold_dir(fold)
        norm = D.load_json(d / "norm.json")
        self.mean = np.array(norm["mean"], np.float32)
        self.std = np.array(norm["std"], np.float32)
        # Reference kernels: the same integer arithmetic as TensorFlow Lite Micro on the ESP32. The default
        # XNNPACK delegate can round differently and flip the argmax when two classes are tied.
        self.interp = tf.lite.Interpreter(
            model_path=str(d / "model_int8.tflite"),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        scale, zero_point = self.inp["quantization"]
        self.scale = np.float32(scale)
        self.zero_point = int(zero_point)

    def quantize_input(self, raw_window):
        """Same float32 arithmetic and round-half-to-even as firmware/src/main.cpp (lrintf)."""
        z = (raw_window.astype(np.float32) - self.mean) / self.std
        q = np.rint(z / self.scale) + self.zero_point
        return np.clip(q, -128, 127).astype(np.int8)

    def predict(self, raw_window):
        self.interp.set_tensor(self.inp["index"], self.quantize_input(raw_window)[None])
        self.interp.invoke()
        return int(np.argmax(self.interp.get_tensor(self.out["index"])[0]))


class CachedPredictor:
    """Predictions for one participant keyed by window end, shared across policies."""

    def __init__(self, model, x):
        self.model, self.x, self.cache = model, x, {}

    def __call__(self, end):
        if end not in self.cache:
            self.cache[end] = self.model.predict(self.x[end - C.WINDOW:end])
        return self.cache[end]
