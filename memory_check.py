# encoding:utf-8
import keras
import gc
import psutil

class MemoryCheck(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        gc.collect()
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"Memory used: {mem_info.rss / (1024 * 1024)} MB")
