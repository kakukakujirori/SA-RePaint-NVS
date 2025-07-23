import subprocess
import threading
import time
import torch

class GPUMemoryMonitor:
    def __init__(self, interval: float = 0.1, gpu_id: int = 0, verbose: bool = False):
        self.interval = interval
        self.gpu_id = gpu_id
        self.verbose = verbose
        self.max_mem = 0
        self._running = False
        self._thread = None

    def _query_memory(self):
        try:
            result = subprocess.check_output(
                ["nvidia-smi", f"--id={self.gpu_id}", "--query-compute-apps=used_gpu_memory",
                 "--format=csv,nounits,noheader"]
            )
            lines = [line.strip() for line in result.decode("utf-8").strip().split("\n") if line.strip().isdigit()]
            if self.verbose:
                print(f"[GPUMemoryMonitor] GPU {self.gpu_id} memory usage (MiB): {lines}")
            if not lines:
                return 0
            return max(int(line) for line in lines)
        except Exception as e:
            print(f"[GPUMemoryMonitor] nvidia-smi query error: {e}")
            return 0

    def _monitor(self):
        while self._running:
            mem = self._query_memory()
            if mem > self.max_mem:
                self.max_mem = mem
            time.sleep(self.interval)

    def start(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.max_mem = self._query_memory()
        self._running = True
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        torch.cuda.synchronize()

    def get_max_memory(self):
        return self.max_mem / 1024  # MB → GB

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
