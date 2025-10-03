import multiprocessing as mp
import tempfile
from pathlib import Path
import time
import torch

from miss_alignment.data._reconstruction_worker import reconstruction_worker
from miss_alignment.data._pool_monitor import SimplePoolMonitor
from miss_alignment.data.shift_generation import create_default_generator


if __name__ == "__main__":
    # data directory
    data = Path("/home/marten/data/datasets/shrec_v2/run1/iter0")

    # Configuration
    n_workers = 1
    pool_size = 10
    tilt_series_pickles = [x for x in data.iterdir() if x.suffix == ".pickle"]
    tomogram_shape = (180, 512, 512)
    patch_size = 96
    benchmark_duration = 60  # seconds
    shift_generator = create_default_generator()

    # interested in the impact of this...
    tilt_series_refresh_rate = 100
    device = 'cuda'

    with tempfile.TemporaryDirectory() as tmpdir:
        pool_dir = Path(tmpdir)

        # Multiprocessing signals
        ready_flag = mp.Value('i', 0)
        stop_event = mp.Event()
        monitor = SimplePoolMonitor()

        # Distribute indices across workers
        indices_per_worker = pool_size // n_workers

        workers = []
        for i in range(n_workers):
            start_idx = i * indices_per_worker
            end_idx = start_idx + indices_per_worker
            assigned_indices = list(range(start_idx, end_idx))

            worker = mp.Process(
                target=reconstruction_worker,
                args=(
                    i,
                    assigned_indices,
                    pool_dir,
                    tilt_series_pickles,
                    tomogram_shape,
                    patch_size,
                    shift_generator,
                    ready_flag,
                    stop_event,
                ),
                kwargs={
                    'monitor': monitor,
                    'tilt_series_refresh_rate': tilt_series_refresh_rate,
                    'device': device,
                },
            )
            workers.append(worker)
            worker.start()

        # Wait for all workers to be ready
        print("Waiting for workers to initialize pool...")
        while ready_flag.value < n_workers:
            time.sleep(0.5)
        print(f"All {n_workers} workers ready!")

        # Benchmark
        initial_rates = monitor.get_rates()
        print(f"\nBenchmarking for {benchmark_duration} seconds...")
        time.sleep(benchmark_duration)

        final_rates = monitor.get_rates()
        productions = final_rates['total_produced'] - initial_rates[
            'total_produced']

        print(f"\nResults:")
        print(f"  Total productions: {productions}")
        print(f"  Rate: {productions / benchmark_duration:.2f} recons/sec")
        print(
            f"  Per worker: {productions / (benchmark_duration * n_workers):.2f} recons/sec")

        # Cleanup
        stop_event.set()
        for worker in workers:
            worker.join(timeout=5), "Worker failed to stop"


