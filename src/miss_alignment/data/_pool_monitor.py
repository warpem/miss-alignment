import time
import multiprocessing as mp


class SimplePoolMonitor:
    """
    Simple monitor for reconstruction pool performance.

    Tracks overall production and consumption rates across all workers.
    """

    def __init__(self):
        manager = mp.Manager()

        # Total counters - just use manager.Value without locks
        self.total_produced = manager.Value("Q", 0)
        self.total_consumed = manager.Value("Q", 0)

        self.start_time = time.time()

    def record_production(self):
        # Manager.Value is already thread-safe, no lock needed
        self.total_produced.value += 1

    def record_consumption(self, batch_size: int):
        # Manager.Value is already thread-safe
        self.total_consumed.value += batch_size

    def get_rates(self) -> dict:
        """
        Calculate production and consumption rates.

        Parameters
        ----------
        window_seconds : float
            Time window for rate calculation

        Returns
        -------
        dict
            Production and consumption rates
        """
        now = time.time()
        elapsed = now - self.start_time
        production_rate = self.total_produced.value / elapsed
        consumption_rate = self.total_consumed.value / elapsed

        return {
            "production_rate": production_rate,
            "consumption_rate": consumption_rate,
            "total_produced": self.total_produced.value,
            "total_consumed": self.total_consumed.value,
            "uptime": elapsed,
        }

    def print_stats(self):
        """Print current statistics."""
        stats = self.get_rates()

        print("\n=== Pool Statistics ===")
        print(f"Production rate: {stats['production_rate']:.2f} files/sec")
        print(f"Consumption rate: {stats['consumption_rate']:.2f} files/sec")
        print(f"Total produced: {stats['total_produced']}")
        print(f"Total consumed: {stats['total_consumed']}")
        print(f"Uptime: {stats['uptime']:.1f}s")

        ratio = (
            stats["consumption_rate"] / stats["production_rate"]
            if stats["production_rate"] > 0
            else 0
        )
        print(f"Consumption/Production ratio: {ratio:.2f}")
