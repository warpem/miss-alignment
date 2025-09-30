#!/usr/bin/env python3
"""
Pool-based 3D reconstruction system with atomic file replacement.

This script demonstrates a system where:
- Multiple workers maintain a pool of pre-computed 3D reconstructions
- DataLoaders fetch from this pool without reconstruction overhead
- File access is safe using atomic rename operations
"""

import os
import time
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple
import random
import signal
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class ReconstructionPoolDataset(Dataset):
    """
    Dataset that reads pre-computed reconstructions from a pool.
    
    The pool files are constantly being updated by background workers,
    but reads are always safe due to atomic rename operations.
    
    Parameters
    ----------
    pool_dir : Path
        Directory containing the reconstruction pool files
    pool_size : int
        Number of reconstructions in the pool
    """
    
    def __init__(self, pool_dir: Path, pool_size: int):
        self.pool_dir = pool_dir
        self.pool_size = pool_size
        
    def __len__(self) -> int:
        return self.pool_size
    
    def __getitem__(self, idx: int) -> dict:
        """
        Fetch a reconstruction from the pool.
        
        Parameters
        ----------
        idx : int
            Index of the reconstruction to fetch
            
        Returns
        -------
        dict
            Dictionary containing the reconstruction data
        """
        file_path = self.pool_dir / f"recon_{idx}.npz"
        
        # This should always succeed due to atomic rename
        data = np.load(file_path, allow_pickle=True)
        
        return {
            'reconstruction': torch.from_numpy(data['reconstruction']).float(),
            'metadata': data['metadata'].item()
        }


def create_mock_reconstruction(source_id: int) -> Tuple[np.ndarray, dict]:
    """
    Simulate expensive 3D reconstruction from 2D images.
    
    Parameters
    ----------
    source_id : int
        ID of the source images to reconstruct from
        
    Returns
    -------
    reconstruction : np.ndarray
        Mock 3D reconstruction (32x32x32 volume)
    metadata : dict
        Metadata about the reconstruction
    """
    # Simulate expensive computation
    time.sleep(0.1 + random.random() * 0.1)
    
    # Create mock 3D volume
    reconstruction = np.random.randn(32, 32, 32).astype(np.float32)
    reconstruction *= (source_id % 10 + 1)  # Scale by source_id for variety
    
    metadata = {
        'source_id': source_id,
        'timestamp': time.time(),
        'quality': random.random()
    }
    
    return reconstruction, metadata


def reconstruction_worker(
    worker_id: int,
    assigned_indices: list,
    pool_dir: Path,
    ready_flag: mp.Value,
    stop_event: mp.Event,
    source_counter: mp.Value
):
    """
    Worker process that maintains a subset of the reconstruction pool.
    
    Parameters
    ----------
    worker_id : int
        ID of this worker
    assigned_indices : list
        Pool indices assigned to this worker
    pool_dir : Path
        Directory to store reconstructions
    ready_flag : mp.Value
        Shared flag indicating pool is ready
    stop_event : mp.Event
        Event to signal worker shutdown
    source_counter : mp.Value
        Shared counter for source IDs
    """
    print(f"Worker {worker_id} starting with indices {assigned_indices[:5]}...")
    
    # Initial fill of assigned pool slots
    for idx in assigned_indices:
        with source_counter.get_lock():
            source_id = source_counter.value
            source_counter.value += 1
        
        recon, meta = create_mock_reconstruction(source_id)
        
        # Write directly first time (no temp file needed)
        file_path = pool_dir / f"recon_{idx:04d}.npz"
        np.savez(file_path, reconstruction=recon, metadata=meta)
    
    print(f"Worker {worker_id} completed initial fill")
    
    # Signal ready after initial fill
    with ready_flag.get_lock():
        ready_flag.value += 1
    
    # Continuously update pool with new reconstructions
    while not stop_event.is_set():
        # Randomly select one of our indices to update
        idx = random.choice(assigned_indices)
        
        with source_counter.get_lock():
            source_id = source_counter.value
            source_counter.value += 1
        
        recon, meta = create_mock_reconstruction(source_id)
        
        # Atomic replacement using temp file and rename
        file_path = pool_dir / f"recon_{idx:04d}.npz"
        with tempfile.NamedTemporaryFile(
            dir=pool_dir, 
            prefix=f"tmp_recon_{idx:04d}_",
            suffix='.npz',
            delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            np.savez(tmp_path, reconstruction=recon, metadata=meta)
        
        # Atomic rename (replaces existing file)
        tmp_path.rename(file_path)
    
    print(f"Worker {worker_id} shutting down")


class ReconstructionDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for pool-based reconstruction loading.
    
    Parameters
    ----------
    pool_size : int
        Size of the reconstruction pool
    num_workers : int
        Number of workers maintaining the pool
    batch_size : int
        Batch size for DataLoaders
    num_dataloader_workers : int
        Number of workers for DataLoader
    """
    
    def __init__(
        self,
        pool_size: int = 1000,
        num_workers: int = 4,
        batch_size: int = 32,
        num_dataloader_workers: int = 2
    ):
        super().__init__()
        self.pool_size = pool_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        
        self.pool_dir: Optional[Path] = None
        self.pool_processes: list = []
        self.stop_event: Optional[mp.Event] = None

    def __enter__(self):
        """Enter context manager and setup pool."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup."""
        self.teardown()
        return False
        
    def setup(self, stage: Optional[str] = None):
        """Initialize the reconstruction pool."""
        # Create temporary directory for pool
        self.pool_dir = Path(tempfile.mkdtemp(prefix='recon_pool_'))
        print(f"Created pool directory: {self.pool_dir}")
        
        # Partition indices among workers
        indices_per_worker = self.pool_size // self.num_workers
        partitions = []
        for i in range(self.num_workers):
            start_idx = i * indices_per_worker
            if i == self.num_workers - 1:
                # Last worker gets any remaining indices
                end_idx = self.pool_size
            else:
                end_idx = start_idx + indices_per_worker
            partitions.append(list(range(start_idx, end_idx)))
        
        # Start worker processes
        self.stop_event = mp.Event()
        ready_flag = mp.Value('i', 0)
        source_counter = mp.Value('i', 0)
        
        for worker_id, indices in enumerate(partitions):
            p = mp.Process(
                target=reconstruction_worker,
                args=(
                    worker_id,
                    indices,
                    self.pool_dir,
                    ready_flag,
                    self.stop_event,
                    source_counter
                )
            )
            p.start()
            self.pool_processes.append(p)
        
        # Wait for all workers to complete initial fill
        print("Waiting for pool to be filled...")
        while ready_flag.value < self.num_workers:
            time.sleep(0.1)
        print("Pool ready!")
        
        # Create dataset
        self.dataset = ReconstructionPoolDataset(self.pool_dir, self.pool_size)
    
    def train_dataloader(self):
        """Create training DataLoader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_dataloader_workers,
            persistent_workers=True if self.num_dataloader_workers > 0 else False
        )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up pool and worker processes."""
        if self.stop_event:
            print("Stopping reconstruction workers...")
            self.stop_event.set()
            
            for p in self.pool_processes:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
                    p.join()
        
        if self.pool_dir and self.pool_dir.exists():
            import shutil
            shutil.rmtree(self.pool_dir)
            print(f"Cleaned up pool directory: {self.pool_dir}")

        print("Cleanup complete")


def main():
    """Demonstrate the pool-based reconstruction system."""
    print("Starting pool-based reconstruction system demo\n")
    
    # Create data module
    with ReconstructionDataModule(
        pool_size=100,  # Small pool for demo
        num_workers=3,
        batch_size=8,
        num_dataloader_workers=2
    ) as dm:
    
        # Create dataloader
        train_loader = dm.train_dataloader()
    
        try:
            # Fetch some batches to demonstrate it works
            print("\nFetching batches from pool:")
            for i, batch in enumerate(train_loader):
                if i >= 5:  # Just show 5 batches
                    break

                recon = batch['reconstruction']
                meta = batch['metadata']
                print(f"Batch {i}: shape={recon.shape}, "
                      f"source_ids={meta['source_id'][:3]}...")

                # Simulate training step
                time.sleep(0.5)

            print("\nSystem working correctly!")
            print("Pool is being continuously updated in background.")
            print("\nPress Ctrl+C to stop...")

            # Keep running to show continuous updates
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    # Handle signals gracefully
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    main()
