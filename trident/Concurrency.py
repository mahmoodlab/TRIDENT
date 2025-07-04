import os
import gc
import torch
import shutil
from typing import List, Callable
from queue import Queue



def cache_batch(wsis: List[str], dest_dir: str) -> List[str]:
    """
    Copies WSIs to a local cache directory. Handles .mrxs subdirectories if present.

    Returns:
        List[str]: Paths to copied WSIs.
    """
    os.makedirs(dest_dir, exist_ok=True)
    copied = []

    for wsi in wsis:
        dest_path = os.path.join(dest_dir, os.path.basename(wsi))
        shutil.copy(wsi, dest_path)
        copied.append(dest_path)

        if wsi.endswith('.mrxs'):
            mrxs_dir = os.path.splitext(wsi)[0]
            if os.path.exists(mrxs_dir):
                dest_mrxs_dir = os.path.join(dest_dir, os.path.basename(mrxs_dir))
                shutil.copytree(mrxs_dir, dest_mrxs_dir)

    return copied


def batch_producer(
    queue: Queue,
    valid_slides: List[str],
    start_idx: int,
    batch_size: int,
    cache_dir: str,
) -> None:
    """
    Produces and caches batches of slides. Sends batch IDs to a queue for downstream processing.

    Args:
        queue (Queue): Queue to communicate with the consumer.
        valid_slides (List[str]): List of valid WSI paths.
        start_idx (int): Index in `valid_slides` to start batching from.
        batch_size (int): Number of slides per batch.
        cache_dir (str): Root directory where batches will be cached.
    """
    for i in range(start_idx, len(valid_slides), batch_size):
        batch_paths = valid_slides[i:i + batch_size]
        batch_id = i // batch_size
        ssd_batch_dir = os.path.join(cache_dir, f"batch_{batch_id}")
        print(f"[PRODUCER] Caching batch {batch_id}: {ssd_batch_dir}")
        cache_batch(batch_paths, ssd_batch_dir)
        queue.put(batch_id)

    queue.put(None)  # Sentinel to signal completion


def batch_consumer(
    queue: Queue,
    task: str,
    cache_dir: str,
    processor_factory: Callable[[str], object],
    run_task_fn: Callable[[object, str], None],
) -> None:
    """
    Consumes cached batches from the queue, processes them, and optionally clears cache.

    Args:
        queue (Queue): Queue from the producer.
        task (str): Task name ('seg', 'coords', 'feat', or 'all').
        cache_dir (str): Directory containing cached batches.
        processor_factory (Callable): Function that creates a processor given a WSI dir.
        run_task_fn (Callable): Function to run a task given a processor and task name.
    """

    while True:
        batch_id = queue.get()
        if batch_id is None:
            queue.task_done()
            break

        ssd_batch_dir = os.path.join(cache_dir, f"batch_{batch_id}")
        print(f"[CONSUMER] Processing batch {batch_id}: {ssd_batch_dir}")

        processor = processor_factory(ssd_batch_dir)

        try:
            if task == 'all':
                for subtask in ['seg', 'coords', 'feat']:
                    run_task_fn(processor, subtask)
            else:
                run_task_fn(processor, task)
        finally:
            # release all WSI and processor resources
            if hasattr(processor, "release"):
                processor.release()
            del processor
            gc.collect()
            torch.cuda.empty_cache()

            print(f"[CONSUMER] Clearing cache for batch {batch_id}")
            shutil.rmtree(ssd_batch_dir, ignore_errors=True)
            queue.task_done()
