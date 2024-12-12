"""
Asynchronous task processing system with rate limiting and progress tracking.

Key components and functionality:

1. SharedRateLimiter: 
   - Implements a token bucket algorithm for rate limiting.
   - Ensures that tasks are processed at a specified rate (e.g., 10 requests per second).

2. Worker Function:
   - Asynchronously processes tasks from a queue.
   - Respects the rate limit using the SharedRateLimiter.
   - Handles exceptions and logs task completion or errors.

3. Progress Tracking:
   - Uses tqdm to display a progress bar.
   - Updates the progress bar asynchronously based on the number of completed tasks.

4. Main Process (process_tasks):
   - Creates a queue and populates it with tasks.
   - Initializes multiple worker coroutines to process tasks concurrently.
   - Manages the lifecycle of workers and the progress tracking task.
   - Ensures all tasks are completed before terminating.

5. Rate Limiting:
   - Prevents overwhelming the target system (e.g., an API) by limiting the request rate.
   - Allows for smooth, controlled task processing.

6. Concurrency:
   - Utilizes Python's asyncio for concurrent task processing.
   - Allows handling multiple tasks simultaneously without using multiple threads or processes.

7. Error Handling:
   - Catches and logs exceptions during task processing.
   - Ensures that an error in one task doesn't stop the entire process.

8. Result Order Preservation:
   - Maintains the original order of tasks in the results, regardless of completion order.

Usage:
- Define your tasks as a list of items to be processed.
- Create an async function (worker_func) that processes a single task.
- Set the desired rate limit and number of concurrent workers.
- Call process_tasks with your tasks, worker function, rate limit, and worker count.

This script is particularly useful for scenarios involving bulk API requests, 
data processing, or any situation where you need to process a large number of 
tasks concurrently while respecting rate limits and preserving task order.
"""

import asyncio
import time
from asyncio import Queue
from typing import List, Any, Coroutine
from tqdm import tqdm
import inspect

class SharedRateLimiter:
    def __init__(self, rate_limit: float):
        self.rate_limit = rate_limit
        self.tokens = 0
        self.last_refill = time.monotonic()

    async def acquire(self):
        while True:
            now = time.monotonic()
            time_passed = now - self.last_refill
            self.tokens += time_passed * self.rate_limit
            if self.tokens > self.rate_limit:
                self.tokens = self.rate_limit
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return
            else:
                await asyncio.sleep(0.01)


async def worker(queue: Queue, rate_limiter: SharedRateLimiter, worker_func: Coroutine, results: List[Any], worker_id: int):
    while True:
        try:
            index, task = await queue.get()
            await rate_limiter.acquire()
            try:
                result = await worker_func(task)
                # print(f"Worker {worker_id} completed task {index}")
            except Exception as e:
                result = f"Error processing task {task}: {str(e)}"
                print(f"Worker {worker_id} encountered error on task {index}: {e}")
            results[index] = result
            queue.task_done()
        except asyncio.CancelledError:
            break


async def update_progress(queue: Queue, total: int, pbar: tqdm):
    while True:
        completed = total - queue.qsize()
        pbar.n = completed
        pbar.refresh()
        if completed == total:
            break
        await asyncio.sleep(0.1)


async def process_tasks(tasks: List[Any], worker_func: Coroutine, rate_limit: float, num_workers: int, desc: str = "Processing tasks") -> List[Any]:
    """
    Process a list of tasks concurrently with rate limiting and progress tracking.

    Parameters:
    tasks (List[Any]): A list of tasks to be processed.
    worker_func (Coroutine): An asynchronous function to process each task.
    rate_limit (float): The maximum number of tasks to process per second.
    num_workers (int): The number of concurrent worker tasks to run.
    desc (str): A description for the progress bar.

    Returns:
    List[Any]: A list of results corresponding to the processed tasks.
    """
    assert inspect.iscoroutinefunction(worker_func), "worker_func must be a coroutine (async function)"
    
    queue = Queue()
    rate_limiter = SharedRateLimiter(rate_limit)
    results = [None] * len(tasks)

    # Add all tasks to the queue with their original indices
    for index, task in enumerate(tasks):
        await queue.put((index, task))

    # Create progress bar
    pbar = tqdm(total=len(tasks), desc=desc)

    # Create worker tasks
    workers = [
        asyncio.create_task(worker(queue, rate_limiter, worker_func, results, i))
        for i in range(num_workers)
    ]

    # Create progress update task
    progress_task = asyncio.create_task(update_progress(queue, len(tasks), pbar))

    # Wait for all tasks to be completed
    await queue.join()

    # Cancel worker tasks and progress task
    for w in workers:
        w.cancel()
    progress_task.cancel()

    # Wait for all tasks to be cancelled
    await asyncio.gather(*workers, progress_task, return_exceptions=True)

    pbar.close()
    return results


if __name__ == "__main__":
    # Example usage:
    async def api_call(article_id: int):
        # Simulate API call with random delay to show out-of-order execution
        await asyncio.sleep(0.1 * (article_id % 5))
        return f"Result for article {article_id}"

    async def main():
        tasks = list(range(20))  # Just a list of article IDs
        rate_limit = 10  # 10 requests per second
        num_workers = 5  # Number of concurrent workers

        results = await process_tasks(tasks, api_call, rate_limit, num_workers)
        print(f"\nProcessed {len(results)} tasks")
        for task, result in zip(tasks, results):
            print(f"Task {task}: {result}")

        # Check for any None results
        none_results = [i for i, r in enumerate(results) if r is None]
        if none_results:
            print(f"Warning: Tasks {none_results} were not completed")

    asyncio.run(main())
