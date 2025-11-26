# ML Infrastructure & Systems Coding Interview Practice

**Target Role:** ML Engineer (Infra), Systems Engineer, Production Engineer  
**Constraints:** Use **Python Standard Library Only** (No `numpy`, `pandas`, `torch` unless specified).  
**Goal:** Prove fluency in concurrency, memory management, and distributed system primitives.

---

## Topic 1: Concurrency & Synchronization
*Core Concept: Managing shared state without race conditions or deadlocks.*

1. **The Reader-Writer Lock**
   - **Task:** Implement a lock where multiple "readers" can access a resource simultaneously, but a "writer" gets exclusive access.
   - **Constraint:** Writers should not starve (wait forever) if new readers keep coming.
   
2. **The Distributed Barrier (Simulation)**
   - **Task:** Implement a synchronization primitive where `N` threads must wait until all `N` have arrived before *any* can proceed.
   - **ML Context:** Simulates `torch.distributed.barrier()` used in synchronous SGD.

3. **The Blocking Queue with Timeout**
   - **Task:** Implement a thread-safe queue with `put(item)` and `get(timeout=x)`. 
   - **Requirement:** If the queue is empty, `get` must wait `x` seconds before raising a custom `TimeoutError`.

4. **The "Hogwild" Counter (Atomic Operations)**
   - **Task:** 1. Simulate a shared counter updated by 100 threads to prove it fails (race condition).
     2. Implement a safe version using `threading.Lock`.
     3. Implement a "Compare-and-Swap" (CAS) version without a heavy lock (optimistic concurrency).

5. **The Semaphore Connection Pool**
   - **Task:** Implement a `ConnectionPool` class that manages a fixed set of dummy DB connections.
   - **Requirement:** Use a `threading.Semaphore`. If no connections are available, the caller blocks.

6. **The Deadlock Detector**
   - **Task:**
     1. Write code that intentionally creates a deadlock between two threads.
     2. Write a context manager `safe_lock(lock, timeout=5)` that attempts to acquire a lock and raises an error if it takes too long, preventing the hang.

---

## Topic 2: Scheduling & Flow Control
*Core Concept: Deciding "when" code runs based on time, dependencies, or capacity.*

7. **The Debounce Decorator**
   - **Task:** Write a decorator `@debounce(seconds=N)` that ensures a function is only called once per `N` seconds, even if triggered 100 times rapidly.

8. **The Token Bucket Rate Limiter**
   - **Task:** Implement a class `RateLimiter(rate_per_sec)`. 
   - **Logic:** Refill tokens at a specific rate. Requests consume tokens. If empty, return `False`.

9. **The Leaky Bucket**
   - **Task:** Similar to Token Bucket, but requests enter a queue and are processed at a constant fixed rate.
   - **Goal:** Smooth out "bursty" traffic into a consistent stream.

10. **The Delayed Task Scheduler**
    - **Task:** Implement a `Scheduler` class with `add_task(func, delay_ms)`.
    - **Constraint:** Do not spawn a thread per task. Use a single "Executor" thread and a Priority Queue (`heapq`) to sleep for the correct amount of time.

11. **The Dependency Graph (DAG) Runner**
    - **Task:** Given a list of tasks `[A, B, C]` and dependencies `[(A->B), (A->C)]`, execute them using a thread pool. 
    - **Rule:** B and C cannot start until A finishes.

12. **The Dynamic Batcher (Inference Simulation)**
    - **Task:** Implement `inference(input)`. 
    - **Logic:** When a request arrives, hold it for 50ms to see if other requests arrive. Group them into a batch of size `N`, run a dummy model once, and return results to individual callers.

---

## Topic 3: Memory & Caching
*Core Concept: Managing finite RAM, eviction policies, and object lifecycles.*

13. **The Thread-Safe LRU Cache**
    - **Task:** Implement a Least Recently Used cache with `get` and `put`.
    - **Constraint:** Must handle concurrent access from multiple threads.
    - **Hint:** `collections.OrderedDict` + `threading.Lock`.

14. **The TTL (Time-To-Live) Cache**
    - **Task:** A cache where every entry expires after `X` seconds.
    - **Challenge:** How do you clean up expired keys? (Lazy deletion vs. Background cleanup thread).

15. **The Circular Buffer**
    - **Task:** Implement a fixed-size ring buffer for streaming data. If full, overwrite the oldest data.
    - **ML Context:** Storing the last 1000 steps of loss metrics.

16. **The Object Pool (Arena)**
    - **Task:** Pre-allocate a list of 1000 dummy objects. Write `alloc()` and `free()` methods to reuse them instead of creating new ones.
    - **Goal:** Reduce Garbage Collection overhead.

17. **The "External" Sort**
    - **Task:** Simulate sorting a list too big for RAM.
    - **Logic:** Break list into chunks -> Sort chunks -> Merge sorted chunks (Merge k-Sorted Lists) using `heapq`.

---

## Topic 4: I/O & Streams
*Core Concept: Efficiently handling files, logs, and data pipelines.*

18. **The "Tail -f" Implementation**
    - **Task:** Write a generator that monitors a specific file. When a new line is written by another process, yield it immediately.

19. **The Log Merger**
    - **Task:** You have `N` generators, each yielding sorted timestamps. Create a single generator that yields them in global sorted order.

20. **The Chunked File Reader**
    - **Task:** Write a class that reads a binary file in fixed-size chunks (e.g., 4KB) but provides a `read_byte()` interface to the user.

21. **The Line Indexer (Sparse Index)**
    - **Task:** Read a huge text file once. Build an index mapping `Line Number -> Byte Offset`. 
    - **Goal:** Allow O(1) random access to any line using `file.seek()`.

---

## Topic 5: Distributed Primitives (Single Node Simulations)**
*Core Concept: Simulating consensus, failure detection, and distribution.*

22. **The Heartbeat Monitor**
    - **Task:** A "Server" class tracks "Worker" threads. Workers must call `heartbeat()` every 5s. If a worker misses it, the Server marks it as dead.

23. **The Consistent Hash Map**
    - **Task:** Map Keys to `N` Nodes (servers).
    - **Requirement:** Adding a new Node should only reshuffle `1/N` keys, not all keys.

24. **The Leader Election**
    - **Task:** Simulate 5 threads trying to write to a shared variable "LEADER". Only one succeeds. 
    - **Challenge:** If the leader thread dies, the others must detect it and elect a new one.

25. **The MapReduce Engine (Local)**
    - **Task:** - Input: List of strings.
      - Map: 4 threads count words in their assigned chunk.
      - Reduce: 1 thread aggregates the counts.

---

## How to Practice

1. **Pick one problem.**
2. **Set a timer for 25 minutes.**
3. **Open a blank Python file.**
4. **Code it using only `import threading, queue, collections, time, heapq`.**
5. **Self-Correction:** Write a `if __name__ == "__main__":` block that spawns 5-10 threads to hammer your function. If it crashes or produces the wrong number, you fail.