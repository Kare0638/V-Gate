#!/usr/bin/env python3
"""
Concurrent testing script for V-Gate Phase 2 features.

Tests:
1. Dynamic request batching - multiple concurrent requests
2. Result caching - duplicate requests should hit cache
3. Batch deduplication - identical prompts in same batch

Usage:
    python scripts/test_concurrent.py

Requirements:
    pip install aiohttp
"""
import asyncio
import aiohttp
import time
from time import monotonic
import sys


BASE_URL = "http://localhost:8000"


async def send_chat_request(session: aiohttp.ClientSession, prompt: str) -> dict:
    """Send a chat completion request."""
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}]
    }
    async with session.post(url, json=payload) as resp:
        return await resp.json()


async def get_metrics(session: aiohttp.ClientSession) -> dict:
    """Get current metrics."""
    async with session.get(f"{BASE_URL}/metrics") as resp:
        return await resp.json()


async def test_batching(session: aiohttp.ClientSession):
    """Test dynamic request batching with concurrent requests."""
    print("\n" + "=" * 50)
    print("TEST 1: Dynamic Request Batching")
    print("=" * 50)

    # Get initial metrics
    initial = await get_metrics(session)
    initial_batches = initial["batcher"]["total_batches"]
    initial_requests = initial["batcher"]["total_requests"]

    # Send 10 concurrent requests with different prompts
    prompts = [f"What is {i} + {i}?" for i in range(10)]

    start = monotonic()
    tasks = [send_chat_request(session, p) for p in prompts]
    results = await asyncio.gather(*tasks)
    elapsed = monotonic() - start

    # Get final metrics
    final = await get_metrics(session)
    new_batches = final["batcher"]["total_batches"] - initial_batches
    new_requests = final["batcher"]["total_requests"] - initial_requests

    print(f"Sent: {len(prompts)} concurrent requests")
    print(f"Time: {elapsed:.2f}s")
    print(f"New batches created: {new_batches}")
    print(f"Average batch size: {new_requests / new_batches:.1f}" if new_batches > 0 else "N/A")
    print(f"Result: {'PASS' if new_batches < len(prompts) else 'CHECK'} (requests were batched)")

    return results


async def test_caching(session: aiohttp.ClientSession):
    """Test result caching with duplicate requests."""
    print("\n" + "=" * 50)
    print("TEST 2: Result Caching")
    print("=" * 50)

    # Get initial metrics
    initial = await get_metrics(session)
    initial_hits = initial["cache"]["hits"]

    prompt = "Tell me a joke about programming."

    # First request - should be cache miss
    print("Request 1 (cache miss expected)...")
    start = monotonic()
    result1 = await send_chat_request(session, prompt)
    time1 = monotonic() - start

    # Second request - should be cache hit
    print("Request 2 (cache hit expected)...")
    start = monotonic()
    result2 = await send_chat_request(session, prompt)
    time2 = monotonic() - start

    # Third request - should be cache hit
    print("Request 3 (cache hit expected)...")
    start = monotonic()
    result3 = await send_chat_request(session, prompt)
    time3 = monotonic() - start

    # Get final metrics
    final = await get_metrics(session)
    new_hits = final["cache"]["hits"] - initial_hits

    print(f"\nRequest 1 time: {time1:.3f}s")
    print(f"Request 2 time: {time2:.3f}s (should be < 0.01s if cached)")
    print(f"Request 3 time: {time3:.3f}s (should be < 0.01s if cached)")
    print(f"Cache hits: {new_hits}")
    print(f"Speedup: {time1 / time2:.1f}x" if time2 > 0 else "N/A")
    print(f"Result: {'PASS' if new_hits >= 2 and time2 < 0.1 else 'FAIL'}")


async def test_batch_dedup(session: aiohttp.ClientSession):
    """Test batch deduplication with identical concurrent requests."""
    print("\n" + "=" * 50)
    print("TEST 3: Batch Deduplication")
    print("=" * 50)

    # Use a unique prompt to avoid cache from previous tests
    unique_prompt = f"What is the meaning of life? (test-{time.time()})"

    # Get initial metrics
    initial = await get_metrics(session)
    initial_batches = initial["batcher"]["total_batches"]
    initial_cache_size = initial["cache"]["size"]

    # Send 5 identical requests concurrently
    # They should all be in the same batch, deduplicated to 1 inference
    print("Sending 5 identical requests concurrently...")

    start = monotonic()
    tasks = [send_chat_request(session, unique_prompt) for _ in range(5)]
    results = await asyncio.gather(*tasks)
    elapsed = monotonic() - start

    # Get final metrics
    final = await get_metrics(session)
    new_batches = final["batcher"]["total_batches"] - initial_batches
    new_cache_entries = final["cache"]["size"] - initial_cache_size

    # All results should be identical
    texts = [r["choices"][0]["message"]["content"] for r in results]
    all_same = all(t == texts[0] for t in texts)

    # Deduplication success criteria:
    # 1. All responses identical (same inference result shared)
    # 2. Only 1 new cache entry (only 1 unique prompt was inferred)
    # 3. Processed in 1 batch (all requests batched together)
    dedup_success = all_same and new_cache_entries == 1 and new_batches == 1

    print(f"Time: {elapsed:.2f}s")
    print(f"New batches: {new_batches} (should be 1)")
    print(f"New cache entries: {new_cache_entries} (should be 1 if deduplicated)")
    print(f"All responses identical: {all_same}")
    print(f"Result: {'PASS' if dedup_success else 'FAIL'}")


async def show_final_stats(session: aiohttp.ClientSession):
    """Display final statistics."""
    print("\n" + "=" * 50)
    print("FINAL STATISTICS")
    print("=" * 50)

    metrics = await get_metrics(session)

    print("\nBatcher:")
    for key, value in metrics["batcher"].items():
        print(f"  {key}: {value}")

    print("\nCache:")
    for key, value in metrics["cache"].items():
        print(f"  {key}: {value}")

    print("\nConfig:")
    print(f"  Batch: {metrics['config']['batch']}")
    print(f"  Cache: {metrics['config']['cache']}")


async def check_server():
    """Check if server is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    return True
    except Exception:
        pass
    return False


async def main():
    print("V-Gate Phase 2 Feature Test")
    print("=" * 50)

    # Check server
    print(f"Checking server at {BASE_URL}...")
    if not await check_server():
        print(f"ERROR: Server not running at {BASE_URL}")
        print("Please start the server first: python main.py")
        sys.exit(1)
    print("Server is running!\n")

    async with aiohttp.ClientSession() as session:
        # Run tests
        await test_batching(session)
        await test_caching(session)
        await test_batch_dedup(session)
        await show_final_stats(session)

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
