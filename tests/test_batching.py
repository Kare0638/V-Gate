#!/usr/bin/env python3
"""
Concurrent request test for dynamic batching.
Sends multiple requests simultaneously to verify batching behavior.

Usage:
    python tests/test_batching.py [num_requests]

Default: 10 concurrent requests
"""

import asyncio
import aiohttp
import time
import sys

API_URL = "http://localhost:8000/v1/chat/completions"
METRICS_URL = "http://localhost:8000/metrics"


async def send_request(session: aiohttp.ClientSession, request_id: int) -> dict:
    """Send a single chat completion request."""
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        "messages": [
            {"role": "user", "content": f"Hello, this is request #{request_id}. Give me a short greeting."}
        ]
    }

    start_time = time.perf_counter()
    async with session.post(API_URL, json=payload) as response:
        result = await response.json()
        end_time = time.perf_counter()

    return {
        "request_id": request_id,
        "latency_ms": (end_time - start_time) * 1000,
        "success": "choices" in result,
        "response_preview": result.get("choices", [{}])[0].get("message", {}).get("content", "")[:50] + "..."
    }


async def get_metrics(session: aiohttp.ClientSession) -> dict:
    """Get batching metrics from the server."""
    async with session.get(METRICS_URL) as response:
        return await response.json()


async def run_concurrent_test(num_requests: int):
    """Run concurrent requests and report results."""
    print(f"\n{'='*60}")
    print(f"Dynamic Batching Test - {num_requests} Concurrent Requests")
    print(f"{'='*60}\n")

    # Get initial metrics
    async with aiohttp.ClientSession() as session:
        initial_metrics = await get_metrics(session)
        print(f"Initial metrics: {initial_metrics['batcher']}")
        print(f"Batch config: {initial_metrics['config']}\n")

        print(f"Sending {num_requests} concurrent requests...")
        start_time = time.perf_counter()

        # Send all requests concurrently
        tasks = [send_request(session, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

        # Get final metrics
        final_metrics = await get_metrics(session)

    # Analyze results
    latencies = [r["latency_ms"] for r in results]
    successful = sum(1 for r in results if r["success"])

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total requests:     {num_requests}")
    print(f"Successful:         {successful}")
    print(f"Total time:         {total_time*1000:.2f} ms")
    print(f"Avg latency:        {sum(latencies)/len(latencies):.2f} ms")
    print(f"Min latency:        {min(latencies):.2f} ms")
    print(f"Max latency:        {max(latencies):.2f} ms")

    print(f"\n{'='*60}")
    print("BATCHING METRICS")
    print(f"{'='*60}")
    batcher_metrics = final_metrics["batcher"]
    print(f"Total requests:     {batcher_metrics['total_requests']}")
    print(f"Total batches:      {batcher_metrics['total_batches']}")
    print(f"Avg batch size:     {batcher_metrics['average_batch_size']}")

    # Calculate efficiency
    new_requests = batcher_metrics['total_requests'] - initial_metrics['batcher']['total_requests']
    new_batches = batcher_metrics['total_batches'] - initial_metrics['batcher']['total_batches']

    if new_batches > 0:
        actual_batch_size = new_requests / new_batches
        print(f"\nThis test:")
        print(f"  Requests sent:    {new_requests}")
        print(f"  Batches used:     {new_batches}")
        print(f"  Actual batch size: {actual_batch_size:.1f}")

        # Efficiency: 1 batch for N requests is better than N batches
        efficiency = (1 - (new_batches / new_requests)) * 100
        print(f"  Batching efficiency: {efficiency:.1f}%")

    print(f"\n{'='*60}")
    print("SAMPLE RESPONSES")
    print(f"{'='*60}")
    for r in results[:3]:
        print(f"  Request #{r['request_id']}: {r['response_preview']}")

    return results


if __name__ == "__main__":
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(run_concurrent_test(num_requests))
