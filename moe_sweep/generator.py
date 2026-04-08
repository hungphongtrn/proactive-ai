import asyncio

from openai import AsyncOpenAI
from tqdm import tqdm

CHUNK_SIZE = 50
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry


async def generate_n(
    prompt: str,
    n: int,
    temperature: float,
    port: int,
    model: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 1024,
    sub_progress: tqdm | None = None,
) -> list[str]:
    """Generate N completions for a prompt using vLLM's OpenAI-compatible API.

    Uses the `n` parameter for efficient batched multi-sample generation.
    If n > CHUNK_SIZE, splits into chunks and updates sub_progress bar.
    """
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")

    async def _call(batch_n: int) -> list[str]:
        async with semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        n=batch_n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return [choice.message.content or "" for choice in response.choices]
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_BACKOFF * (2 ** attempt)
                        print(f"\nRetry {attempt+1}/{MAX_RETRIES} after {wait}s (n={batch_n}): {e}")
                        await asyncio.sleep(wait)
                    else:
                        print(f"\nFailed after {MAX_RETRIES} retries (n={batch_n}): {e}")
                        return [""] * batch_n

    if n <= CHUNK_SIZE:
        results = await _call(n)
        if sub_progress:
            sub_progress.update(n)
        return results

    # Split into chunks for sub-progress
    all_results = []
    remaining = n
    while remaining > 0:
        chunk = min(CHUNK_SIZE, remaining)
        results = await _call(chunk)
        all_results.extend(results)
        remaining -= chunk
        if sub_progress:
            sub_progress.update(chunk)
    return all_results


async def generate_batch(
    prompts: list[tuple[int, str]],
    n: int,
    temperature: float,
    port: int,
    model: str,
    max_concurrent: int = 32,
    max_tokens: int = 1024,
) -> dict[int, list[str]]:
    """Generate N completions for each prompt in a batch.

    Args:
        prompts: List of (index, prompt_text) tuples.
        n: Number of samples per prompt.
        temperature: Sampling temperature.
        port: vLLM server port.
        model: Model name for the API.
        max_concurrent: Max concurrent requests.
        max_tokens: Max tokens per generation.

    Returns:
        Dict mapping index -> list of N generation strings.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    total_generations = len(prompts) * n
    show_sub = n > CHUNK_SIZE

    if show_sub:
        pbar = tqdm(total=total_generations, desc=f"Generating (t={temperature}, n={n})", unit="gen")

        async def _gen(idx: int, prompt: str):
            generations = await generate_n(prompt, n, temperature, port, model, semaphore, max_tokens, sub_progress=pbar)
            results[idx] = generations

        tasks = [_gen(idx, prompt) for idx, prompt in prompts]
        await asyncio.gather(*tasks)
        pbar.close()
    else:
        async def _gen(idx: int, prompt: str):
            generations = await generate_n(prompt, n, temperature, port, model, semaphore, max_tokens)
            results[idx] = generations

        tasks = [_gen(idx, prompt) for idx, prompt in prompts]

        from tqdm.asyncio import tqdm_asyncio
        await tqdm_asyncio.gather(*tasks, desc=f"Generating (t={temperature}, n={n})")

    return results
