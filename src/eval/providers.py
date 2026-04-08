from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass

from .config import EvalConfig


@dataclass
class ProviderResult:
    generation: str
    transcription: str | None = None


def _make_openai_client(provider: str):
    """Create an AsyncOpenAI client for the given provider."""
    from openai import AsyncOpenAI

    if provider == "openrouter":
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    # Default: standard OpenAI (uses OPENAI_API_KEY)
    return AsyncOpenAI()


def load_audio_as_base64(audio_path: str) -> str:
    """Load audio file and encode as base64."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_voice_path(config: EvalConfig, dialog_id: str) -> str:
    """Get voice file path from dialog_id."""
    pattern = config.input.voice.file_pattern.format(dialog_id=dialog_id)
    return os.path.join(config.input.voice.data_dir, pattern)


async def _retry_on_429(coro_fn, max_retries=3, base_delay=10):
    """Retry an async callable on 429 rate limit errors with exponential backoff."""
    from openai import RateLimitError

    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except RateLimitError:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)


async def generate_text_openai(
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
    provider: str = "openai",
) -> ProviderResult:
    """Generate response via OpenAI-compatible text API (OpenAI or OpenRouter)."""
    client = _make_openai_client(provider)

    async def _call():
        async with semaphore:
            return await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

    response = await _retry_on_429(_call)
    return ProviderResult(generation=response.choices[0].message.content)


async def generate_voice_openai(
    prompt: str,
    audio_path: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> ProviderResult:
    """Generate response via OpenAI with audio input."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    audio_b64 = load_audio_as_base64(audio_path)

    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                    ],
                }
            ],
        )
    return ProviderResult(generation=response.choices[0].message.content)


async def generate_voice_gemini(
    prompt: str,
    audio_path: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> ProviderResult:
    """Generate response via Gemini with native audio input."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    async with semaphore:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[
                prompt,
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            ],
            config=types.GenerateContentConfig(temperature=0),
        )
    return ProviderResult(generation=response.text)


async def transcribe_with_deepgram(audio_path: str, model: str = "nova-3") -> str:
    """Transcribe audio using Deepgram."""
    from deepgram import DeepgramClient

    client = DeepgramClient()
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.listen.v1.media.transcribe_file(
            request=audio_data,
            model=model,
        ),
    )
    return response.results.channels[0].alternatives[0].transcript


async def generate_voice_cascade(
    prompt: str,
    audio_path: str,
    stt_model: str,
    llm_model: str,
    semaphore: asyncio.Semaphore,
) -> ProviderResult:
    """Cascade: Deepgram STT -> LLM. Returns generation + transcription."""
    from openai import AsyncOpenAI

    transcribed_text = await transcribe_with_deepgram(audio_path, model=stt_model)

    client = AsyncOpenAI()
    async with semaphore:
        response = await client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
    return ProviderResult(
        generation=response.choices[0].message.content,
        transcription=transcribed_text,
    )


def create_generate_fn(config: EvalConfig):
    """Factory that returns the appropriate async generate function based on config.

    Returns a callable with signature:
        async def generate(prompt: str, sample: dict, semaphore: asyncio.Semaphore) -> ProviderResult
    """
    input_type = config.input.type
    provider = config.model.provider
    backend = config.model.backend

    if input_type == "text":
        model_name = config.model.name

        async def generate(prompt: str, sample: dict, semaphore: asyncio.Semaphore) -> ProviderResult:
            return await generate_text_openai(prompt, model_name, semaphore, provider=provider)

        return generate

    # Voice input
    if backend == "openai":
        model_name = config.model.audio_model

        async def generate(prompt: str, sample: dict, semaphore: asyncio.Semaphore) -> ProviderResult:
            audio_path = get_voice_path(config, sample["dialog_id"])
            if not os.path.exists(audio_path):
                return ProviderResult(generation="", transcription=None)
            return await generate_voice_openai(prompt, audio_path, model_name, semaphore)

        return generate

    if backend == "gemini":
        model_name = config.model.gemini_model

        async def generate(prompt: str, sample: dict, semaphore: asyncio.Semaphore) -> ProviderResult:
            audio_path = get_voice_path(config, sample["dialog_id"])
            if not os.path.exists(audio_path):
                return ProviderResult(generation="", transcription=None)
            return await generate_voice_gemini(prompt, audio_path, model_name, semaphore)

        return generate

    if backend == "cascade":
        stt_model = config.model.cascade.stt_model
        llm_model = config.model.cascade.llm_model

        async def generate(prompt: str, sample: dict, semaphore: asyncio.Semaphore) -> ProviderResult:
            audio_path = get_voice_path(config, sample["dialog_id"])
            if not os.path.exists(audio_path):
                return ProviderResult(generation="", transcription=None)
            return await generate_voice_cascade(prompt, audio_path, stt_model, llm_model, semaphore)

        return generate

    raise ValueError(f"Unsupported backend: {backend} with input type: {input_type}")
