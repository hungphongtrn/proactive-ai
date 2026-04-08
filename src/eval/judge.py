from __future__ import annotations

import asyncio
from pathlib import Path

from .parsing import extract_tag_content


async def judge_implicature(
    speech: str,
    pred_implicature: str,
    gt_implicature: str,
    model: str,
    prompt_template: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Judge if predicted implicature matches groundtruth in meaning.

    Returns dict with implicature_judge_output, implicature_reasoning, implicature_score.
    """
    from openai import AsyncOpenAI

    prompt = prompt_template.format(
        speech=speech,
        predicted_implicature=pred_implicature,
        groundtruth_implicature=gt_implicature,
    )

    client = AsyncOpenAI()
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

    output = response.choices[0].message.content
    score_str = extract_tag_content(output, "score")
    score = 1 if score_str.strip() == "1" else 0

    return {
        "implicature_judge_output": output,
        "implicature_reasoning": extract_tag_content(output, "reasoning"),
        "implicature_score": score,
    }


def load_judge_prompt(prompt_path: str) -> str:
    """Load judge prompt template from file."""
    return Path(prompt_path).read_text()
