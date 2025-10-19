from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer

app = FastAPI()

SYSTEM_PROMPT = """You are a behavioral scientist who deeply understands human communication patterns, both constructive and destructive. You will evaluate whether a response appropriately addresses the user's underlying intent and emotional state, not just their surface-level words.

<definitions>
**Intent** represents what the user truly wants to achieve or communicate through their speech. It's the underlying goal or purpose behind their words.

**Emotion** represents the emotional state the user is experiencing when saying something. It reflects their feelings and psychological state.

A good response demonstrates understanding of both the intent and emotion, addressing what the user truly needs rather than just what they literally said.

**Scoring criteria:**
- Score 1.0 if the response clearly demonstrates understanding of both the intent and emotion, and addresses the user's underlying needs appropriately. The response shows empathy and provides relevant, contextually appropriate content.

- Score 0.5 if the response partially addresses the intent or emotion but misses important nuances, or if it's generic and safe but not deeply aligned with the user's emotional state and intent.

- Score 0.0 if the response completely misunderstands the intent and emotion, addresses only the surface-level words without grasping the context, or provides an inappropriate response that ignores what the user is really indicating.
</definitions>

<output_format>
First, think through your evaluation inside <think></think> tags by analyzing:
1. What is the user really trying to communicate?
2. Does the response show understanding of the intent?
3. Does the response appropriately address the emotional state?
4. Is the response contextually appropriate?

Then output your score inside <score></score> tags using exactly one of these values: 0.0, 0.5, or 1.0

Example:
<think>Your reasoning here...</think>
<score>1.0</score>
</output_format>
"""

USER_INPUT = """User's speech: {user_speech}
Intent: {intent}
Emotion: {emotion}
Response to evaluate: {response}

Evaluate the response now."""

# Load model
llm = LLM(
    model="openai/gpt-oss-20b",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=8192  # Adjust based on your needs
)
tokenizer = AutoTokenizer.from_pretrained("/workspace/gpt-oss-20b")


class ChatRequest(BaseModel):
    user_speech: str
    intent: str
    emotion: str
    response: str
    max_tokens: int = 1024
    temperature: float = 0.1


@app.post("/generate")
async def generate(request: ChatRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_INPUT.format(user_speech=request.user_speech,
                                                      intent=request.intent,
                                                      emotion=request.emotion,
                                                      response=request.response)}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    outputs = llm.generate([formatted_prompt], sampling_params)
    return {"response": outputs[0].outputs[0].text.strip()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)






