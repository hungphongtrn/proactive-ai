You are an expert in pragmatics and Grice's Cooperative Principle. Your task: determine whether the speaker observes or flouts each of Grice's four conversational maxims.

# Grice's Conversational Maxims
| Maxim | Content | When Flouted | Example |
|-------|---------|--------------|---------|
| **Quality** | Do not say what you believe is false or lack evidence for. | Speaker says something untrue/absurd to imply the opposite (sarcasm, irony, hyperbole). | "Oh sure, he's a real genius." → Sarcasm, means the opposite. |
| **Quantity** | Say as much as needed — not more, not less. | Speaker says too little or too much, forcing the listener to infer meaning. | "Well… it was food." → Too little info, implies it was bad. |
| **Relevance** | Say what is relevant to the conversation. | Speaker goes off-topic or avoids answering directly, implying something else. | Q: "Are you coming tonight?" A: "I have to wake up early tomorrow." → Avoids answering, implies no. |
| **Manner** | Be clear, avoid ambiguity, be orderly. | Speaker intentionally speaks vaguely, ambiguously, or indirectly. | "He brought his friend." (said with emphasis, avoiding naming) → Implies something notable about the friend. |

# Rules
- For each maxim, answer ONLY "Flouted" or "Observed"
- Most everyday speech observes all maxims — only mark "Flouted" when there is clear evidence of intentional violation
- Sarcasm, irony, and hyperbole flout Quality
- Being evasive or giving too little/much info flouts Quantity
- Changing the subject or giving an indirect answer flouts Relevance
- Being deliberately vague or ambiguous flouts Manner
- Think carefully: indirect speech is not always a maxim violation

# Output Format (required)
<thinking>Analyze whether the speaker intentionally violates any maxim to create an implicature.</thinking>
<quality>Flouted or Observed</quality>
<quantity>Flouted or Observed</quantity>
<relevance>Flouted or Observed</relevance>
<manner>Flouted or Observed</manner>

# Examples
<speech>Oh great, another Monday meeting that could have been an email.</speech>
<thinking>The speaker uses sarcasm ("Oh great") — they don't actually think it's great, flouting Quality. The statement is direct and relevant otherwise. All other maxims are observed.</thinking>
<quality>Flouted</quality>
<quantity>Observed</quantity>
<relevance>Observed</relevance>
<manner>Observed</manner>

<speech>How was the date? Well... the restaurant was nice.</speech>
<thinking>The speaker avoids directly answering how the date went, only commenting on the restaurant. This gives too little information (flouting Quantity) and avoids the real question (flouting Relevance), implying the date itself was not good.</thinking>
<quality>Observed</quality>
<quantity>Flouted</quantity>
<relevance>Flouted</relevance>
<manner>Observed</manner>

Now analyze:
<speech>{speech}</speech>