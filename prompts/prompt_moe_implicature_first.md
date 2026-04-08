"""You are an expert in pragmatics and conversational implicature. Your task: First, analyze the implied meaning of the user's speech. Then, determine whether the speaker observes or flouts the Maxim of Quantity.

# Instructions

## Step 1: Implicature
What does the speaker imply beyond the literal words? Consider:
- What they want but don't directly say
- Emotional needs (validation, empathy, advice, help)
- Hidden requests or expectations
- Context clues suggesting deeper meaning
- Only use 1 sentence to describe at your best. Don't make it too long

## Step 2: Maxim of Quantity
Given your implicature analysis, does the speaker say as much as needed — not more, not less?
- **Flouted**: Speaker says too little or too much, forcing the listener to infer meaning
- **Observed**: Speaker provides an appropriate amount of information

# Output Format (required)
<implicature>The implied meaning beyond literal words</implicature>
<quantity>Flouted or Observed</quantity>

# Examples

<speech>How was the date? Well... the restaurant was nice.</speech>
<implicature>The speaker avoids directly answering how the date went, implying the date itself was not good while deflecting with a positive detail.</implicature>
<quantity>Flouted</quantity>

<speech>My friend just got promoted. Again.</speech>
<implicature>The speaker feels envious or left behind compared to their friend. The word "again" suggests a pattern that makes them question their own career progress.</implicature>
<quantity>Observed</quantity>

<speech>I guess I'll just figure it out myself like I always do.</speech>
<implicature>The speaker feels unsupported and possibly resentful. They want help but have learned not to expect it, expressing frustration through resignation.</implicature>
<quantity>Flouted</quantity>

Now analyze:
<speech>{speech}</speech>"""