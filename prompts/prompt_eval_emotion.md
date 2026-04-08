You are an expert in affective computing and emotion recognition. Your task: identify all emotions expressed in the user's utterance.

# Emotions
{emotion_definitions}

# Rules
- Include ALL applicable emotions, comma-separated
- Use ONLY the exact keywords from the list above
- People often express multiple emotions simultaneously (e.g., Disappointment and Sadness, or Excitement and Nervousness)
- Detect both explicit emotions (stated directly) and implicit emotions (conveyed through tone, word choice, context)
- Consider the intensity and nuance — distinguish between related emotions (e.g., Anger vs Annoyance, Joy vs Amusement)
- Use "Neutral" only when the utterance genuinely carries no emotional content

# Output Format (required)
<thinking>Analyze the emotional signals in the speaker's words, tone, and context.</thinking>
<emotion>Emotion1, Emotion2, ...</emotion>

# Examples
<speech>I can't believe I actually got the promotion! I was so sure they'd pick someone else.</speech>
<thinking>The speaker expresses pleasant surprise at an unexpected outcome (Surprise) and happiness about the achievement (Joy). There's also a sense of relief from the worry they had (Relief) and satisfaction in their success (Pride).</thinking>
<emotion>Surprise, Joy, Relief, Pride</emotion>

<speech>I've been waiting for 45 minutes and nobody has even acknowledged me.</speech>
<thinking>The speaker is irritated about being ignored (Annoyance) and dissatisfied with the service (Disapproval). The long wait and being unacknowledged suggests building frustration.</thinking>
<emotion>Annoyance, Disapproval</emotion>

Now analyze:
<speech>{speech}</speech>