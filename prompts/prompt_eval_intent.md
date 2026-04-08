You are an expert in conversational AI and user intent recognition. Your task: identify all intents in the user's utterance.

# Intents
{intent_definitions}

# Rules
- Include ALL applicable intents, comma-separated
- Use ONLY the exact keywords from the list above
- A single utterance can express multiple intents (e.g., venting frustration while also seeking empathy)
- Focus on what the user is trying to accomplish or communicate, not just the surface words
- Think step by step about the speaker's goals

# Output Format (required)
<thinking>Analyze the speaker's underlying goals and communicative purposes.</thinking>
<intent>Intent1, Intent2, ...</intent>

# Examples
<speech>Hey, I just wanted to let you know I had a really rough day. My manager criticized my work in front of everyone.</speech>
<thinking>The speaker is sharing a personal experience (Self_Disclosure) and venting about a negative event (Vent_Frustration). They may also be seeking emotional support (Empathy_Seek).</thinking>
<intent>Self_Disclosure, Vent_Frustration, Empathy_Seek</intent>

<speech>Can I reschedule my appointment to next Friday? The 3pm slot would work best.</speech>
<thinking>The speaker wants to change an existing appointment (Reschedule) and is choosing a specific time option (Choose_Option).</thinking>
<intent>Reschedule, Choose_Option</intent>

Now analyze:
<speech>{speech}</speech>