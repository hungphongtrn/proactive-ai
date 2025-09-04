import pandas as pd
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import re


load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

def load_dataset_from_source(path=None):
    """Load dataset from HuggingFace or CSV file"""
    if path:
        # Load from CSV
        return pd.read_csv(path)
    else:
        # Load from HuggingFace
        dataset = load_dataset("richelle05/maxims_50_golden_samples")
        return pd.DataFrame(dataset['train'])


def gen_normal_response(df):
    prompt_template = "[response like you are a normal guy talking to me]\n{user1}"
    responses = []
    for idx, user_text in enumerate(df['user1']):
        print(f"Processing response {idx + 1}/{len(df)}: {user_text}")
        try:
            prompt = prompt_template.format(user1=user_text)
            message = HumanMessage(content=prompt)
            response = llm.invoke([message])
            responses.append(response.content)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            responses.append(f"Error: {str(e)}")

    df['response'] = responses
    return df


def gen_good_bad_responses(df):
    prompt_template = """You are a behaviour scientist who understand deeply about what people often say in bad way and good way. 
<task>
I will give you an user's speech with his intent and emotion when saying that. I want you to give me a good and a bad response you can give to him as a normal person that he is talking to. You can see the speech has an under meaning, and your response is an example for intention understanding.
</task>

<response_definition>
- A good response is a response fit to the emotion and intent, it shows you understand what he is trying to say and express.

- A bad response is when you does not understand the context of intent and/or emotion, what is he indicating when saying so, but only the speech instead. Bad response does not mean impolite or rude response or you try to generate an opposite intent/emotion. It is just you look at the speech only, not the under meaning. Don't make it looks unnatural just to highlight the opposite intent and/or emotion, or the opposite to the good response. It's just not understand the context of intent and/or emotion.
</response_definition>

<output_format>
The responses should be separated by tags and you should not return anything else, like: "<good>...</good> 
 <bad>...</bad>". I will use regex to extract the responses, so make sure you follow the format.
</output_format>

speech: {user1}
intent: {intent}
emotion: {emotion}
"""

    good_responses = []
    bad_responses = []

    for idx, row in df.iterrows():
        print(f"Processing good/bad responses {idx + 1}/{len(df)}: {row['user1']}")
        try:
            formatted_prompt = prompt_template.format(
                user1=row['user1'],
                intent=row['intent'],
                emotion=row['emotion']
            )
            message = HumanMessage(content=formatted_prompt)
            response = llm.invoke([message])

            # Extract good and bad responses using regex
            good_match = re.search(r'<good>(.*?)</good>', response.content, re.DOTALL)
            bad_match = re.search(r'<bad>(.*?)</bad>', response.content, re.DOTALL)

            good_res = good_match.group(1).strip() if good_match else "No good response found"
            bad_res = bad_match.group(1).strip() if bad_match else "No bad response found"

            print(f"Good: {good_res}\nBad: {bad_res}\n")

            good_responses.append(good_res)
            bad_responses.append(bad_res)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            good_responses.append(f"Error: {str(e)}")
            bad_responses.append(f"Error: {str(e)}")

    df['good_response'] = good_responses
    df['bad_response'] = bad_responses
    return df


# Main execution
if __name__ == "__main__":
    # Load dataset (use path=None for HuggingFace, or path="file.csv" for CSV)
    df = load_dataset_from_source(path='maxims_with_responses.csv')

    # Add response column
    df = gen_normal_response(df)

    # Add good_response and bad_response columns
    df = gen_good_bad_responses(df)

    # Save new dataframe
    df.to_csv('maxims_with_all_responses.csv', index=False)

    print(f"Dataset processed successfully! Saved {len(df)} rows with all responses.")
    print(f"Columns: {list(df.columns)}")