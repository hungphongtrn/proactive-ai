import requests


def query_hosted_model(user_speech: str, intent: str, emotion: str, response: str, server_url):
    response = requests.post(f"{server_url}/generate", json={"user_speech": user_speech,
                                                             "intent": intent,
                                                             "emotion": emotion,
                                                             "response": response})
    print(response.json())
    return response.json()["response"]  # Extract the text
