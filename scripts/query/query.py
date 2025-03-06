from openai import OpenAI
import os

query = input("Type your query here: ")
modelid = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
modelurl = os.getenv("MODEL_URL", "http://localhost:9000/v1")

# Note: Ray Serve doesn't support all OpenAI client arguments and may ignore some.
client = OpenAI(
    # Replace the URL if deploying your app remotely
    # (e.g., on Anyscale or KubeRay).
    base_url=modelurl,
    api_key="NOT A REAL KEY",
)
chat_completion = client.chat.completions.create(
    model=modelid,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": query,
        },
    ],
    temperature=0.01,
    stream=True,
)

for chat in chat_completion:
    if chat.choices[0].delta.content is not None:
        print(chat.choices[0].delta.content, end="")
print("")
