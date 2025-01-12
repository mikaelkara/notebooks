## This demo app shows how to query Llama 3 using the Gradio UI.

Since we are using OctoAI in this example, you'll need to obtain an OctoAI token:

- You will need to first sign into [OctoAI](https://octoai.cloud/) with your Github or Google account
- Then create a free API token [here](https://octo.ai/docs/getting-started/how-to-create-an-octoai-access-token) that you can use for a while (a month or $10 in OctoAI credits, whichever one runs out first)

**Note** After the free trial ends, you will need to enter billing info to continue to use Llama 3 hosted on OctoAI.

To run this example:
- Run the notebook
- Set up your OCTOAI API token and enter it when prompted
- Enter your question and click Submit

In the notebook or a browser with URL http://127.0.0.1:7860 you should see a UI with your answer.

Let's start by installing the necessary packages:
- openai for us to use its APIs to talk to the OctoAI endpoint
- gradio is used for the UI elements

And setting up the OctoAI token.


```python
!pip install openai gradio
```


```python
from getpass import getpass
import os

OCTOAI_API_TOKEN = getpass()
os.environ["OCTOAI_API_TOKEN"] = OCTOAI_API_TOKEN
```


```python
import gradio as gr
import openai

# Init OctoAI client
client = openai.OpenAI(
    base_url="https://text.octoai.run/v1",
    api_key=os.environ["OCTOAI_API_TOKEN"]
)

def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model = 'meta-llama-3-70b-instruct',
        messages = history_openai_format,
        temperature = 0.0,
        stream = True
     )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              partial_message = partial_message + chunk.choices[0].delta.content
              yield partial_message

gr.ChatInterface(predict).launch()
```
