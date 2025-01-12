# HuggingGPT
Implementation of [HuggingGPT](https://github.com/microsoft/JARVIS). HuggingGPT is a system to connect LLMs (ChatGPT) with ML community (Hugging Face).

+ 🔥 Paper: https://arxiv.org/abs/2303.17580
+ 🚀 Project: https://github.com/microsoft/JARVIS
+ 🤗 Space: https://huggingface.co/spaces/microsoft/HuggingGPT

## Set up tools

We set up the tools available from [Transformers Agent](https://huggingface.co/docs/transformers/transformers_agents#tools). It includes a library of tools supported by Transformers and some customized tools such as image generator, video generator, text downloader and other tools.


```python
from transformers import load_tool
```


```python
hf_tools = [
    load_tool(tool_name)
    for tool_name in [
        "document-question-answering",
        "image-captioning",
        "image-question-answering",
        "image-segmentation",
        "speech-to-text",
        "summarization",
        "text-classification",
        "text-question-answering",
        "translation",
        "huggingface-tools/text-to-image",
        "huggingface-tools/text-to-video",
        "text-to-speech",
        "huggingface-tools/text-download",
        "huggingface-tools/image-transformation",
    ]
]
```

## Setup model and HuggingGPT

We create an instance of HuggingGPT and use ChatGPT as the controller to rule the above tools.


```python
from langchain_experimental.autonomous_agents import HuggingGPT
from langchain_openai import OpenAI

# %env OPENAI_API_BASE=http://localhost:8000/v1
```


```python
llm = OpenAI(model_name="gpt-3.5-turbo")
agent = HuggingGPT(llm, hf_tools)
```

## Run an example

Given a text, show a related image and video.


```python
agent.run("please show me a video and an image of 'a boy is running'")
```
