---
sidebar_label: Cerebras
---
# ChatCerebras

This notebook provides a quick overview for getting started with Cerebras [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatCerebras features and configurations head to the [API reference](https://api.python.langchain.com/en/latest/chat_models/langchain_cerebras.chat_models.ChatCerebras.html).

At Cerebras, we've developed the world's largest and fastest AI processor, the Wafer-Scale Engine-3 (WSE-3). The Cerebras CS-3 system, powered by the WSE-3, represents a new class of AI supercomputer that sets the standard for generative AI training and inference with unparalleled performance and scalability.

With Cerebras as your inference provider, you can:
- Achieve unprecedented speed for AI inference workloads
- Build commercially with high throughput
- Effortlessly scale your AI workloads with our seamless clustering technology

Our CS-3 systems can be quickly and easily clustered to create the largest AI supercomputers in the world, making it simple to place and run the largest models. Leading corporations, research institutions, and governments are already using Cerebras solutions to develop proprietary models and train popular open-source models.

Want to experience the power of Cerebras? Check out our [website](https://cerebras.ai) for more resources and explore options for accessing our technology through the Cerebras Cloud or on-premise deployments!

For more information about Cerebras Cloud, visit [cloud.cerebras.ai](https://cloud.cerebras.ai/). Our API reference is available at [inference-docs.cerebras.ai](https://inference-docs.cerebras.ai/).

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/cerebras) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatCerebras](https://api.python.langchain.com/en/latest/chat_models/langchain_cerebras.chat_models.ChatCerebras.html) | [langchain-cerebras](https://api.python.langchain.com/en/latest/cerebras_api_reference.html) | ❌ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-cerebras?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-cerebras?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling/) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅  | ✅ | ❌ | 

## Setup

```bash
pip install langchain-cerebras
```

### Credentials

Get an API Key from [cloud.cerebras.ai](https://cloud.cerebras.ai/) and add it to your environment variables:
```
export CEREBRAS_API_KEY="your-api-key-here"
```


```python
import getpass
import os

if "CEREBRAS_API_KEY" not in os.environ:
    os.environ["CEREBRAS_API_KEY"] = getpass.getpass("Enter your Cerebras API key: ")
```

    Enter your Cerebras API key:  ········
    

If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:


```python
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"
```

### Installation

The LangChain Cerebras integration lives in the `langchain-cerebras` package:


```python
%pip install -qU langchain-cerebras
```

## Instantiation

Now we can instantiate our model object and generate chat completions:



```python
from langchain_cerebras import ChatCerebras

llm = ChatCerebras(
    model="llama3.1-70b",
    # other params...
)
```

## Invocation


```python
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg
```




    AIMessage(content='Je adore le programmation.', response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 35, 'total_tokens': 42}, 'model_name': 'llama3-8b-8192', 'system_fingerprint': 'fp_be27ec77ff', 'finish_reason': 'stop'}, id='run-e5d66faf-019c-4ac6-9265-71093b13202d-0', usage_metadata={'input_tokens': 35, 'output_tokens': 7, 'total_tokens': 42})



## Chaining

We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:


```python
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate

llm = ChatCerebras(
    model="llama3.1-70b",
    # other params...
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)
```




    AIMessage(content='Ich liebe Programmieren!\n\n(Literally: I love programming!)', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 30, 'total_tokens': 44}, 'model_name': 'llama3-8b-8192', 'system_fingerprint': 'fp_be27ec77ff', 'finish_reason': 'stop'}, id='run-e1d2ebb8-76d1-471b-9368-3b68d431f16a-0', usage_metadata={'input_tokens': 30, 'output_tokens': 14, 'total_tokens': 44})



## Streaming


```python
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate

llm = ChatCerebras(
    model="llama3.1-70b",
    # other params...
)

system = "You are an expert on animals who must answer questions in a manner that a 5 year old can understand."
human = "I want to learn more about this animal: {animal}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | llm

for chunk in chain.stream({"animal": "Lion"}):
    print(chunk.content, end="", flush=True)
```

    OH BOY! Let me tell you all about LIONS!
    
    Lions are the kings of the jungle! They're really big and have beautiful, fluffy manes around their necks. The mane is like a big, golden crown!
    
    Lions live in groups called prides. A pride is like a big family, and the lionesses (that's what we call the female lions) take care of the babies. The lionesses are like the mommies, and they teach the babies how to hunt and play.
    
    Lions are very good at hunting. They work together to catch their food, like zebras and antelopes. They're super fast and can run really, really fast!
    
    But lions are also very sleepy. They like to take long naps in the sun, and they can sleep for up to 20 hours a day! Can you imagine sleeping that much?
    
    Lions are also very loud. They roar really loudly to talk to each other. It's like they're saying, "ROAR! I'm the king of the jungle!"
    
    And guess what? Lions are very social. They like to play and cuddle with each other. They're like big, furry teddy bears!
    
    So, that's lions! Aren't they just the coolest?

## Async


```python
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate

llm = ChatCerebras(
    model="llama3.1-70b",
    # other params...
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Let's play a game of opposites. What's the opposite of {topic}? Just give me the answer with no extra input.",
        )
    ]
)
chain = prompt | llm
await chain.ainvoke({"topic": "fire"})
```




    AIMessage(content='Ice', response_metadata={'token_usage': {'completion_tokens': 2, 'prompt_tokens': 36, 'total_tokens': 38}, 'model_name': 'llama3-8b-8192', 'system_fingerprint': 'fp_be27ec77ff', 'finish_reason': 'stop'}, id='run-7434bdde-1bec-44cf-827b-8d978071dfe8-0', usage_metadata={'input_tokens': 36, 'output_tokens': 2, 'total_tokens': 38})



## Async Streaming


```python
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate

llm = ChatCerebras(
    model="llama3.1-70b",
    # other params...
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Write a long convoluted story about {subject}. I want {num_paragraphs} paragraphs.",
        )
    ]
)
chain = prompt | llm

async for chunk in chain.astream({"num_paragraphs": 3, "subject": "blackholes"}):
    print(chunk.content, end="", flush=True)
```

    In the distant reaches of the cosmos, there existed a peculiar phenomenon known as the "Eclipse of Eternity," a swirling vortex of darkness that had been shrouded in mystery for eons. It was said that this blackhole, born from the cataclysmic collision of two ancient stars, had been slowly devouring the fabric of space-time itself, warping the very essence of reality. As the celestial bodies of the galaxy danced around it, they began to notice a strange, almost imperceptible distortion in the fabric of space, as if the blackhole's gravitational pull was exerting an influence on the very course of events itself.
    
    As the centuries passed, astronomers from across the galaxy became increasingly fascinated by the Eclipse of Eternity, pouring over ancient texts and scouring the cosmos for any hint of its secrets. One such scholar, a brilliant and reclusive astrophysicist named Dr. Elara Vex, became obsessed with unraveling the mysteries of the blackhole. She spent years pouring over ancient texts, deciphering cryptic messages and hidden codes that hinted at the existence of a long-lost civilization that had once thrived in the heart of the blackhole itself. According to legend, this ancient civilization had possessed knowledge of the cosmos that was beyond human comprehension, and had used their mastery of the universe to create the Eclipse of Eternity as a gateway to other dimensions.
    
    As Dr. Vex delved deeper into her research, she began to experience strange and vivid dreams, visions that seemed to transport her to the very heart of the blackhole itself. In these dreams, she saw ancient beings, their faces twisted in agony as they were consumed by the void. She saw stars and galaxies, their light warped and distorted by the blackhole's gravitational pull. And she saw the Eclipse of Eternity itself, its swirling vortex of darkness pulsing with an otherworldly energy that seemed to be calling to her. As the dreams grew more vivid and more frequent, Dr. Vex became convinced that she was being drawn into the heart of the blackhole, and that the secrets of the universe lay waiting for her on the other side.

## API reference

For detailed documentation of all ChatCerebras features and configurations head to the API reference: https://api.python.langchain.com/en/latest/chat_models/langchain_cerebras.chat_models.ChatCerebras.html
