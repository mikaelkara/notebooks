---
sidebar_label: Llama 2 Chat
---
# Llama2Chat

This notebook shows how to augment Llama-2 `LLM`s with the `Llama2Chat` wrapper to support the [Llama-2 chat prompt format](https://huggingface.co/blog/llama2#how-to-prompt-llama-2). Several `LLM` implementations in LangChain can be used as interface to Llama-2 chat models. These include [ChatHuggingFace](/docs/integrations/chat/huggingface), [LlamaCpp](/docs/tutorials/local_rag), [GPT4All](/docs/integrations/llms/gpt4all), ..., to mention a few examples. 

`Llama2Chat` is a generic wrapper that implements `BaseChatModel` and can therefore be used in applications as [chat model](/docs/how_to#chat-models). `Llama2Chat` converts a list of Messages into the [required chat prompt format](https://huggingface.co/blog/llama2#how-to-prompt-llama-2) and forwards the formatted prompt as `str` to the wrapped `LLM`.


```python
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
```

For the chat application examples below, we'll use the following chat `prompt_template`:


```python
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)
```

## Chat with Llama-2 via `HuggingFaceTextGenInference` LLM

A HuggingFaceTextGenInference LLM encapsulates access to a [text-generation-inference](https://github.com/huggingface/text-generation-inference) server. In the following example, the inference server serves a [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) model. It can be started locally with:

```bash
docker run \
  --rm \
  --gpus all \
  --ipc=host \
  -p 8080:80 \
  -v ~/.cache/huggingface/hub:/data \
  -e HF_API_TOKEN=${HF_API_TOKEN} \
  ghcr.io/huggingface/text-generation-inference:0.9 \
  --hostname 0.0.0.0 \
  --model-id meta-llama/Llama-2-13b-chat-hf \
  --quantize bitsandbytes \
  --num-shard 4
```

This works on a machine with 4 x RTX 3080ti cards, for example. Adjust the `--num_shard` value to the number of GPUs available. The `HF_API_TOKEN` environment variable holds the Hugging Face API token.


```python
# !pip3 install text-generation
```

Create a `HuggingFaceTextGenInference` instance that connects to the local inference server and wrap it into `Llama2Chat`.


```python
from langchain_community.llms import HuggingFaceTextGenInference

llm = HuggingFaceTextGenInference(
    inference_server_url="http://127.0.0.1:8080/",
    max_new_tokens=512,
    top_k=50,
    temperature=0.1,
    repetition_penalty=1.03,
)

model = Llama2Chat(llm=llm)
```

Then you are ready to use the chat `model` together with `prompt_template` and conversation `memory` in an `LLMChain`.


```python
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)
```


```python
print(
    chain.run(
        text="What can I see in Vienna? Propose a few locations. Names only, no details."
    )
)
```

     Sure, I'd be happy to help! Here are a few popular locations to consider visiting in Vienna:
    
    1. Schönbrunn Palace
    2. St. Stephen's Cathedral
    3. Hofburg Palace
    4. Belvedere Palace
    5. Prater Park
    6. Vienna State Opera
    7. Albertina Museum
    8. Museum of Natural History
    9. Kunsthistorisches Museum
    10. Ringstrasse
    


```python
print(chain.run(text="Tell me more about #2."))
```

     Certainly! St. Stephen's Cathedral (Stephansdom) is one of the most recognizable landmarks in Vienna and a must-see attraction for visitors. This stunning Gothic cathedral is located in the heart of the city and is known for its intricate stone carvings, colorful stained glass windows, and impressive dome.
    
    The cathedral was built in the 12th century and has been the site of many important events throughout history, including the coronation of Holy Roman emperors and the funeral of Mozart. Today, it is still an active place of worship and offers guided tours, concerts, and special events. Visitors can climb up the south tower for panoramic views of the city or attend a service to experience the beautiful music and chanting.
    

## Chat with Llama-2 via `LlamaCPP` LLM

For using a Llama-2 chat model with a [LlamaCPP](/docs/integrations/llms/llamacpp) `LMM`, install the `llama-cpp-python` library using [these installation instructions](/docs/integrations/llms/llamacpp#installation). The following example uses a quantized [llama-2-7b-chat.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf) model stored locally at `~/Models/llama-2-7b-chat.Q4_0.gguf`. 

After creating a `LlamaCpp` instance, the `llm` is again wrapped into `Llama2Chat`


```python
from os.path import expanduser

from langchain_community.llms import LlamaCpp

model_path = expanduser("~/Models/llama-2-7b-chat.Q4_0.gguf")

llm = LlamaCpp(
    model_path=model_path,
    streaming=False,
)
model = Llama2Chat(llm=llm)
```

and used in the same way as in the previous example.


```python
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)
```


```python
print(
    chain.run(
        text="What can I see in Vienna? Propose a few locations. Names only, no details."
    )
)
```

      Of course! Vienna is a beautiful city with a rich history and culture. Here are some of the top tourist attractions you might want to consider visiting:
    1. Schönbrunn Palace
    2. St. Stephen's Cathedral
    3. Hofburg Palace
    4. Belvedere Palace
    5. Prater Park
    6. MuseumsQuartier
    7. Ringstrasse
    8. Vienna State Opera
    9. Kunsthistorisches Museum
    10. Imperial Palace
    
    These are just a few of the many amazing places to see in Vienna. Each one has its own unique history and charm, so I hope you enjoy exploring this beautiful city!
    

    
    llama_print_timings:        load time =     250.46 ms
    llama_print_timings:      sample time =      56.40 ms /   144 runs   (    0.39 ms per token,  2553.37 tokens per second)
    llama_print_timings: prompt eval time =    1444.25 ms /    47 tokens (   30.73 ms per token,    32.54 tokens per second)
    llama_print_timings:        eval time =    8832.02 ms /   143 runs   (   61.76 ms per token,    16.19 tokens per second)
    llama_print_timings:       total time =   10645.94 ms
    


```python
print(chain.run(text="Tell me more about #2."))
```

    Llama.generate: prefix-match hit
    

      Of course! St. Stephen's Cathedral (also known as Stephansdom) is a stunning Gothic-style cathedral located in the heart of Vienna, Austria. It is one of the most recognizable landmarks in the city and is considered a symbol of Vienna.
    Here are some interesting facts about St. Stephen's Cathedral:
    1. History: The construction of St. Stephen's Cathedral began in the 12th century on the site of a former Romanesque church, and it took over 600 years to complete. The cathedral has been renovated and expanded several times throughout its history, with the most significant renovation taking place in the 19th century.
    2. Architecture: St. Stephen's Cathedral is built in the Gothic style, characterized by its tall spires, pointed arches, and intricate stone carvings. The cathedral features a mix of Romanesque, Gothic, and Baroque elements, making it a unique blend of styles.
    3. Design: The cathedral's design is based on the plan of a cross with a long nave and two shorter arms extending from it. The main altar is
    

    
    llama_print_timings:        load time =     250.46 ms
    llama_print_timings:      sample time =     100.60 ms /   256 runs   (    0.39 ms per token,  2544.73 tokens per second)
    llama_print_timings: prompt eval time =    5128.71 ms /   160 tokens (   32.05 ms per token,    31.20 tokens per second)
    llama_print_timings:        eval time =   16193.02 ms /   255 runs   (   63.50 ms per token,    15.75 tokens per second)
    llama_print_timings:       total time =   21988.57 ms
    
