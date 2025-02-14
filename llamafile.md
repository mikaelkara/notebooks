# Llamafile

[Llamafile](https://github.com/Mozilla-Ocho/llamafile) lets you distribute and run LLMs with a single file.

Llamafile does this by combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one framework that collapses all the complexity of LLMs down to a single-file executable (called a "llamafile") that runs locally on most computers, with no installation.

## Setup

1. Download a llamafile for the model you'd like to use. You can find many models in llamafile format on [HuggingFace](https://huggingface.co/models?other=llamafile). In this guide, we will download a small one, `TinyLlama-1.1B-Chat-v1.0.Q5_K_M`. Note: if you don't have `wget`, you can just download the model via this [link](https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile?download=true).

```bash
wget https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
```

2. Make the llamafile executable. First, if you haven't done so already, open a terminal. **If you're using MacOS, Linux, or BSD,** you'll need to grant permission for your computer to execute this new file using `chmod` (see below). **If you're on Windows,** rename the file by adding ".exe" to the end (model file should be named `TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile.exe`).


```bash
chmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile  # run if you're on MacOS, Linux, or BSD
```

3. Run the llamafile in "server mode":

```bash
./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser
```

Now you can make calls to the llamafile's REST API. By default, the llamafile server listens at http://localhost:8080. You can find full server documentation [here](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints). You can interact with the llamafile directly via the REST API, but here we'll show how to interact with it using LangChain.


## Usage


```python
from langchain_community.llms.llamafile import Llamafile

llm = Llamafile()

llm.invoke("Tell me a joke")
```




    '? \nI\'ve got a thing for pink, but you know that.\n"Can we not talk about work anymore?" - What did she say?\nI don\'t want to be a burden on you.\nIt\'s hard to keep a good thing going.\nYou can\'t tell me what I want, I have a life too!'



To stream tokens, use the `.stream(...)` method:


```python
query = "Tell me a joke"

for chunks in llm.stream(query):
    print(chunks, end="")

print()
```

    .
    - She said, "I’m tired of my life. What should I do?"
    - The man replied, "I hear you. But don’t worry. Life is just like a joke. It has its funny parts too."
    - The woman looked at him, amazed and happy to hear his wise words. - "Thank you for your wisdom," she said, smiling. - He replied, "Any time. But it doesn't come easy. You have to laugh and keep moving forward in life."
    - She nodded, thanking him again. - The man smiled wryly. "Life can be tough. Sometimes it seems like you’re never going to get out of your situation."
    - He said, "I know that. But the key is not giving up. Life has many ups and downs, but in the end, it will turn out okay."
    - The woman's eyes softened. "Thank you for your advice. It's so important to keep moving forward in life," she said. - He nodded once again. "You’re welcome. I hope your journey is filled with laughter and joy."
    - They both smiled and left the bar, ready to embark on their respective adventures.
    

To learn more about the LangChain Expressive Language and the available methods on an LLM, see the [LCEL Interface](/docs/concepts/runnables)
