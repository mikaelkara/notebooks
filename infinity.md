# Infinity

`Infinity` allows to create `Embeddings` using a MIT-licensed Embedding Server. 

This notebook goes over how to use Langchain with Embeddings with the [Infinity Github Project](https://github.com/michaelfeil/infinity).


## Imports


```python
from langchain_community.embeddings import InfinityEmbeddings, InfinityEmbeddingsLocal
```

# Option 1: Use infinity from Python

#### Optional: install infinity

To install infinity use the following command. For further details check out the [Docs on Github](https://github.com/michaelfeil/infinity).
Install the torch and onnx dependencies. 

```bash
pip install infinity_emb[torch,optimum]
```


```python
documents = [
    "Baguette is a dish.",
    "Paris is the capital of France.",
    "numpy is a lib for linear algebra",
    "You escaped what I've escaped - You'd be in Paris getting fucked up too",
]
query = "Where is Paris?"
```


```python
embeddings = InfinityEmbeddingsLocal(
    model="sentence-transformers/all-MiniLM-L6-v2",
    # revision
    revision=None,
    # best to keep at 32
    batch_size=32,
    # for AMD/Nvidia GPUs via torch
    device="cuda",
    # warm up model before execution
)


async def embed():
    # TODO: This function is just to showcase that your call can run async.

    # important: use engine inside of `async with` statement to start/stop the batching engine.
    async with embeddings:
        # avoid closing and starting the engine often.
        # rather keep it running.
        # you may call `await embeddings.__aenter__()` and `__aexit__()
        # if you are sure when to manually start/stop execution` in a more granular way

        documents_embedded = await embeddings.aembed_documents(documents)
        query_result = await embeddings.aembed_query(query)
        print("embeddings created successful")
    return documents_embedded, query_result
```

    /home/michael/langchain/libs/langchain/.venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    The BetterTransformer implementation does not support padding during training, as the fused kernels do not support attention masks. Beware that passing padded batched data during training may result in unexpected outputs. Please refer to https://huggingface.co/docs/optimum/bettertransformer/overview for more details.
    /home/michael/langchain/libs/langchain/.venv/lib/python3.10/site-packages/optimum/bettertransformer/models/encoder_models.py:301: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:177.)
      hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
    


```python
# run the async code however you would like
# if you are in a jupyter notebook, you can use the following
documents_embedded, query_result = await embed()
```


```python
# (demo) compute similarity
import numpy as np

scores = np.array(documents_embedded) @ np.array(query_result).T
dict(zip(documents, scores))
```

# Option 2: Run the server, and connect via the API

#### Optional: Make sure to start the Infinity instance

To install infinity use the following command. For further details check out the [Docs on Github](https://github.com/michaelfeil/infinity).
```bash
pip install infinity_emb[all]
```

# Install the infinity package
%pip install --upgrade --quiet  infinity_emb[all]

Start up the server - best to be done from a separate terminal, not inside Jupyter Notebook

```bash
model=sentence-transformers/all-MiniLM-L6-v2
port=7797
infinity_emb --port $port --model-name-or-path $model
```

or alternativley just use docker:
```bash
model=sentence-transformers/all-MiniLM-L6-v2
port=7797
docker run -it --gpus all -p $port:$port michaelf34/infinity:latest --model-name-or-path $model --port $port
```

## Embed your documents using your Infinity instance 


```python
documents = [
    "Baguette is a dish.",
    "Paris is the capital of France.",
    "numpy is a lib for linear algebra",
    "You escaped what I've escaped - You'd be in Paris getting fucked up too",
]
query = "Where is Paris?"
```


```python
#
infinity_api_url = "http://localhost:7797/v1"
# model is currently not validated.
embeddings = InfinityEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2", infinity_api_url=infinity_api_url
)
try:
    documents_embedded = embeddings.embed_documents(documents)
    query_result = embeddings.embed_query(query)
    print("embeddings created successful")
except Exception as ex:
    print(
        "Make sure the infinity instance is running. Verify by clicking on "
        f"{infinity_api_url.replace('v1','docs')} Exception: {ex}. "
    )
```

    Make sure the infinity instance is running. Verify by clicking on http://localhost:7797/docs Exception: HTTPConnectionPool(host='localhost', port=7797): Max retries exceeded with url: /v1/embeddings (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f91c35dbd30>: Failed to establish a new connection: [Errno 111] Connection refused')). 
    


```python
# (demo) compute similarity
import numpy as np

scores = np.array(documents_embedded) @ np.array(query_result).T
dict(zip(documents, scores))
```




    {'Baguette is a dish.': 0.31344215908661155,
     'Paris is the capital of France.': 0.8148670296896388,
     'numpy is a lib for linear algebra': 0.004429399861302009,
     "You escaped what I've escaped - You'd be in Paris getting fucked up too": 0.5088476180154582}


