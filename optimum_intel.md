# Embedding Documents using Optimized and Quantized Embedders

Embedding all documents using Quantized Embedders.

The embedders are based on optimized models, created by using [optimum-intel](https://github.com/huggingface/optimum-intel.git) and [IPEX](https://github.com/intel/intel-extension-for-pytorch).

Example text is based on [SBERT](https://www.sbert.net/docs/pretrained_cross-encoders.html).


```python
from langchain_community.embeddings import QuantizedBiEncoderEmbeddings

model_name = "Intel/bge-small-en-v1.5-rag-int8-static"
encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

model = QuantizedBiEncoderEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent this sentence for searching relevant passages: ",
)
```

    loading configuration file inc_config.json from cache at 
    INCConfig {
      "distillation": {},
      "neural_compressor_version": "2.4.1",
      "optimum_version": "1.16.2",
      "pruning": {},
      "quantization": {
        "dataset_num_samples": 50,
        "is_static": true
      },
      "save_onnx_model": false,
      "torch_version": "2.2.0",
      "transformers_version": "4.37.2"
    }
    
    Using `INCModel` to load a TorchScript model will be deprecated in v1.15.0, to load your model please use `IPEXModel` instead.
    

Lets ask a question, and compare to 2 documents. The first contains the answer to the question, and the second one does not. 

We can check better suits our query.


```python
question = "How many people live in Berlin?"
```


```python
documents = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin is well known for its museums.",
]
```


```python
doc_vecs = model.embed_documents(documents)
```

    Batches: 100%|██████████| 1/1 [00:00<00:00,  4.18it/s]
    


```python
query_vec = model.embed_query(question)
```


```python
import torch
```


```python
doc_vecs_torch = torch.tensor(doc_vecs)
```


```python
query_vec_torch = torch.tensor(query_vec)
```


```python
query_vec_torch @ doc_vecs_torch.T
```




    tensor([0.7980, 0.6529])



We can see that indeed the first one ranks higher.
