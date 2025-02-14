# LLMRails

Let's load the LLMRails Embeddings class.

To use LLMRails embedding you need to pass api key by argument or set it in environment with `LLM_RAILS_API_KEY` key.
To gey API Key you need to sign up in https://console.llmrails.com/signup and then go to https://console.llmrails.com/api-keys and copy key from there after creating one key in platform.


```python
from langchain_community.embeddings import LLMRailsEmbeddings
```


```python
embeddings = LLMRailsEmbeddings(model="embedding-english-v1")  # or embedding-multi-v1
```


```python
text = "This is a test document."
```

To generate embeddings, you can either query an invidivual text, or you can query a list of texts.


```python
query_result = embeddings.embed_query(text)
query_result[:5]
```




    [-0.09996652603149414,
     0.015568195842206478,
     0.17670190334320068,
     0.16521021723747253,
     0.21193109452724457]




```python
doc_result = embeddings.embed_documents([text])
doc_result[0][:5]
```




    [-0.04242777079343796,
     0.016536075621843338,
     0.10052520781755447,
     0.18272875249385834,
     0.2079043835401535]


