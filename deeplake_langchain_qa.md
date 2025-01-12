# Question Answering with LangChain, Deep Lake, & OpenAI

This notebook shows how to implement a question answering system with LangChain, [Deep Lake](https://activeloop.ai/) as a vector store and OpenAI embeddings. We will take the following steps to achieve this:

1. Load a Deep Lake text dataset
2. Initialize a [Deep Lake vector store with LangChain](https://docs.activeloop.ai/tutorials/vector-store/deep-lake-vector-store-in-langchain)
3. Add text to the vector store
4. Run queries on the database
5. Done!

You can also follow other tutorials such as question answering over any type of data (PDFs, json, csv, text): [chatting with any data](https://www.activeloop.ai/resources/data-chad-an-ai-app-with-lang-chain-deep-lake-to-chat-with-any-data/) stored in Deep Lake, [code understanding](https://www.activeloop.ai/resources/lang-chain-gpt-4-for-code-understanding-twitter-algorithm/), or [question answering over PDFs](https://www.activeloop.ai/resources/ultimate-guide-to-lang-chain-deep-lake-build-chat-gpt-to-answer-questions-on-your-financial-data/), or [recommending songs](https://www.activeloop.ai/resources/3-ways-to-build-a-recommendation-engine-for-songs-with-lang-chain/).

## Install requirements
Let's install the following packages.


```python
!pip install deeplake langchain openai tiktoken
```

## Authentication
Provide your OpenAI API key here:


```python
import getpass
import os

os.environ['OPENAI_API_KEY'] = getpass.getpass()
```

    ··········
    

## Load a Deep Lake text dataset
We will use a 20000 sample subset of the [cohere-wikipedia-22](https://app.activeloop.ai/davitbun/cohere-wikipedia-22) dataset for this example.


```python
import deeplake

ds = deeplake.load("hub://activeloop/cohere-wikipedia-22-sample")
ds.summary()
```

    \

    Opening dataset in read-only mode as you don't have write permissions.
    

    -

    This dataset can be visualized in Jupyter Notebook by ds.visualize() or at https://app.activeloop.ai/activeloop/cohere-wikipedia-22-sample
    
    

    |

    hub://activeloop/cohere-wikipedia-22-sample loaded successfully.
    
    Dataset(path='hub://activeloop/cohere-wikipedia-22-sample', read_only=True, tensors=['ids', 'metadata', 'text'])
    
      tensor    htype     shape      dtype  compression
     -------   -------   -------    -------  ------- 
       ids      text    (20000, 1)    str     None   
     metadata   json    (20000, 1)    str     None   
       text     text    (20000, 1)    str     None   
    

     

Let's take a look at a few samples:


```python
ds[:3].text.data()["value"]
```




    ['The 24-hour clock is a way of telling the time in which the day runs from midnight to midnight and is divided into 24 hours, numbered from 0 to 23. It does not use a.m. or p.m. This system is also referred to (only in the US and the English speaking parts of Canada) as military time or (only in the United Kingdom and now very rarely) as continental time. In some parts of the world, it is called railway time. Also, the international standard notation of time (ISO 8601) is based on this format.',
     'A time in the 24-hour clock is written in the form hours:minutes (for example, 01:23), or hours:minutes:seconds (01:23:45). Numbers under 10 have a zero in front (called a leading zero); e.g. 09:07. Under the 24-hour clock system, the day begins at midnight, 00:00, and the last minute of the day begins at 23:59 and ends at 24:00, which is identical to 00:00 of the following day. 12:00 can only be mid-day. Midnight is called 24:00 and is used to mean the end of the day and 00:00 is used to mean the beginning of the day. For example, you would say "Tuesday at 24:00" and "Wednesday at 00:00" to mean exactly the same time.',
     'However, the US military prefers not to say 24:00 - they do not like to have two names for the same thing, so they always say "23:59", which is one minute before midnight.']



## LangChain's Deep Lake vector store
Let's define a `dataset_path`, this is where your Deep Lake vector store will house the text embeddings.


```python
dataset_path = 'wikipedia-embeddings-deeplake'
```

We will setup OpenAI's `text-embedding-3-small` as our embedding function and initialize a Deep Lake vector store at `dataset_path`...


```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = DeepLake(dataset_path, embedding=embedding, overwrite=True)
```

    

... and populate it with samples, one batch at a time, using the `add_texts` method.


```python
from tqdm.auto import tqdm

batch_size = 100

nsamples = 10  # for testing. Replace with len(ds) to append everything
for i in tqdm(range(0, nsamples, batch_size)):
    # find end of batch
    i_end = min(nsamples, i + batch_size)

    batch = ds[i:i_end]
    id_batch = batch.ids.data()["value"]
    text_batch = batch.text.data()["value"]
    meta_batch = batch.metadata.data()["value"]

    db.add_texts(text_batch, metadatas=meta_batch, ids=id_batch)
```


      0%|          | 0/1 [00:00<?, ?it/s]


    
    creating embeddings:   0%|          | 0/1 [00:00<?, ?it/s][A
    creating embeddings: 100%|██████████| 1/1 [00:02<00:00,  2.11s/it]
    
    100%|██████████| 10/10 [00:00<00:00, 462.42it/s]

    Dataset(path='wikipedia-embeddings-deeplake', tensors=['text', 'metadata', 'embedding', 'id'])
    
      tensor      htype      shape      dtype  compression
      -------    -------    -------    -------  ------- 
       text       text      (10, 1)      str     None   
     metadata     json      (10, 1)      str     None   
     embedding  embedding  (10, 1536)  float32   None   
        id        text      (10, 1)      str     None   
    

    
    

## Run user queries on the database
The underlying Deep Lake dataset object is accessible through `db.vectorstore.dataset`, and the data structure can be summarized using `db.vectorstore.summary()`, which shows 4 tensors with 10 samples:


```python
db.vectorstore.summary()
```

    Dataset(path='wikipedia-embeddings-deeplake', tensors=['text', 'metadata', 'embedding', 'id'])
    
      tensor      htype      shape      dtype  compression
      -------    -------    -------    -------  ------- 
       text       text      (10, 1)      str     None   
     metadata     json      (10, 1)      str     None   
     embedding  embedding  (10, 1536)  float32   None   
        id        text      (10, 1)      str     None   
    

We will now setup QA on our vector store with GPT-3.5-Turbo as our LLM.


```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Re-load the vector store in case it's no longer initialized
# db = DeepLake(dataset_path = dataset_path, embedding_function=embedding)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-3.5-turbo'), chain_type="stuff", retriever=db.as_retriever())
```

Let's try running a prompt and check the output. Internally, this API performs an embedding search to find the most relevant data to feed into the LLM context.


```python
query = 'Why does the military not say 24:00?'
qa.run(query)
```




    'The military prefers not to say 24:00 because they do not like to have two names for the same thing. Instead, they always say "23:59", which is one minute before midnight.'



Et voila!
