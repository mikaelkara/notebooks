#  Activeloop Deep Lake

>[Activeloop Deep Lake](https://docs.activeloop.ai/) as a Multi-Modal Vector Store that stores embeddings and their metadata including text, Jsons, images, audio, video, and more. It saves the data locally, in your cloud, or on Activeloop storage. It performs hybrid search including embeddings and their attributes.

This notebook showcases basic functionality related to `Activeloop Deep Lake`. While `Deep Lake` can store embeddings, it is capable of storing any type of data. It is a serverless data lake with version control, query engine and streaming dataloaders to deep learning frameworks.  

For more information, please see the Deep Lake [documentation](https://docs.activeloop.ai) or [api reference](https://docs.deeplake.ai)

## Setting up


```python
%pip install --upgrade --quiet  langchain-openai langchain-community 'deeplake[enterprise]' tiktoken
```

## Example provided by Activeloop

[Integration with LangChain](https://docs.activeloop.ai/tutorials/vector-store/deep-lake-vector-store-in-langchain).


## Deep Lake locally


```python
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
```


```python
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
activeloop_token = getpass.getpass("activeloop token:")
embeddings = OpenAIEmbeddings()
```


```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
```

### Create a local dataset

Create a dataset locally at `./deeplake/`, then run similarity search. The Deeplake+LangChain integration uses Deep Lake datasets under the hood, so `dataset` and `vector store` are used interchangeably. To create a dataset in your own cloud, or in the Deep Lake storage, [adjust the path accordingly](https://docs.activeloop.ai/storage-and-credentials/storage-options).


```python
db = DeepLake(dataset_path="./my_deeplake/", embedding=embeddings, overwrite=True)
db.add_documents(docs)
# or shorter
# db = DeepLake.from_documents(docs, dataset_path="./my_deeplake/", embedding=embeddings, overwrite=True)
```

### Query dataset


```python
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
```

    

    Dataset(path='./my_deeplake/', tensors=['embedding', 'id', 'metadata', 'text'])
    
      tensor      htype      shape      dtype  compression
      -------    -------    -------    -------  ------- 
     embedding  embedding  (42, 1536)  float32   None   
        id        text      (42, 1)      str     None   
     metadata     json      (42, 1)      str     None   
       text       text      (42, 1)      str     None   
    

    

To disable dataset summary printings all the time, you can specify verbose=False during VectorStore initialization.


```python
print(docs[0].page_content)
```

    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 
    
    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 
    
    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 
    
    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
    

Later, you can reload the dataset without recomputing embeddings


```python
db = DeepLake(dataset_path="./my_deeplake/", embedding=embeddings, read_only=True)
docs = db.similarity_search(query)
```

    Deep Lake Dataset in ./my_deeplake/ already exists, loading from the storage
    

Deep Lake, for now, is single writer and multiple reader. Setting `read_only=True` helps to avoid acquiring the writer lock.

### Retrieval Question/Answering


```python
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIChat

qa = RetrievalQA.from_chain_type(
    llm=OpenAIChat(model="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=db.as_retriever(),
)
```

    /home/ubuntu/langchain_activeloop/langchain/libs/langchain/langchain/llms/openai.py:786: UserWarning: You are trying to use a chat model. This way of initializing it is no longer supported. Instead, please use: `from langchain_openai import ChatOpenAI`
      warnings.warn(
    


```python
query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)
```




    'The president said that Ketanji Brown Jackson is a former top litigator in private practice and a former federal public defender. She comes from a family of public school educators and police officers. She is a consensus builder and has received a broad range of support since being nominated.'



### Attribute based filtering in metadata

Let's create another vector store containing metadata with the year the documents were created.


```python
import random

for d in docs:
    d.metadata["year"] = random.randint(2012, 2014)

db = DeepLake.from_documents(
    docs, embeddings, dataset_path="./my_deeplake/", overwrite=True
)
```

    

    Dataset(path='./my_deeplake/', tensors=['embedding', 'id', 'metadata', 'text'])
    
      tensor      htype      shape     dtype  compression
      -------    -------    -------   -------  ------- 
     embedding  embedding  (4, 1536)  float32   None   
        id        text      (4, 1)      str     None   
     metadata     json      (4, 1)      str     None   
       text       text      (4, 1)      str     None   
    

    


```python
db.similarity_search(
    "What did the president say about Ketanji Brown Jackson",
    filter={"metadata": {"year": 2013}},
)
```

    100%|██████████| 4/4 [00:00<00:00, 2936.16it/s]
    




    [Document(page_content='Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. \n\nTonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. \n\nOne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. \n\nAnd I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2013}),
     Document(page_content='A former top litigator in private practice. A former federal public defender. And from a family of public school educators and police officers. A consensus builder. Since she’s been nominated, she’s received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democrats and Republicans. \n\nAnd if we are to advance liberty and justice, we need to secure the Border and fix the immigration system. \n\nWe can do both. At our border, we’ve installed new technology like cutting-edge scanners to better detect drug smuggling.  \n\nWe’ve set up joint patrols with Mexico and Guatemala to catch more human traffickers.  \n\nWe’re putting in place dedicated immigration judges so families fleeing persecution and violence can have their cases heard faster. \n\nWe’re securing commitments and supporting partners in South and Central America to host more refugees and secure their own borders.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2013}),
     Document(page_content='Tonight, I’m announcing a crackdown on these companies overcharging American businesses and consumers. \n\nAnd as Wall Street firms take over more nursing homes, quality in those homes has gone down and costs have gone up.  \n\nThat ends on my watch. \n\nMedicare is going to set higher standards for nursing homes and make sure your loved ones get the care they deserve and expect. \n\nWe’ll also cut costs and keep the economy going strong by giving workers a fair shot, provide more training and apprenticeships, hire them based on their skills not degrees. \n\nLet’s pass the Paycheck Fairness Act and paid leave.  \n\nRaise the minimum wage to $15 an hour and extend the Child Tax Credit, so no one has to raise a family in poverty. \n\nLet’s increase Pell Grants and increase our historic support of HBCUs, and invest in what Jill—our First Lady who teaches full-time—calls America’s best-kept secret: community colleges.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2013})]



### Choosing distance function
Distance function `L2` for Euclidean, `L1` for Nuclear, `Max` l-infinity distance, `cos` for cosine similarity, `dot` for dot product 


```python
db.similarity_search(
    "What did the president say about Ketanji Brown Jackson?", distance_metric="cos"
)
```




    [Document(page_content='Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. \n\nTonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. \n\nOne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. \n\nAnd I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2013}),
     Document(page_content='A former top litigator in private practice. A former federal public defender. And from a family of public school educators and police officers. A consensus builder. Since she’s been nominated, she’s received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democrats and Republicans. \n\nAnd if we are to advance liberty and justice, we need to secure the Border and fix the immigration system. \n\nWe can do both. At our border, we’ve installed new technology like cutting-edge scanners to better detect drug smuggling.  \n\nWe’ve set up joint patrols with Mexico and Guatemala to catch more human traffickers.  \n\nWe’re putting in place dedicated immigration judges so families fleeing persecution and violence can have their cases heard faster. \n\nWe’re securing commitments and supporting partners in South and Central America to host more refugees and secure their own borders.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2013}),
     Document(page_content='Tonight, I’m announcing a crackdown on these companies overcharging American businesses and consumers. \n\nAnd as Wall Street firms take over more nursing homes, quality in those homes has gone down and costs have gone up.  \n\nThat ends on my watch. \n\nMedicare is going to set higher standards for nursing homes and make sure your loved ones get the care they deserve and expect. \n\nWe’ll also cut costs and keep the economy going strong by giving workers a fair shot, provide more training and apprenticeships, hire them based on their skills not degrees. \n\nLet’s pass the Paycheck Fairness Act and paid leave.  \n\nRaise the minimum wage to $15 an hour and extend the Child Tax Credit, so no one has to raise a family in poverty. \n\nLet’s increase Pell Grants and increase our historic support of HBCUs, and invest in what Jill—our First Lady who teaches full-time—calls America’s best-kept secret: community colleges.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2013}),
     Document(page_content='And for our LGBTQ+ Americans, let’s finally get the bipartisan Equality Act to my desk. The onslaught of state laws targeting transgender Americans and their families is wrong. \n\nAs I said last year, especially to our younger transgender Americans, I will always have your back as your President, so you can be yourself and reach your God-given potential. \n\nWhile it often appears that we never agree, that isn’t true. I signed 80 bipartisan bills into law last year. From preventing government shutdowns to protecting Asian-Americans from still-too-common hate crimes to reforming military justice. \n\nAnd soon, we’ll strengthen the Violence Against Women Act that I first wrote three decades ago. It is important for us to show the nation that we can come together and do big things. \n\nSo tonight I’m offering a Unity Agenda for the Nation. Four big things we can do together.  \n\nFirst, beat the opioid epidemic.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2012})]



### Maximal Marginal relevance
Using maximal marginal relevance


```python
db.max_marginal_relevance_search(
    "What did the president say about Ketanji Brown Jackson?"
)
```




    [Document(page_content='Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. \n\nTonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. \n\nOne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. \n\nAnd I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2013}),
     Document(page_content='Tonight, I’m announcing a crackdown on these companies overcharging American businesses and consumers. \n\nAnd as Wall Street firms take over more nursing homes, quality in those homes has gone down and costs have gone up.  \n\nThat ends on my watch. \n\nMedicare is going to set higher standards for nursing homes and make sure your loved ones get the care they deserve and expect. \n\nWe’ll also cut costs and keep the economy going strong by giving workers a fair shot, provide more training and apprenticeships, hire them based on their skills not degrees. \n\nLet’s pass the Paycheck Fairness Act and paid leave.  \n\nRaise the minimum wage to $15 an hour and extend the Child Tax Credit, so no one has to raise a family in poverty. \n\nLet’s increase Pell Grants and increase our historic support of HBCUs, and invest in what Jill—our First Lady who teaches full-time—calls America’s best-kept secret: community colleges.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2013}),
     Document(page_content='A former top litigator in private practice. A former federal public defender. And from a family of public school educators and police officers. A consensus builder. Since she’s been nominated, she’s received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democrats and Republicans. \n\nAnd if we are to advance liberty and justice, we need to secure the Border and fix the immigration system. \n\nWe can do both. At our border, we’ve installed new technology like cutting-edge scanners to better detect drug smuggling.  \n\nWe’ve set up joint patrols with Mexico and Guatemala to catch more human traffickers.  \n\nWe’re putting in place dedicated immigration judges so families fleeing persecution and violence can have their cases heard faster. \n\nWe’re securing commitments and supporting partners in South and Central America to host more refugees and secure their own borders.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2013}),
     Document(page_content='And for our LGBTQ+ Americans, let’s finally get the bipartisan Equality Act to my desk. The onslaught of state laws targeting transgender Americans and their families is wrong. \n\nAs I said last year, especially to our younger transgender Americans, I will always have your back as your President, so you can be yourself and reach your God-given potential. \n\nWhile it often appears that we never agree, that isn’t true. I signed 80 bipartisan bills into law last year. From preventing government shutdowns to protecting Asian-Americans from still-too-common hate crimes to reforming military justice. \n\nAnd soon, we’ll strengthen the Violence Against Women Act that I first wrote three decades ago. It is important for us to show the nation that we can come together and do big things. \n\nSo tonight I’m offering a Unity Agenda for the Nation. Four big things we can do together.  \n\nFirst, beat the opioid epidemic.', metadata={'source': '../../how_to/state_of_the_union.txt', 'year': 2012})]



### Delete dataset


```python
db.delete_dataset()
```

    

and if delete fails you can also force delete


```python
DeepLake.force_delete_by_path("./my_deeplake")
```

    

## Deep Lake datasets on cloud (Activeloop, AWS, GCS, etc.) or in memory
By default, Deep Lake datasets are stored locally. To store them in memory, in the Deep Lake Managed DB, or in any object storage, you can provide the [corresponding path and credentials when creating the vector store](https://docs.activeloop.ai/storage-and-credentials/storage-options). Some paths require registration with Activeloop and creation of an API token that can be [retrieved here](https://app.activeloop.ai/)


```python
os.environ["ACTIVELOOP_TOKEN"] = activeloop_token
```


```python
# Embed and store the texts
username = "<USERNAME_OR_ORG>"  # your username on app.activeloop.ai
dataset_path = f"hub://{username}/langchain_testing_python"  # could be also ./local/path (much faster locally), s3://bucket/path/to/dataset, gcs://path/to/dataset, etc.

docs = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
db = DeepLake(dataset_path=dataset_path, embedding=embeddings, overwrite=True)
ids = db.add_documents(docs)
```

    Your Deep Lake dataset has been successfully created!
    

    

    Dataset(path='hub://adilkhan/langchain_testing_python', tensors=['embedding', 'id', 'metadata', 'text'])
    
      tensor      htype      shape      dtype  compression
      -------    -------    -------    -------  ------- 
     embedding  embedding  (42, 1536)  float32   None   
        id        text      (42, 1)      str     None   
     metadata     json      (42, 1)      str     None   
       text       text      (42, 1)      str     None   
    

    


```python
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 
    
    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 
    
    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 
    
    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
    

#### `tensor_db` execution option 

In order to utilize Deep Lake's Managed Tensor Database, it is necessary to specify the runtime parameter as `{'tensor_db': True}` during the creation of the vector store. This configuration enables the execution of queries on the Managed Tensor Database, rather than on the client side. It should be noted that this functionality is not applicable to datasets stored locally or in-memory. In the event that a vector store has already been created outside of the Managed Tensor Database, it is possible to transfer it to the Managed Tensor Database by following the prescribed steps.


```python
# Embed and store the texts
username = "<USERNAME_OR_ORG>"  # your username on app.activeloop.ai
dataset_path = f"hub://{username}/langchain_testing"

docs = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
db = DeepLake(
    dataset_path=dataset_path,
    embedding=embeddings,
    overwrite=True,
    runtime={"tensor_db": True},
)
ids = db.add_documents(docs)
```

    Your Deep Lake dataset has been successfully created!
    

    |

    Dataset(path='hub://adilkhan/langchain_testing', tensors=['embedding', 'id', 'metadata', 'text'])
    
      tensor      htype      shape      dtype  compression
      -------    -------    -------    -------  ------- 
     embedding  embedding  (42, 1536)  float32   None   
        id        text      (42, 1)      str     None   
     metadata     json      (42, 1)      str     None   
       text       text      (42, 1)      str     None   
    

     

### TQL Search

Furthermore, the execution of queries is also supported within the similarity_search method, whereby the query can be specified utilizing Deep Lake's Tensor Query Language (TQL).


```python
search_id = db.vectorstore.dataset.id[0].numpy()
```


```python
search_id[0]
```




    '8a6ff326-3a85-11ee-b840-13905694aaaf'




```python
docs = db.similarity_search(
    query=None,
    tql=f"SELECT * WHERE id == '{search_id[0]}'",
)
```


```python
db.vectorstore.summary()
```

    Dataset(path='hub://adilkhan/langchain_testing', tensors=['embedding', 'id', 'metadata', 'text'])
    
      tensor      htype      shape      dtype  compression
      -------    -------    -------    -------  ------- 
     embedding  embedding  (42, 1536)  float32   None   
        id        text      (42, 1)      str     None   
     metadata     json      (42, 1)      str     None   
       text       text      (42, 1)      str     None   
    

### Creating vector stores on AWS S3


```python
dataset_path = "s3://BUCKET/langchain_test"  # could be also ./local/path (much faster locally), hub://bucket/path/to/dataset, gcs://path/to/dataset, etc.

embedding = OpenAIEmbeddings()
db = DeepLake.from_documents(
    docs,
    dataset_path=dataset_path,
    embedding=embeddings,
    overwrite=True,
    creds={
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "aws_session_token": os.environ["AWS_SESSION_TOKEN"],  # Optional
    },
)
```

    s3://hub-2.0-datasets-n/langchain_test loaded successfully.
    

    Evaluating ingest: 100%|██████████| 1/1 [00:10<00:00
    \

    Dataset(path='s3://hub-2.0-datasets-n/langchain_test', tensors=['embedding', 'ids', 'metadata', 'text'])
    
      tensor     htype     shape     dtype  compression
      -------   -------   -------   -------  ------- 
     embedding  generic  (4, 1536)  float32   None   
        ids      text     (4, 1)      str     None   
     metadata    json     (4, 1)      str     None   
       text      text     (4, 1)      str     None   
    

     

## Deep Lake API
you can access the Deep Lake  dataset at `db.vectorstore`


```python
# get structure of the dataset
db.vectorstore.summary()
```

    Dataset(path='hub://adilkhan/langchain_testing', tensors=['embedding', 'id', 'metadata', 'text'])
    
      tensor      htype      shape      dtype  compression
      -------    -------    -------    -------  ------- 
     embedding  embedding  (42, 1536)  float32   None   
        id        text      (42, 1)      str     None   
     metadata     json      (42, 1)      str     None   
       text       text      (42, 1)      str     None   
    


```python
# get embeddings numpy array
embeds = db.vectorstore.dataset.embedding.numpy()
```

### Transfer local dataset to cloud
Copy already created dataset to the cloud. You can also transfer from cloud to local.


```python
import deeplake

username = "davitbun"  # your username on app.activeloop.ai
source = f"hub://{username}/langchain_testing"  # could be local, s3, gcs, etc.
destination = f"hub://{username}/langchain_test_copy"  # could be local, s3, gcs, etc.

deeplake.deepcopy(src=source, dest=destination, overwrite=True)
```

    Copying dataset: 100%|██████████| 56/56 [00:38<00:00
    

    This dataset can be visualized in Jupyter Notebook by ds.visualize() or at https://app.activeloop.ai/davitbun/langchain_test_copy
    Your Deep Lake dataset has been successfully created!
    The dataset is private so make sure you are logged in!
    




    Dataset(path='hub://davitbun/langchain_test_copy', tensors=['embedding', 'ids', 'metadata', 'text'])




```python
db = DeepLake(dataset_path=destination, embedding=embeddings)
db.add_documents(docs)
```

     

    This dataset can be visualized in Jupyter Notebook by ds.visualize() or at https://app.activeloop.ai/davitbun/langchain_test_copy
    
    

    /

    hub://davitbun/langchain_test_copy loaded successfully.
    
    

    Deep Lake Dataset in hub://davitbun/langchain_test_copy already exists, loading from the storage
    

    Dataset(path='hub://davitbun/langchain_test_copy', tensors=['embedding', 'ids', 'metadata', 'text'])
    
      tensor     htype     shape     dtype  compression
      -------   -------   -------   -------  ------- 
     embedding  generic  (4, 1536)  float32   None   
        ids      text     (4, 1)      str     None   
     metadata    json     (4, 1)      str     None   
       text      text     (4, 1)      str     None   
    

    Evaluating ingest: 100%|██████████| 1/1 [00:31<00:00
    -

    Dataset(path='hub://davitbun/langchain_test_copy', tensors=['embedding', 'ids', 'metadata', 'text'])
    
      tensor     htype     shape     dtype  compression
      -------   -------   -------   -------  ------- 
     embedding  generic  (8, 1536)  float32   None   
        ids      text     (8, 1)      str     None   
     metadata    json     (8, 1)      str     None   
       text      text     (8, 1)      str     None   
    

     




    ['ad42f3fe-e188-11ed-b66d-41c5f7b85421',
     'ad42f3ff-e188-11ed-b66d-41c5f7b85421',
     'ad42f400-e188-11ed-b66d-41c5f7b85421',
     'ad42f401-e188-11ed-b66d-41c5f7b85421']


