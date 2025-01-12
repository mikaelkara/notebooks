# Azure AI Search as a vector database for OpenAI embeddings

This notebook provides step by step instuctions on using Azure AI Search (f.k.a Azure Cognitive Search) as a vector database with OpenAI embeddings. Azure AI Search is a cloud search service that gives developers infrastructure, APIs, and tools for building a rich search experience over private, heterogeneous content in web, mobile, and enterprise applications.

## Prerequistites:
For the purposes of this exercise you must have the following:
- [Azure AI Search Service](https://learn.microsoft.com/azure/search/)
- [OpenAI Key](https://platform.openai.com/account/api-keys) or [Azure OpenAI credentials](https://learn.microsoft.com/azure/cognitive-services/openai/)


```python
! pip install wget
! pip install azure-search-documents 
! pip install azure-identity
! pip install openai
```

## Import required libraries


```python
import json  
import wget
import pandas as pd
import zipfile
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

```

## Configure OpenAI settings

This section guides you through setting up authentication for Azure OpenAI, allowing you to securely interact with the service using either Azure Active Directory (AAD) or an API key. Before proceeding, ensure you have your Azure OpenAI endpoint and credentials ready. For detailed instructions on setting up AAD with Azure OpenAI, refer to the [official documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity).



```python
endpoint: str = "YOUR_AZURE_OPENAI_ENDPOINT"
api_key: str = "YOUR_AZURE_OPENAI_KEY"
api_version: str = "2023-05-15"
deployment = "YOUR_AZURE_OPENAI_DEPLOYMENT_NAME"
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

# Set this flag to True if you are using Azure Active Directory
use_aad_for_aoai = True 

if use_aad_for_aoai:
    # Use Azure Active Directory (AAD) authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )
else:
    # Use API key authentication
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
```

## Configure Azure AI Search Vector Store settings
This section explains how to set up the Azure AI Search client for integrating with the Vector Store feature. You can locate your Azure AI Search service details in the Azure Portal or programmatically via the [Search Management SDK](https://learn.microsoft.com/rest/api/searchmanagement/).



```python
# Configuration
search_service_endpoint: str = "YOUR_AZURE_SEARCH_ENDPOINT"
search_service_api_key: str = "YOUR_AZURE_SEARCH_ADMIN_KEY"
index_name: str = "azure-ai-search-openai-cookbook-demo"

# Set this flag to True if you are using Azure Active Directory
use_aad_for_search = True  

if use_aad_for_search:
    # Use Azure Active Directory (AAD) authentication
    credential = DefaultAzureCredential()
else:
    # Use API key authentication
    credential = AzureKeyCredential(search_service_api_key)

# Initialize the SearchClient with the selected authentication method
search_client = SearchClient(
    endpoint=search_service_endpoint, index_name=index_name, credential=credential
)
```

## Load data



```python
embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"

# The file is ~700 MB so this will take some time
wget.download(embeddings_url)
```




    'vector_database_wikipedia_articles_embedded.zip'




```python
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../../data")
```


```python
article_df = pd.read_csv("../../data/vector_database_wikipedia_articles_embedded.csv")

# Read vectors from strings back into a list using json.loads
article_df["title_vector"] = article_df.title_vector.apply(json.loads)
article_df["content_vector"] = article_df.content_vector.apply(json.loads)
article_df["vector_id"] = article_df["vector_id"].apply(str)
article_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>url</th>
      <th>title</th>
      <th>text</th>
      <th>title_vector</th>
      <th>content_vector</th>
      <th>vector_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>https://simple.wikipedia.org/wiki/April</td>
      <td>April</td>
      <td>April is the fourth month of the year in the J...</td>
      <td>[0.001009464613161981, -0.020700545981526375, ...</td>
      <td>[-0.011253940872848034, -0.013491976074874401,...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://simple.wikipedia.org/wiki/August</td>
      <td>August</td>
      <td>August (Aug.) is the eighth month of the year ...</td>
      <td>[0.0009286514250561595, 0.000820168002974242, ...</td>
      <td>[0.0003609954728744924, 0.007262262050062418, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>https://simple.wikipedia.org/wiki/Art</td>
      <td>Art</td>
      <td>Art is a creative activity that expresses imag...</td>
      <td>[0.003393713850528002, 0.0061537534929811954, ...</td>
      <td>[-0.004959689453244209, 0.015772193670272827, ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>https://simple.wikipedia.org/wiki/A</td>
      <td>A</td>
      <td>A or a is the first letter of the English alph...</td>
      <td>[0.0153952119871974, -0.013759135268628597, 0....</td>
      <td>[0.024894846603274345, -0.022186409682035446, ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>https://simple.wikipedia.org/wiki/Air</td>
      <td>Air</td>
      <td>Air refers to the Earth's atmosphere. Air is a...</td>
      <td>[0.02224554680287838, -0.02044147066771984, -0...</td>
      <td>[0.021524671465158463, 0.018522677943110466, -...</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



##  Create an index
This code snippet demonstrates how to define and create a search index using the `SearchIndexClient` from the Azure AI Search Python SDK. The index incorporates both vector search and semantic ranker capabilities. For more details, visit our documentation on how to [Create a Vector Index](https://learn.microsoft.com/azure/search/vector-search-how-to-create-index?.tabs=config-2023-11-01%2Crest-2023-11-01%2Cpush%2Cportal-check-index)


```python
# Initialize the SearchIndexClient
index_client = SearchIndexClient(
    endpoint=search_service_endpoint, credential=credential
)

# Define the fields for the index
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String),
    SimpleField(name="vector_id", type=SearchFieldDataType.String, key=True),
    SimpleField(name="url", type=SearchFieldDataType.String),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="text", type=SearchFieldDataType.String),
    SearchField(
        name="title_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,
        vector_search_profile_name="my-vector-config",
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,
        vector_search_profile_name="my-vector-config",
    ),
]

# Configure the vector search configuration
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="my-hnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="my-vector-config",
            algorithm_configuration_name="my-hnsw",
        )
    ],
)

# Configure the semantic search configuration
semantic_search = SemanticSearch(
    configurations=[
        SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                keywords_fields=[SemanticField(field_name="url")],
                content_fields=[SemanticField(field_name="text")],
            ),
        )
    ]
)

# Create the search index with the vector search and semantic search configurations
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_search=semantic_search,
)

# Create or update the index
result = index_client.create_or_update_index(index)
print(f"{result.name} created")
```

    azure-ai-search-openai-cookbook-demo created
    

## Uploading Data to Azure AI Search Index

The following code snippet outlines the process of uploading a batch of documents—specifically, Wikipedia articles with pre-computed embeddings—from a pandas DataFrame to an Azure AI Search index. For a detailed guide on data import strategies and best practices, refer to  [Data Import in Azure AI Search](https://learn.microsoft.com/azure/search/search-what-is-data-import).



```python
from azure.core.exceptions import HttpResponseError

# Convert the 'id' and 'vector_id' columns to string so one of them can serve as our key field
article_df["id"] = article_df["id"].astype(str)
article_df["vector_id"] = article_df["vector_id"].astype(str)
# Convert the DataFrame to a list of dictionaries
documents = article_df.to_dict(orient="records")

# Create a SearchIndexingBufferedSender
batch_client = SearchIndexingBufferedSender(
    search_service_endpoint, index_name, credential
)

try:
    # Add upload actions for all documents in a single call
    batch_client.upload_documents(documents=documents)

    # Manually flush to send any remaining documents in the buffer
    batch_client.flush()
except HttpResponseError as e:
    print(f"An error occurred: {e}")
finally:
    # Clean up resources
    batch_client.close()

print(f"Uploaded {len(documents)} documents in total")
```

    Uploaded 25000 documents in total
    

If your dataset didn't already contain pre-computed embeddings, you can create embeddings by using the below function using the `openai` python library. You'll also notice the same function and model are being used to generate query embeddings for performing vector searches.


```python
# Example function to generate document embedding
def generate_embeddings(text, model):
    # Generate embeddings for the provided text using the specified model
    embeddings_response = client.embeddings.create(model=model, input=text)
    # Extract the embedding data from the response
    embedding = embeddings_response.data[0].embedding
    return embedding


first_document_content = documents[0]["text"]
print(f"Content: {first_document_content[:100]}")

content_vector = generate_embeddings(first_document_content, deployment)
print("Content vector generated")
```

    Content: April is the fourth month of the year in the Julian and Gregorian calendars, and comes between March
    Content vector generated
    

## Perform a vector similarity search


```python
# Pure Vector Search
query = "modern art in Europe"
  
search_client = SearchClient(search_service_endpoint, index_name, credential)  
vector_query = VectorizedQuery(vector=generate_embeddings(query, deployment), k_nearest_neighbors=3, fields="content_vector")
  
results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query], 
    select=["title", "text", "url"] 
)
  
for result in results:  
    print(f"Title: {result['title']}")  
    print(f"Score: {result['@search.score']}")  
    print(f"URL: {result['url']}\n")  
```

    Title: Documenta
    Score: 0.8599451
    URL: https://simple.wikipedia.org/wiki/Documenta
    
    Title: Museum of Modern Art
    Score: 0.85260946
    URL: https://simple.wikipedia.org/wiki/Museum%20of%20Modern%20Art
    
    Title: Expressionism
    Score: 0.852354
    URL: https://simple.wikipedia.org/wiki/Expressionism
    
    

## Perform a Hybrid Search
Hybrid search combines the capabilities of traditional keyword-based search with vector-based similarity search to provide more relevant and contextual results. This approach is particularly useful when dealing with complex queries that benefit from understanding the semantic meaning behind the text.

The provided code snippet demonstrates how to execute a hybrid search query:


```python
# Hybrid Search
query = "Famous battles in Scottish history"  
  
search_client = SearchClient(search_service_endpoint, index_name, credential)  
vector_query = VectorizedQuery(vector=generate_embeddings(query, deployment), k_nearest_neighbors=3, fields="content_vector")
  
results = search_client.search(  
    search_text=query,  
    vector_queries= [vector_query], 
    select=["title", "text", "url"],
    top=3
)
  
for result in results:  
    print(f"Title: {result['title']}")  
    print(f"Score: {result['@search.score']}")  
    print(f"URL: {result['url']}\n")  
```

    Title: Wars of Scottish Independence
    Score: 0.03306011110544205
    URL: https://simple.wikipedia.org/wiki/Wars%20of%20Scottish%20Independence
    
    Title: Battle of Bannockburn
    Score: 0.022253260016441345
    URL: https://simple.wikipedia.org/wiki/Battle%20of%20Bannockburn
    
    Title: Scottish
    Score: 0.016393441706895828
    URL: https://simple.wikipedia.org/wiki/Scottish
    
    

## Perform a Hybrid Search with Reranking (powered by Bing)
[Semantic ranker](https://learn.microsoft.com/azure/search/semantic-search-overview) measurably improves search relevance by using language understanding to rerank search results. Additionally, you can get extractive captions, answers, and highlights. 


```python
# Semantic Hybrid Search
query = "What were the key technological advancements during the Industrial Revolution?"

search_client = SearchClient(search_service_endpoint, index_name, credential)
vector_query = VectorizedQuery(
    vector=generate_embeddings(query, deployment),
    k_nearest_neighbors=3,
    fields="content_vector",
)

results = search_client.search(
    search_text=query,
    vector_queries=[vector_query],
    select=["title", "text", "url"],
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name="my-semantic-config",
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
    top=3,
)

semantic_answers = results.get_answers()
for answer in semantic_answers:
    if answer.highlights:
        print(f"Semantic Answer: {answer.highlights}")
    else:
        print(f"Semantic Answer: {answer.text}")
    print(f"Semantic Answer Score: {answer.score}\n")

for result in results:
    print(f"Title: {result['title']}")
    print(f"Reranker Score: {result['@search.reranker_score']}")
    print(f"URL: {result['url']}")
    captions = result["@search.captions"]
    if captions:
        caption = captions[0]
        if caption.highlights:
            print(f"Caption: {caption.highlights}\n")
        else:
            print(f"Caption: {caption.text}\n")
```

    Semantic Answer: Advancements  During the industrial revolution, new technology brought many changes. For example:<em>   Canals</em> were built to allow heavy goods to be moved easily where they were needed. The steam engine became the main source of power. It replaced horses and human labor. Cheap iron and steel became mass-produced.
    Semantic Answer Score: 0.90478515625
    
    Title: Industrial Revolution
    Reranker Score: 3.408700942993164
    URL: https://simple.wikipedia.org/wiki/Industrial%20Revolution
    Caption: Advancements  During the industrial revolution, new technology brought many changes. For example:   Canals were built to allow heavy goods to be moved easily where they were needed. The steam engine became the main source of power. It replaced horses and human labor. Cheap iron and steel became mass-produced.
    
    Title: Printing
    Reranker Score: 1.603400707244873
    URL: https://simple.wikipedia.org/wiki/Printing
    Caption: Machines to speed printing, cheaper paper, automatic stitching and binding all arrived in the 19th century during the industrial revolution. What had once been done by a few men by hand was now done by limited companies on huge machines. The result was much lower prices, and a much wider readership.
    
    Title: Industrialisation
    Reranker Score: 1.3238357305526733
    URL: https://simple.wikipedia.org/wiki/Industrialisation
    Caption: <em>Industrialisation</em> (or<em> industrialization)</em> is a process that happens in countries when they start to use machines to do work that was once done by people.<em> Industrialisation changes</em> the things people do.<em> Industrialisation</em> caused towns to grow larger. Many people left farming to take higher paid jobs in factories in towns.
    
    
