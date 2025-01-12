<a target="_blank" href="https://colab.research.google.com/drive/1Z7CQ5LE6p7TFl-QFxv-F_RrKB8_diB3C?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Koda Retriever + MongoDB + Fireworks: package solution for best RAG data quality

As people are building more advanced abstractions on top of RAG, we can compose that together with MongoDB and Fireworks to provide a turn key solution that provides the best quality out of the box. Koda retriever from LlamaIndex is a good example, it implements hybrid retrieval out of the box, and would only need users to some config tunings on the weight of different search methods to be able to get the best results. In this case, we will make use of
- Fireworks embeddings, reranker and LLMs to drive the ranking end to end
- MongoDB Atlas for embedding features

For more information on how Koda Retriever pack for LlamaIndex works, please check out [LlamaIndex's page for Koda Retriever](https://github.com/run-llama/llama_index/tree/7ce7058d0f781e7ebd8f73d40e8888471f867af0/llama-index-packs/llama-index-packs-koda-retriever)

We will start from the beginning, perform the data import, embed the data, and then use Koda Retriever on top of the imported data

## Basic installations
We will install all the relevant dependencies for Fireworks, MongoDB, Koda Retriever


```python
!pip install -q llama-index llama-index-llms-fireworks llama-index-embeddings-fireworks pymongo
!pip install -q llama-index-packs-koda-retriever llama-index-vector-stores-mongodb datasets
```


```python
from llama_index.llms.fireworks import Fireworks
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.fireworks import FireworksEmbedding
from llama_index.core.postprocessor import LLMRerank 
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.packs.koda_retriever import KodaRetriever
import os
```


```python
import pymongo

mongo_url = input()
mongo_client = pymongo.MongoClient(mongo_url)
```


```python
DB_NAME = "movies"
COLLECTION_NAME = "movies_records"

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
```

we are going to delete this collection to clean up all the previous documents inserted.


```python
collection.delete_many({})
```




    DeleteResult({'n': 18067, 'electionId': ObjectId('7fffffff00000000000001cf'), 'opTime': {'ts': Timestamp(1709534085, 5295), 't': 463}, 'ok': 1.0, '$clusterTime': {'clusterTime': Timestamp(1709534085, 5295), 'signature': {'hash': b'\xac;\x86\x9b\xbe\xb4\xa7\xed\xe6\x03\xc0jY\xb8p\xef\x9a\x05\xe7\xce', 'keyId': 7294687148333072386}}, 'operationTime': Timestamp(1709534085, 5295)}, acknowledged=True)




```python
# set up Fireworks.ai Key
import os
import getpass

fw_api_key = getpass.getpass("Fireworks API Key:")
os.environ["FIREWORKS_API_KEY"] = fw_api_key
```


```python
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

Settings.llm = Fireworks()
Settings.embed_model = FireworksEmbedding()


vector_store = MongoDBAtlasVectorSearch(    mongo_client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    index_name="vector_index",
)
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=Settings.embed_model
)

reranker = LLMRerank(llm=Settings.llm)

```

## data import
We are going to import from MongoDB's huggingface dataset with 25k restaurants


```python
from datasets import load_dataset
import pandas as pd

# https://huggingface.co/datasets/AIatMongoDB/whatscooking.restaurants
dataset = load_dataset("AIatMongoDB/whatscooking.restaurants")

# Convert the dataset to a pandas dataframe

dataset_df = pd.DataFrame(dataset["train"])

import json
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

# Convert the DataFrame to a JSON string representation
documents_json = dataset_df.to_json(orient="records")
# Load the JSON string into a Python list of dictionaries
documents_list = json.loads(documents_json)

llama_documents = []

for document in documents_list:
    # Value for metadata must be one of (str, int, float, None)
    document["name"] = json.dumps(document["name"])
    document["cuisine"] = json.dumps(document["cuisine"])
    document["attributes"] = json.dumps(document["attributes"])
    document["menu"] = json.dumps(document["menu"])
    document["borough"] = json.dumps(document["borough"])
    document["address"] = json.dumps(document["address"])
    document["PriceRange"] = json.dumps(document["PriceRange"])
    document["HappyHour"] = json.dumps(document["HappyHour"])
    document["review_count"] = json.dumps(document["review_count"])
    del document["embedding"]
    del document["location"]

    # Create a Document object with the text and excluded metadata for llm and embedding models
    llama_document = Document(
        text=json.dumps(document),
        metadata=document,
        metadata_template="{key}=>{value}",
        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
    )

    llama_documents.append(llama_document)
```


```python
embed_model = FireworksEmbedding(
    embed_batch_size=512,
    model_name="nomic-ai/nomic-embed-text-v1.5",
    api_key=fw_api_key,
)
```


```python
print(
    "\nThe Embedding model sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.EMBED),
)

```

    
    The Embedding model sees this: 
     Metadata: _id=>{'$oid': '6095a34a7c34416a90d3206b'}
    DogsAllowed=>None
    TakeOut=>True
    sponsored=>None
    review_count=>10
    OutdoorSeating=>True
    HappyHour=>null
    cuisine=>"Tex-Mex"
    PriceRange=>1.0
    address=>{"building": "627", "coord": [-73.975981, 40.745132], "street": "2 Avenue", "zipcode": "10016"}
    restaurant_id=>40366661
    menu=>null
    attributes=>{"Alcohol": "'none'", "Ambience": "{'romantic': False, 'intimate': False, 'classy': False, 'hipster': False, 'divey': False, 'touristy': False, 'trendy': False, 'upscale': False, 'casual': False}", "BYOB": null, "BestNights": null, "BikeParking": null, "BusinessAcceptsBitcoin": null, "BusinessAcceptsCreditCards": null, "BusinessParking": "None", "Caters": "True", "DriveThru": null, "GoodForDancing": null, "GoodForKids": "True", "GoodForMeal": null, "HasTV": "True", "Music": null, "NoiseLevel": "'average'", "RestaurantsAttire": "'casual'", "RestaurantsDelivery": "True", "RestaurantsGoodForGroups": "True", "RestaurantsReservations": "True", "RestaurantsTableService": "False", "WheelchairAccessible": "True", "WiFi": "'free'"}
    name=>"Baby Bo'S Burritos"
    borough=>"Manhattan"
    stars=>2.5
    -----
    Content: {"_id": {"$oid": "6095a34a7c34416a90d3206b"}, "DogsAllowed": null, "TakeOut": true, "sponsored": null, "review_count": "10", "OutdoorSeating": true, "HappyHour": "null", "cuisine": "\"Tex-Mex\"", "PriceRange": "1.0", "address": "{\"building\": \"627\", \"coord\": [-73.975981, 40.745132], \"street\": \"2 Avenue\", \"zipcode\": \"10016\"}", "restaurant_id": "40366661", "menu": "null", "attributes": "{\"Alcohol\": \"'none'\", \"Ambience\": \"{'romantic': False, 'intimate': False, 'classy': False, 'hipster': False, 'divey': False, 'touristy': False, 'trendy': False, 'upscale': False, 'casual': False}\", \"BYOB\": null, \"BestNights\": null, \"BikeParking\": null, \"BusinessAcceptsBitcoin\": null, \"BusinessAcceptsCreditCards\": null, \"BusinessParking\": \"None\", \"Caters\": \"True\", \"DriveThru\": null, \"GoodForDancing\": null, \"GoodForKids\": \"True\", \"GoodForMeal\": null, \"HasTV\": \"True\", \"Music\": null, \"NoiseLevel\": \"'average'\", \"RestaurantsAttire\": \"'casual'\", \"RestaurantsDelivery\": \"True\", \"RestaurantsGoodForGroups\": \"True\", \"RestaurantsReservations\": \"True\", \"RestaurantsTableService\": \"False\", \"WheelchairAccessible\": \"True\", \"WiFi\": \"'free'\"}", "name": "\"Baby Bo'S Burritos\"", "borough": "\"Manhattan\"", "stars": 2.5}
    

We are going to use SentenceSplitter to split the documents. We will also reduce the size to 2.5k documents to make sure this example fits into a free MongoDB Atlas instance


```python
from llama_index.core.node_parser import SentenceSplitter
parser = SentenceSplitter(chunk_size=4096)
nodes = parser.get_nodes_from_documents(llama_documents[:2500])
```


```python
# embed all the nodes
node_embeddings = embed_model(nodes)
```


```python
for idx, n in enumerate(nodes):
  n.embedding = node_embeddings[idx].embedding
  if "_id" in n.metadata:
    del n.metadata["_id"]
```


```python
vector_store.add(nodes)
```

And add the following index in MongoDB Atlas. We will name it `vector_index`

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "dotProduct"
    }
  ]
}
```

## Using Koda Retriever

With all that preparation work, now we are finally going to show case Koda Retriever!

Koda Retriever has most of the settings prepared out of the box for you, so we can plug the index, llm, reranker and make the example run out of the box. For more advanced settings on how to run KodaRetriever, please check out the guide [here](https://github.com/run-llama/llama_index/tree/7ce7058d0f781e7ebd8f73d40e8888471f867af0/llama-index-packs/llama-index-packs-koda-retriever)


```python
retriever = KodaRetriever(
    index=vector_index,
    llm=Settings.llm,
    reranker=reranker,
    verbose=True,
)

```

And now we will query the retriever to find some bakery recommendations in Manhattan


```python
query = "search_query: Any recommendations for bakeries in Manhattan?"
query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
response = query_engine.query(query)
print(response)
```

    Based on the context provided, I would recommend two bakeries in Manhattan. The first one is Lung Moon Bakery, which offers a variety of baked goods such as Stuffed Croissants, Gourmet Doughnuts, Brownies, and Cookie sandwiches with icing in the middle. They have outdoor seating and take-out options available. The second recommendation is Zaro's Bread Basket, which has a selection including Stuffed Croissants, Pecan tart, Chocolate strawberries, and Lemon cupcakes. Zaro's Bread Basket also offers delivery and has a bike parking facility. Both bakeries are rated quite well by their customers.
    

That is it! A solution with that combines query engine, embedding, llms and reranker all in just a few lines of LlamaIndex code. Fireworks is here to support you for your complete RAG journey, and please let us know if there are any other high quality RAG setups like Koda Retriever that you want us to support in our Discord channel.
