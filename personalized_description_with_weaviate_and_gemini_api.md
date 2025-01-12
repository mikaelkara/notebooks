##### Copyright 2024 Google LLC.


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Personalized Product Descriptions with Weaviate and the Gemini API

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/weaviate/personalized_description_with_weaviate_and_gemini_api.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>


Weaviate is an open-source vector database that enables you to build AI-powered applications with the Gemini API! This notebook has four parts:
1. [Part 1: Connect to Weaviate, Define Schema, and Import Data](#part-1-install-dependencies-and-connect-to-weaviate)

2. [Part 2: Run Vector Search Queries](#part-2-vector-search)

3. [Part 3: Generative Feedback Loops](#part-3-generative-feedback-loops)

4. [Part 4: Personalized Product Descriptions](#part-4-personalization)


In this demo, you will learn how to embed your data, run a semantic search, make a generative call to the Gemini API and store the output in your vector database, and personalize the description based on the user profile.

## Use Case

You will be working with an e-commerce dataset containing Google merch. You will load the data into the Weaviate vector database and use the semantic search features to retrieve data. Next, you'll generate product descriptions and store them back into the database with a vector embedding for retrieval (aka, generative feedback loops). Lastly, you'll create a small knowledge graph with uniquely generated product descriptions for the buyer personas Alice and Bob.

## Requirements
You will need a running Weaviate cluster and Gemini API key. You'll set up these requirements as you progress through this notebook!

1. Weaviate vector database
    1. Serverless
    1. Embedded
    1. Local (Docker)
1. Gemini API key
    1. Create an API key via [AI Studio](https://aistudio.google.com/)

## Video
**For an awesome walk through of this demo, check out [this](https://youtu.be/WORgeRAAN-4?si=-WvqNkPn8oCmnLGQ&t=1138) presentation from Google Cloud Next!**

[![From RAG to autonomous apps with Weaviate and Gemini API on Google Kubernetes Engine](http://i3.ytimg.com/vi/WORgeRAAN-4/hqdefault.jpg)](https://youtu.be/WORgeRAAN-4?si=-WvqNkPn8oCmnLGQ&t=1138)

## Install Dependencies and Libraries


```
!sudo apt-get install python3.11
!pip install weaviate-client==4.7.1
!pip install -U -q "google-generativeai>=0.7.2"
!pip install requests
!pip install 'protobuf>=5'
```


```
import weaviate
from weaviate.classes.config import Configure
from weaviate.embedded import EmbeddedOptions
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, ReferenceProperty
from weaviate.util import generate_uuid5
from weaviate.classes.query import QueryReference

import os
import json
import requests
import PIL
import IPython

from PIL import Image
from io import BytesIO
from IPython.display import Markdown

import google
import google.generativeai as genai
from google.colab import userdata

# Convert image links to PIL object
def url_to_pil(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# display images
def display_image(url, size=100):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    image = Image.open(image_data)

    resized_image = image.resize((size,size))

    display(resized_image)
```

## Part 1: Connect to Weaviate, Define Schema, and Import Data

### Connect to Weaviate

You will need to create a Weaviate cluster. There are a few ways to do this:

1. [Weaviate Embedded](https://weaviate.io/developers/weaviate/installation/embedded): Run Weaviate in your runtime

2. [Weaviate Cloud](console.weaviate.cloud): Create a sandbox on our managed service. You will need to deploy it in US West, US East, or Australia.

3. Local Host: [Docker](https://weaviate.io/developers/weaviate/installation/docker-compose#starter-docker-compose-file) or [Kubernetes](https://weaviate.io/developers/weaviate/installation/kubernetes)

For the full list of installation options, see [this page](https://weaviate.io/developers/weaviate/installation).

#### Weaviate Embedded
We will default to Weaviate Embedded. This runs Weaviate inside your notebook and is ideal for quick experimentation. 

**Note: It will disconnect once you stop the terminal.**

**Set up your API key**

To run the following cell, your Gemini API key must be stored in a Colab Secret and named `GEMINI_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.


```
# Grab Gemini API key 
GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
```


```
client = weaviate.WeaviateClient(
    embedded_options=EmbeddedOptions(
        version="1.25.10",
        additional_env_vars={
            "ENABLE_MODULES": "text2vec-palm, generative-palm"
        }),
        additional_headers={
            "X-Google-Studio-Api-Key": GEMINI_API_KEY 
        }
)

client.connect()
```

#### Other Options: Weaviate Cloud and Local Host

#### **Weaviate Cloud**

You can connect your notebook to a serverless Weaviate cluster to keep the data persistent in the cloud. You can register [here](https://console.weaviate.cloud/) and create a free 14-day sandbox!

To connect to your WCD cluster:
```python
WCD_URL = "https://sandbox.gcp.weaviate.cloud"
WCD_AUTH_KEY = "sk-key"
GEMINI_API_KEY = "sk-key"

client = weaviate.connect_to_wcs(
    cluster_url=WCD_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WCD_AUTH_KEY),
    headers={"X-Google-Studio-Api-Key": GEMINI_API_KEY},
)

print(client.is_ready())
```

#### **Local Host**

If you want to run Weaviate yourself, you can download the [Docker files](https://weaviate.io/developers/weaviate/installation/docker-compose) and run it locally on your machine or in the cloud. There is also a `yaml` file in this folder you can use.

To connect to Weaviate locally:
```python
client = weaviate.connect_to_local()

print(client.is_ready())
```

### Create schema
The schema tells Weaviate how you want to store your data. 

You will first create two collections: Products and Personas. Each collection has metadata (properties) and specifies the embedding and language model.

In [Part 4](#part-4-personalization), you will create another collection, `Personalized`, that will generate product descriptions based on the persona. 


```
# This is optional to empty your database
result = client.collections.delete("Products")
print(result)
result = client.collections.delete("Personas")
print(result)
```


```
PROJECT_ID = "" # leave this empty
API_ENDPOINT = "generativelanguage.googleapis.com"
embedding_model = "embedding-001" # embedding model 
generative_model = "gemini-pro" # language model 

# Products Collection
if not client.collections.exists("Products"):
  collection = client.collections.create(
    name="Products",
    vectorizer_config=Configure.Vectorizer.text2vec_palm
    (
        project_id=PROJECT_ID,
        api_endpoint=API_ENDPOINT,
        model_id = embedding_model
    ),
    generative_config=Configure.Generative.palm(
        project_id=PROJECT_ID,
        api_endpoint=API_ENDPOINT,
        model_id = generative_model
    ),
    properties=[ # properties for the Products collection
            Property(name="product_id", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
            Property(name="link", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="brand", data_type=DataType.TEXT),
            Property(name="generated_description", data_type=DataType.TEXT),
      ]
  )

# Personas Collection
if not client.collections.exists("Personas"):
  collection = client.collections.create(
    name="Personas",
    vectorizer_config=Configure.Vectorizer.text2vec_palm
    (
        project_id=PROJECT_ID,
        api_endpoint=API_ENDPOINT,
        model_id = embedding_model
    ),
    generative_config=Configure.Generative.palm(
        project_id=PROJECT_ID,
        api_endpoint=API_ENDPOINT,
        model_id = generative_model
    ),
    properties=[ # properties for the Personas collection
            Property(name="name", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
      ]
  )
```

### Import Objects


```
# URL to the raw JSON file
url = 'https://raw.githubusercontent.com/bkauf/next-store/main/first_99_objects.json'
response = requests.get(url)

# Load the entire JSON content
data = json.loads(response.text)
```


```
# Print first object
data[0]
```




    {'id': 'id_1',
     'product_id': 'GGOEGAYC135814',
     'title': 'Google Badge Tee',
     'category': 'Apparel  Accessories Tops  Tees Tshirts',
     'link': 'https://shop.googlemerchandisestore.com/store/20160512512/assets/items/images/GGOEGXXX1358.jpg',
     'description': 'A classic crew neck tee made from 100 cotton Its soft and comfortable and features a small Google logo on the chest',
     'color': "['Blue']",
     'gender': 'Unisex',
     'brand': 'Google'}



#### Upload to Weaviate

To make sure everything is set, you will upload only one object and confirm it's in the database. 


```
products = client.collections.get("Products")

first_object = data[0]

products.data.insert(
    properties={
        "product_id": first_object['product_id'],
        "title": first_object['title'],
        "category": first_object['category'],
        "link": first_object['link'],
        "description": first_object['description'],
        "brand": first_object['brand']
    }
)

response = products.aggregate.over_all(total_count=True)
print(response.total_count) # This should output 1
```

Let's import the remainder of our dataset. You will use Weaviate's batch import to get the 98 objects into our database.


```
products = client.collections.get("Products")

remaining_data = data[1:]

with products.batch.dynamic() as batch:
  for item in remaining_data:
    batch.add_object(
      properties={
        "product_id": item['product_id'],
        "title": item['title'],
        "category": item['category'],
        "link": item['link'],
        "description": item['description'],
        "brand": item['brand']
    }
  )

response = products.aggregate.over_all(total_count=True)
print(response.total_count) # this should print 99 
```


```
# print the objects uuid and properties
for product in products.iterator():
    print(product.uuid, product.properties)
```

You will fetch the object by the UUID that was created. It will print out the vector embedding as well!


```
product = products.query.fetch_object_by_id(
    product.uuid,
    include_vector=True
)

print(product.properties["title"], product.vector["default"])
```

## Part 2: Vector Search

### Vector Search
Vector search returns the objects with most similar vectors to that of the query. You will use the `near_text` operator to find objects with the nearest vector to an input text.


```
products = client.collections.get("Products")

response = products.query.near_text(
        query="travel mug",
        return_properties=["title", "description", "link"], # only return these 3 properties
        limit=3 # limited to 3 objects
)

for product in response.objects:
    print(json.dumps(product.properties, indent=2))
    display_image(product.properties['link'])
    print('===')
```

### Hybrid Search
[Hybrid search](https://weaviate.io/developers/weaviate/search/hybrid) combines keyword (BM25) and vector search together, giving you the best of both algorithms.

To use hybrid search in Weaviate, all you have to do is define the `alpha` parameter to determine the weighting.

`alpha` = 0 --> pure BM25

`alpha` = 0.5 --> half BM25, half vector search

`alpha` = 1 --> pure vector search


```
products = client.collections.get("Products")

response = products.query.hybrid(
    query = "dishwasher safe container", # query
    alpha = 0.75, # leaning more towards vector search
    return_properties=["title", "description", "link"], # return these 3 properties
    limit = 3 # limited to only 3 objects
)

for product in response.objects:
    print(json.dumps(product.properties, indent=2))
    display_image(product.properties['link'])
    print('===')
```

### Autocut
Rather than hard-coding the limit on the number of objects (seen above), the [autocut](https://weaviate.io/developers/weaviate/api/graphql/additional-operators#autocut) feature can be used to cut off the result set. Autocut limits the number of results returned based on significant variations in the result set's metrics, such as vector distance or score.


To use autocut, you must specify the `auto_limit` parameter, which will stop returning results after the specified number of variations, or "jumps" is reached.

You will use the same hybrid search query above but use `auto_limit` rather than `limit`. Notice how there are actually 4 objects retrieved in this case, compared to the 3 objects returned in the previous query.


```
# auto_limit set to 1

products = client.collections.get("Products")

response = products.query.hybrid(
    query = "dishwasher safe container", # query
    alpha = 0.75, # leaning more towards vector search
    return_properties=["title", "description", "link"], # return these 3 properties
    auto_limit = 1 # autocut after 1 jump
)

for product in response.objects:
    print(json.dumps(product.properties, indent=2))
    display_image(product.properties['link'])
    print('===')
```

### Filters
Narrow down the results by adding a filter to the query.

Find objects where `category` is equal to `drinkware`.


```
products = client.collections.get("Products")

response = products.query.near_text(
    query="travel cup",
    return_properties=["title", "description", "category", "link"], # returned properties
    filters=wvc.query.Filter.by_property("category").equal("Drinkware"), # filter
    limit=3, # limit to 3 objects
)

for product in response.objects:
    print(json.dumps(product.properties, indent=2))
    display_image(product.properties['link'])
    print('===')
```

## Part 3: Generative Feedback Loops

[Generative Feedback Loops](https://weaviate.io/blog/generative-feedback-loops-with-llms) refers to the process of storing the output from the language model back to the database.

You will generate a description for each product in our database using the Gemini API and save it to the `generated_description` property in the `Products` collection.

### Connect and configure the Gemini API model

Make sure you have set your Gemini API key in `GEMINI_API_KEY`. Please confirm this step was done in [Part 1](#part-1-connect-to-weaviate-define-schema-and-import-data).


```
genai.configure(api_key=GEMINI_API_KEY) # gemini api key

gemini_flash_model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest') # this model handles both images and text
```

### Generate a description and store it in the `Products` collection

Steps for the below cell:
1. Run a vector search query to find travel jackets 
    * Learn more about autocut (`auto_limit`) [here](https://weaviate.io/developers/weaviate/api/graphql/additional-operators#autocut).

2. Grab the returned objects, prompt the Gemini API with the task and image, store the description in the `generated_description` property


```
response = products.query.near_text( # first find travel jackets
    query="travel jacket",
    return_properties=["title", "description", "category", "link"],
    auto_limit=1, # limit it to 1 close group
)

for product in response.objects:
    if "link" in product.properties:
        id = product.uuid
        img_url = product.properties["link"]

        pil_image = url_to_pil(img_url) # convert image to PIL object
        generated_description = gemini_flash_model.generate_content(["Write a short Facebook ad about this product photo.", pil_image]) # prompt to the Gemini API
        generated_description = generated_description.text
        display_image(product.properties['link'])
        print(generated_description)
        print('===')

        # Update the Product collection with the generated description
        products.data.update(uuid=id, properties={"generated_description": generated_description})
```

### Vector Search on the `generated_description` property

Since the product description was saved in our `Products` collection, you can run a vector search query on it.


```
products = client.collections.get("Products")

response = products.query.near_text(
        query="travel jacket",
        return_properties=["generated_description", "description", "title"],
        limit=1
    )

for o in response.objects:
    print(o.uuid)
    print(json.dumps(o.properties, indent=2))
```

## Part 4: Personalization

So far, you've generated product descriptions using the Gemini API with the `gemini-1.5-flash` model. In Part 4, you will generate product descriptions tailored to the persona.

You will use [cross-references](https://weaviate.io/developers/weaviate/manage-data/cross-references) to establish directional relationships between the Products and Personas collections.


```
result = client.collections.delete("Personalized")
print(result)
```


```
PROJECT_ID = "" # leave this empty
API_ENDPOINT = "generativelanguage.googleapis.com"
embedding_model = "embedding-001" # embedding model 
generative_model = "gemini-pro" # language mdodel 

# Personalized Collection

if not client.collections.exists("Personalized"):
  collection = client.collections.create(
    name="Personalized",
    vectorizer_config=Configure.Vectorizer.text2vec_palm(
        project_id=PROJECT_ID,
        api_endpoint=API_ENDPOINT,
        model_id = embedding_model
    ),
    generative_config=Configure.Generative.palm(
        project_id=PROJECT_ID,
        api_endpoint=API_ENDPOINT,
        model_id = generative_model
    ),
    properties=[
            Property(name="description", data_type=DataType.TEXT),
    ],
    # cross-references
    references=[
        ReferenceProperty(
            name="ofProduct",
            target_collection="Products" # connect personalized to the products collection
        ),
        ReferenceProperty(
            name="ofPersona",
            target_collection="Personas" # connect personalized to the personas collection
        )
    ]
)
```

### Create two personas (Alice and Bob)


```
personas = client.collections.get("Personas")

for persona in ['Alice', 'Bob']:
  generated_description = gemini_flash_model.generate_content(["Create a fictional buyer persona named " + persona + ", write a short description about them"]) # use gemini-pro to generate persona description
  uuid = personas.data.insert({
    "name": persona,
    "description": generated_description.text
  })
  print(uuid)
  print(generated_description.text)
  print("===")
```


```
# print objects in the Personas collection

personas = client.collections.get("Personas")

for persona in personas.iterator():
    print(persona.uuid, persona.properties)
```

### Generate a product description tailored to the persona

Grab the product uuid from Part 1 and paste it below


```
personalized = client.collections.get("Personalized")

product = products.query.fetch_object_by_id(product.uuid)
display_image(product.properties['link'])

personas = client.collections.get("Personas")

for persona in personas.iterator():
    generated_description = gemini_flash_model.generate_content(["Create a product description tailored to the following person, make sure to use the name (", persona.properties["name"],") of the persona.\n\n", "# Product Description\n", product.properties["description"], "# Persona", persona.properties["description"]]) # generate a description tailored to the persona
    print(generated_description.text)
    print('====')
    # Add the personalized description to the `description` property in the Personalized collection
    new_uuid = personalized.data.insert(
        properties={
            "description": generated_description.text },
        references={
            "ofProduct": product.uuid, # add cross-reference to the Product collection
            "ofPersona": persona.uuid # add cross-reference to the Persona collection
        },
    )

```

### Fetch the objects in the `Personalized` collection


```
personalized = client.collections.get("Personalized")

response = personalized.query.fetch_objects(
    limit=2,
    return_references=[QueryReference(
            link_on="ofProduct", # return the title property from the Product collection
            return_properties=["title", "link"]
        ),
        QueryReference(
            link_on="ofPersona",
            return_properties=["name"] # return the name property from the Persona collection
        )
    ]
)

for item in response.objects:
    print(item.properties)
    for ref_obj in item.references["ofProduct"].objects:
        print(ref_obj.properties)
    for ref_obj in item.references["ofPersona"].objects:
        print(ref_obj.properties)
    display_image(product.properties['link'])
    print("===")
```

## Notebook Recap

In this notebook, you learned how to:
1. Create a Weaviate cluster using Embedded
2. Define a Weaviate schema and select the embedding and generative model
3. Connect to the Gemini API
4. Perform vector and hybrid search with filtering and autocut 
6. Use Generative Feedback Loops to store the output of the language model back to the database for future retrieval
7. Use cross-references to build relationships between collections

You can learn more about Weaviate through our [documentation](https://weaviate.io/developers/weaviate), and you can find more Weaviate and Google cookbooks [here](https://github.com/weaviate/recipes/tree/main/integrations/cloud-hyperscalers/google)!

**Authors: Erika Cardenas and Bob Van Luijt** 

Connect with us and let us know if you have any questions!

Erika's accounts:
* [Follow on X](https://x.com/ecardenas300)
* [Connect on LinkedIn](https://www.linkedin.com/in/erikacardenas300/)

Bob's accounts:
* [Follow on X](https://x.com/bobvanluijt)
* [Connect on LinkedIn](https://www.linkedin.com/in/bobvanluijt/)
