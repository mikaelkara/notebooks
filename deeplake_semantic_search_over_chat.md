# QA using Activeloop's DeepLake
In this tutorial, we are going to use Langchain + Activeloop's Deep Lake with GPT4 to semantically search and ask questions over a group chat.

View a working demo [here](https://twitter.com/thisissukh_/status/1647223328363679745)

## 1. Install required packages


```python
!python3 -m pip install --upgrade langchain 'deeplake[enterprise]' openai tiktoken
```

## 2. Add API keys




```python
import getpass
import os

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
activeloop_token = getpass.getpass("Activeloop Token:")
os.environ["ACTIVELOOP_TOKEN"] = activeloop_token
os.environ["ACTIVELOOP_ORG"] = getpass.getpass("Activeloop Org:")

org_id = os.environ["ACTIVELOOP_ORG"]
embeddings = OpenAIEmbeddings()

dataset_path = "hub://" + org_id + "/data"
```



## 2. Create sample data

You can generate a sample group chat conversation using ChatGPT with this prompt:

```
Generate a group chat conversation with three friends talking about their day, referencing real places and fictional names. Make it funny and as detailed as possible.
```

I've already generated such a chat in `messages.txt`. We can keep it simple and use this for our example.

## 3. Ingest chat embeddings

We load the messages in the text file, chunk and upload to ActiveLoop Vector store.


```python
with open("messages.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
pages = text_splitter.split_text(state_of_the_union)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.create_documents(pages)

print(texts)

dataset_path = "hub://" + org_id + "/data"
embeddings = OpenAIEmbeddings()
db = DeepLake.from_documents(
    texts, embeddings, dataset_path=dataset_path, overwrite=True
)
```

    [Document(page_content='Participants:\n\nJerry: Loves movies and is a bit of a klutz.\nSamantha: Enthusiastic about food and always trying new restaurants.\nBarry: A nature lover, but always manages to get lost.\nJerry: Hey, guys! You won\'t believe what happened to me at the Times Square AMC theater. I tripped over my own feet and spilled popcorn everywhere! 🍿💥\n\nSamantha: LOL, that\'s so you, Jerry! Was the floor buttery enough for you to ice skate on after that? 😂\n\nBarry: Sounds like a regular Tuesday for you, Jerry. Meanwhile, I tried to find that new hiking trail in Central Park. You know, the one that\'s supposed to be impossible to get lost on? Well, guess what...\n\nJerry: You found a hidden treasure?\n\nBarry: No, I got lost. AGAIN. 🧭🙄\n\nSamantha: Barry, you\'d get lost in your own backyard! But speaking of treasures, I found this new sushi place in Little Tokyo. "Samantha\'s Sushi Symphony" it\'s called. Coincidence? I think not!\n\nJerry: Maybe they named it after your ability to eat your body weight in sushi. 🍣', metadata={}), Document(page_content='Barry: How do you even FIND all these places, Samantha?\n\nSamantha: Simple, I don\'t rely on Barry\'s navigation skills. 😉 But seriously, the wasabi there was hotter than Jerry\'s love for Marvel movies!\n\nJerry: Hey, nothing wrong with a little superhero action. By the way, did you guys see the new "Captain Crunch: Breakfast Avenger" trailer?\n\nSamantha: Captain Crunch? Are you sure you didn\'t get that from one of your Saturday morning cereal binges?\n\nBarry: Yeah, and did he defeat his arch-enemy, General Mills? 😆\n\nJerry: Ha-ha, very funny. Anyway, that sushi place sounds awesome, Samantha. Next time, let\'s go together, and maybe Barry can guide us... if we want a city-wide tour first.\n\nBarry: As long as we\'re not hiking, I\'ll get us there... eventually. 😅\n\nSamantha: It\'s a date! But Jerry, you\'re banned from carrying any food items.\n\nJerry: Deal! Just promise me no wasabi challenges. I don\'t want to end up like the time I tried Sriracha ice cream.', metadata={}), Document(page_content="Barry: Wait, what happened with Sriracha ice cream?\n\nJerry: Let's just say it was a hot situation. Literally. 🔥\n\nSamantha: 🤣 I still have the video!\n\nJerry: Samantha, if you value our friendship, that video will never see the light of day.\n\nSamantha: No promises, Jerry. No promises. 🤐😈\n\nBarry: I foresee a fun weekend ahead! 🎉", metadata={})]
    

    Your Deep Lake dataset has been successfully created!
    

    \

    Dataset(path='hub://adilkhan/data', tensors=['embedding', 'id', 'metadata', 'text'])
    
      tensor      htype      shape     dtype  compression
      -------    -------    -------   -------  ------- 
     embedding  embedding  (3, 1536)  float32   None   
        id        text      (3, 1)      str     None   
     metadata     json      (3, 1)      str     None   
       text       text      (3, 1)      str     None   
    

     

`Optional`: You can also use Deep Lake's Managed Tensor Database as a hosting service and run queries there. In order to do so, it is necessary to specify the runtime parameter as {'tensor_db': True} during the creation of the vector store. This configuration enables the execution of queries on the Managed Tensor Database, rather than on the client side. It should be noted that this functionality is not applicable to datasets stored locally or in-memory. In the event that a vector store has already been created outside of the Managed Tensor Database, it is possible to transfer it to the Managed Tensor Database by following the prescribed steps.


```python
# with open("messages.txt") as f:
#     state_of_the_union = f.read()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# pages = text_splitter.split_text(state_of_the_union)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# texts = text_splitter.create_documents(pages)

# print(texts)

# dataset_path = "hub://" + org + "/data"
# embeddings = OpenAIEmbeddings()
# db = DeepLake.from_documents(
#     texts, embeddings, dataset_path=dataset_path, overwrite=True, runtime={"tensor_db": True}
# )
```

## 4. Ask questions

Now we can ask a question and get an answer back with a semantic search:


```python
db = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["k"] = 4

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=False
)

# What was the restaurant the group was talking about called?
query = input("Enter query:")

# The Hungry Lobster
ans = qa({"query": query})

print(ans)
```
