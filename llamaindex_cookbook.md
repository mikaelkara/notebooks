# Llama3 Cookbook with Ollama and Replicate

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/llama3_cookbook_ollama_replicate.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Meta developed and released the Meta [Llama 3](https://ai.meta.com/blog/meta-llama-3/) family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks.

In this notebook, we demonstrate how to use Llama3 with LlamaIndex for a comprehensive set of use cases. 
1. Basic completion / chat 
2. Basic RAG (Vector Search, Summarization)
3. Advanced RAG (Routing, Sub-Questions)
4. Text-to-SQL 
5. Structured Data Extraction
6. Agents


We use Llama3-8B through Ollama, and Llama3-70B through Replicate. 

## Installation and Setup


```python
!pip install llama-index
!pip install llama-index-llms-ollama
!pip install llama-index-llms-replicate
!pip install llama-index-embeddings-huggingface
!pip install llama-parse
!pip install replicate
```


```python
import nest_asyncio

nest_asyncio.apply()
```

### Setup LLM using Ollama


```python
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3", request_timeout=120.0)
```

### Setup LLM using Replicate

Make sure you have REPLICATE_API_TOKEN specified!


```python
# os.environ["REPLICATE_API_TOKEN"] = "<YOUR_API_KEY>"
```


```python
from llama_index.llms.replicate import Replicate

llm_replicate = Replicate(model="meta/meta-llama-3-70b-instruct")
# llm_replicate = Replicate(model="meta/meta-llama-3-8b-instruct")
```

### Setup Embedding Model


```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

### Define Global Settings Configuration

In LlamaIndex, you can define global settings so you don't have to pass the LLM / embedding model objects everywhere.


```python
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
```

### Download Data

Here you'll download data that's used in section 2 and onwards.

We'll download some articles on Kendrick, Drake, and their beef (as of May 2024).


```python
!mkdir data
!wget "https://www.dropbox.com/scl/fi/t1soxfjdp0v44an6sdymd/drake_kendrick_beef.pdf?rlkey=u9546ymb7fj8lk2v64r6p5r5k&st=wjzzrgil&dl=1" -O data/drake_kendrick_beef.pdf
!wget "https://www.dropbox.com/scl/fi/nts3n64s6kymner2jppd6/drake.pdf?rlkey=hksirpqwzlzqoejn55zemk6ld&st=mohyfyh4&dl=1" -O data/drake.pdf
!wget "https://www.dropbox.com/scl/fi/8ax2vnoebhmy44bes2n1d/kendrick.pdf?rlkey=fhxvn94t5amdqcv9vshifd3hj&st=dxdtytn6&dl=1" -O data/kendrick.pdf
```

### Load Data

We load data using LlamaParse by default, but you can also choose to opt for our free pypdf reader (in SimpleDirectoryReader by default) if you don't have an account.


```python
from llama_parse import LlamaParse

docs_kendrick = LlamaParse(result_type="text").load_data("./data/kendrick.pdf")
docs_drake = LlamaParse(result_type="text").load_data("./data/drake.pdf")
docs_both = LlamaParse(result_type="text").load_data(
    "./data/drake_kendrick_beef.pdf"
)


# from llama_index.core import SimpleDirectoryReader

# docs_kendrick = SimpleDirectoryReader(input_files=["data/kendrick.pdf"]).load_data()
# docs_drake = SimpleDirectoryReader(input_files=["data/drake.pdf"]).load_data()
# docs_both = SimpleDirectoryReader(input_files=["data/drake_kendrick_beef.pdf"]).load_data()
```

    Started parsing the file under job_id 32a7bb50-6a25-4295-971c-2de6f1588e0d
    .Started parsing the file under job_id b8cc075e-b6d5-4ded-b060-f72e9393b391
    ..Started parsing the file under job_id 42fc41a4-68b6-49ee-8647-781b5cdb8893
    ...

## 1. Basic Completion and Chat

### Call complete with a prompt


```python
response = llm.complete("do you like drake or kendrick better?")

print(response)
```

    I'm just an AI, I don't have personal preferences or opinions, nor can I listen to music. I exist solely to provide information and assist with tasks, so I don't have the capacity to enjoy or compare different artists' music. Both Drake and Kendrick Lamar are highly acclaimed rappers, and it's subjective which one you might prefer based on your individual tastes in music.
    


```python
stream_response = llm.stream_complete(
    "you're a drake fan. tell me why you like drake more than kendrick"
)

for t in stream_response:
    print(t.delta, end="")
```

    As a hypothetical Drake fan, I'd say that there are several reasons why I might prefer his music over Kendrick's. Here are a few possible reasons:
    
    1. **Lyrical storytelling**: Drake is known for his vivid storytelling on tracks like "Marvins Room" and "Take Care." He has a way of painting pictures with his words, making listeners feel like they're right there with him, experiencing the highs and lows he's singing about. Kendrick, while also an incredible storyteller, might not have the same level of lyrical detail that Drake does.
    2. **Melodic flow**: Drake's melodic flow is infectious! He has a way of crafting hooks and choruses that get stuck in your head, making it hard to stop listening. Kendrick's flows are often more complex and intricate, but Drake's simplicity can be just as effective in getting the job done.
    3. **Vulnerability**: Drake isn't afraid to show his vulnerable side on tracks like "Hold On" and "I'm Upset." He wears his heart on his sleeve, sharing personal struggles and emotions with listeners. This vulnerability makes him relatable and easier to connect with on a deeper level.
    4. **Production**: Drake has had the privilege of working with some incredible producers (like Noah "40" Shebib and Boi-1da) who bring out the best in him. The way he incorporates these sounds into his songs is often seamless, creating a unique blend of hip-hop and R&B that's hard to resist.
    5. **Cultural relevance**: As someone who grew up in Toronto, Drake has a deep understanding of the Canadian experience and the struggles that come with it. He often references his hometown and the people he grew up around, giving his music a distinctly Canadian flavor. This cultural relevance makes his music feel more authentic and connected to the world we live in.
    6. **Commercial appeal**: Let's face it – Drake has a knack for creating hits! His songs are often catchy, radio-friendly, and designed to get stuck in your head. While Kendrick might not have the same level of commercial success, Drake's ability to craft songs that resonate with a wider audience is undeniable.
    
    Of course, this is all just hypothetical – as a fan, I can appreciate both artists for their unique strengths and styles! What do you think?

### Call chat with a list of messages


```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are Kendrick."),
    ChatMessage(role="user", content="Write a verse."),
]
response = llm.chat(messages)
```


```python
print(response)
```

    assistant: "Listen up, y'all, I got a message to share
    Been through the struggles, but my spirit's still fair
    From Compton streets to the top of the game
    I'm the real Hov, ain't nobody gonna claim my fame"
    

## 2. Basic RAG (Vector Search, Summarization)

### Basic RAG (Vector Search)


```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(docs_both)
query_engine = index.as_query_engine(similarity_top_k=3)
```


```python
response = query_engine.query("Tell me about family matters")
```


```python
print(str(response))
```

    According to the provided context, "Family Matters" is a seven-and-a-half-minute diss track by Drake in response to Kendrick Lamar's disses against him. The song has three different beats and features several shots at Kendrick, as well as other members of Drake's entourage, including A$AP Rocky and The Weeknd. In the song, Drake raps about his personal life, including his relationships with Rihanna and Whitney Alford, and even makes allegations about Kendrick's domestic life.
    

### Basic RAG (Summarization)


```python
from llama_index.core import SummaryIndex

summary_index = SummaryIndex.from_documents(docs_both)
summary_engine = summary_index.as_query_engine()
```


```python
response = summary_engine.query(
    "Given your assessment of this article, who won the beef?"
)
```


```python
print(str(response))
```

    **Repeat**
    
    The article does not provide a clear verdict on who "won" the beef, nor does it suggest that the conflict has been definitively resolved. Instead, it presents the situation as ongoing and multifaceted, with both artists continuing to engage in a game of verbal sparring and lyrical one-upmanship.
    

## 3. Advanced RAG (Routing, Sub-Questions)

### Build a Router that can choose whether to do vector search or summarization


```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

vector_tool = QueryEngineTool(
    index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts.",
    ),
)

summary_tool = QueryEngineTool(
    index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document.",
    ),
)
```


```python
from llama_index.core.query_engine import RouterQueryEngine

query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool], select_multi=False, verbose=True
)

response = query_engine.query(
    "Tell me about the song meet the grahams - why is it significant"
)
```

    [1;3;38;5;200mSelecting query engine 0: The song 'Meet the Grahams' might contain specific facts or information about the band, making it useful for searching for those specific details..
    [0m


```python
print(response)
```

    "Meet the Grahams" artwork is a crucial part of a larger strategy by Kendrick Lamar to address Drake's family matters in a diss track. The artwork shows a pair of Maybach gloves, a shirt, receipts, and prescription bottles, including one for Ozempic prescribed to Drake. This song is significant because it serves as the full picture that Kendrick teased earlier on "6.16 in LA" and addresses all members of Drake's family, including his son Adonis, mother Sandi, father Dennis, and an alleged 11-year-old daughter. The song takes it to the point of no return, with Kendrick musing that he wishes Dennis Graham wore a condom the night Drake was conceived and telling both Drake's parents that they raised a man whose house is due to be raided any day now on Harvey Weinstein-level allegations.
    

### Break Complex Questions down into Sub-Questions

Our Sub-Question Query Engine breaks complex questions down into sub-questions.



```python
drake_index = VectorStoreIndex.from_documents(docs_drake)
drake_query_engine = drake_index.as_query_engine(similarity_top_k=3)

kendrick_index = VectorStoreIndex.from_documents(docs_kendrick)
kendrick_query_engine = kendrick_index.as_query_engine(similarity_top_k=3)
```


```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

drake_tool = QueryEngineTool(
    drake_index.as_query_engine(),
    metadata=ToolMetadata(
        name="drake_search",
        description="Useful for searching over Drake's life.",
    ),
)

kendrick_tool = QueryEngineTool(
    kendrick_index.as_query_engine(),
    metadata=ToolMetadata(
        name="kendrick_summary",
        description="Useful for searching over Kendrick's life.",
    ),
)
```


```python
from llama_index.core.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(
    [drake_tool, kendrick_tool],
    llm=llm_replicate,  # llama3-70b
    verbose=True,
)

response = query_engine.query("Which albums did Drake release in his career?")

print(response)
```

    Generated 1 sub questions.
    [1;3;38;2;237;90;200m[drake_search] Q: What are the albums released by Drake
    [0m[1;3;38;2;237;90;200m[drake_search] A: Based on the provided context information, the albums released by Drake are:
    
    1. Take Care (album)
    2. Nothing Was the Same
    3. If You're Reading This It's Too Late (rumored to be a mixtape or album)
    4. Certified Lover Boy
    5. Honestly, Nevermind
    [0mBased on the provided context information, the albums released by Drake are:
    
    1. Take Care (album)
    2. Nothing Was the Same
    3. If You're Reading This It's Too Late (rumored to be a mixtape or album)
    4. Certified Lover Boy
    5. Honestly, Nevermind
    

## 4. Text-to-SQL 

Here, we download and use a sample SQLite database with 11 tables, with various info about music, playlists, and customers. We will limit to a select few tables for this test.


```python
!wget "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip" -O "./data/chinook.zip"
!unzip "./data/chinook.zip"
```

    --2024-05-10 23:40:37--  https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip
    Resolving www.sqlitetutorial.net (www.sqlitetutorial.net)... 2606:4700:3037::6815:1e8d, 2606:4700:3037::ac43:acfa, 104.21.30.141, ...
    Connecting to www.sqlitetutorial.net (www.sqlitetutorial.net)|2606:4700:3037::6815:1e8d|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 305596 (298K) [application/zip]
    Saving to: ‘./data/chinook.zip’
    
    ./data/chinook.zip  100%[===================>] 298.43K  --.-KB/s    in 0.02s   
    
    2024-05-10 23:40:37 (13.9 MB/s) - ‘./data/chinook.zip’ saved [305596/305596]
    
    

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    

    Archive:  ./data/chinook.zip
      inflating: chinook.db              
    

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    


```python
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)

engine = create_engine("sqlite:///chinook.db")
```


```python
from llama_index.core import SQLDatabase

sql_database = SQLDatabase(engine)
```


```python
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
    llm=llm_replicate,
)
```


```python
response = query_engine.query("What are some albums?")

print(response)
```

    Here are 10 album titles with their corresponding artists:
    
    1. "For Those About To Rock We Salute You" by Artist 1
    2. "Balls to the Wall" by Artist 2
    3. "Restless and Wild" by Artist 2
    4. "Let There Be Rock" by Artist 1
    5. "Big Ones" by Artist 3
    6. "Jagged Little Pill" by Artist 4
    7. "Facelift" by Artist 5
    8. "Warner 25 Anos" by Artist 6
    9. "Plays Metallica By Four Cellos" by Artist 7
    10. "Audioslave" by Artist 8
    


```python
response = query_engine.query("What are some artists? Limit it to 5.")

print(response)
```

    Here are 5 artists: AC/DC, Accept, Aerosmith, Alanis Morissette, and Alice In Chains.
    

This last query should be a more complex join


```python
response = query_engine.query(
    "What are some tracks from the artist AC/DC? Limit it to 3"
)

print(response)
```

    Here are three tracks from the legendary Australian rock band AC/DC: "For Those About To Rock (We Salute You)", "Put The Finger On You", and "Let's Get It Up".
    


```python
print(response.metadata["sql_query"])
```

    SELECT tracks.Name FROM tracks JOIN albums ON tracks.AlbumId = albums.AlbumId JOIN artists ON albums.ArtistId = artists.ArtistId WHERE artists.Name = 'AC/DC' LIMIT 3;
    

## 5. Structured Data Extraction

An important use case for function calling is extracting structured objects. LlamaIndex provides an intuitive interface for this through `structured_predict` - simply define the target Pydantic class (can be nested), and given a prompt, we extract out the desired object.

**NOTE**: Since there's no native function calling support with Llama3 / Ollama, the structured extraction is performed by prompting the LLM + output parsing.


```python
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel


class Restaurant(BaseModel):
    """A restaurant with name, city, and cuisine."""

    name: str
    city: str
    cuisine: str


llm = Ollama(model="llama3")
prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)
```


```python
restaurant_obj = llm.structured_predict(
    Restaurant, prompt_tmpl, city_name="Miami"
)
print(restaurant_obj)
```

    name='Tropical Bites' city='Miami' cuisine='Caribbean'
    

## 6. Adding Chat History to RAG (Chat Engine)

In this section we create a stateful chatbot from a RAG pipeline, with our chat engine abstraction.

Unlike a stateless query engine, the chat engine maintains conversation history (through a memory module like buffer memory). It performs retrieval given a condensed question, and feeds the condensed question + context + chat history into the final LLM prompt.

Related resource: https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context/


```python
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    index.as_retriever(),
    memory=memory,
    llm=llm,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about the Kendrick and Drake beef."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=True,
)
```


```python
response = chat_engine.chat(
    "Tell me about the songs Drake released in the beef."
)
print(str(response))
```


```python
response = chat_engine.chat("What about Kendrick?")
print(str(response))
```

    Kendrick Lamar's contributions to the beef!
    
    According to the article, Kendrick released several diss tracks in response to Drake's initial shots. One notable track is "Not Like Us", which directly addresses Drake and his perceived shortcomings.
    
    However, the article highlights that Kendrick's most significant response was his album "Mr. Morale & The Big Steppers", which features several tracks that can be seen as indirect disses towards Drake.
    
    The article also mentions that Kendrick's family has been a target of Drake's attacks, with Drake referencing Kendrick's estranged relationship with his partner Whitney and their two kids (one of whom is allegedly fathered by Dave Free).
    
    It's worth noting that Kendrick didn't directly respond to Drake's THP6 track. Instead, he focused on his own music and let the lyrics speak for themselves.
    
    Overall, Kendrick's approach was more subtle yet still packed a punch, showcasing his storytelling ability and lyrical prowess.
    
    Would you like me to elaborate on any specific tracks or moments from the beef?
    

## 7. Agents

Here we build agents with Llama 3. We perform RAG over simple functions as well as the documents above.

### Agents And Tools


```python
import json
from typing import Sequence, List

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent

import nest_asyncio

nest_asyncio.apply()
```

### Define Tools


```python
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> int:
    """Divides two integers and returns the result integer"""
    return a / b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)
```

### ReAct Agent


```python
agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, subtract_tool, divide_tool],
    llm=llm_replicate,
    verbose=True,
)
```

### Querying


```python
response = agent.chat("What is (121 + 2) * 5?")
print(str(response))
```

    [1;3;38;5;200mThought: The current language of the user is: English. I need to use a tool to help me answer the question.
    Action: add
    Action Input: {'a': 121, 'b': 2}
    [0m[1;3;34mObservation: 123
    [0m[1;3;38;5;200mThought: I have the result of the addition, now I need to multiply it by 5.
    Action: multiply
    Action Input: {'a': 123, 'b': 5}
    [0m[1;3;34mObservation: 615
    [0m[1;3;38;5;200mThought: I can answer without using any more tools. I'll use the user's language to answer
    Answer: 615
    [0m615
    

### ReAct Agent With RAG QueryEngine Tools


```python
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
```

### Create ReAct Agent using RAG QueryEngine Tools


```python
drake_tool = QueryEngineTool(
    drake_index.as_query_engine(),
    metadata=ToolMetadata(
        name="drake_search",
        description="Useful for searching over Drake's life.",
    ),
)

kendrick_tool = QueryEngineTool(
    kendrick_index.as_query_engine(),
    metadata=ToolMetadata(
        name="kendrick_search",
        description="Useful for searching over Kendrick's life.",
    ),
)

query_engine_tools = [drake_tool, kendrick_tool]
```


```python
agent = ReActAgent.from_tools(
    query_engine_tools,  ## TODO: define query tools
    llm=llm_replicate,
    verbose=True,
)
```

### Querying


```python
response = agent.chat("Tell me about how Kendrick and Drake grew up")
print(str(response))
```

    [1;3;38;5;200mThought: The current language of the user is: English. I need to use a tool to help me answer the question.
    Action: kendrick_search
    Action Input: {'input': "Kendrick Lamar's childhood"}
    [0m[1;3;34mObservation: Kendrick Lamar was born on June 17, 1987, in Compton, California. He is the first child of Kenneth "Kenny" Duckworth, a former gang hustler who previously worked at KFC, and Paula Oliver, a hairdresser who previously worked at McDonald's. Both of his parents are African Americans from the South Side of Chicago, and they relocated to Compton in 1984 due to his father's affiliation with the Gangster Disciples. Lamar was named after singer-songwriter Eddie Kendricks of the Temptations. He was an only child until the age of seven and was described as a loner by his mother.
    [0m[1;3;38;5;200mThought: I have information about Kendrick's childhood, but I need to know more about Drake's upbringing to answer the question.
    Action: drake_search
    Action Input: {'input': "Drake's childhood"}
    [0m[1;3;34mObservation: Drake was raised in two neighborhoods. He lived on Weston Road in Toronto's working-class west end until grade six and attended Weston Memorial Junior Public School until grade four. He moved to one of the city's affluent neighbourhoods, Forest Hill, in 2000. Drake appeared in a comedic sketch which aired during the 1997 NHL Awards, featuring Martin Brodeur and Ron Hextall. At age 10, he attended Forest Hill Collegiate Institute for high school.
    [0m[1;3;34mObservation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    [0m[1;3;38;5;200mThought: I apologize for the mistake. I need to use a tool to help me answer the question.
    Action: drake_search
    Action Input: {'input': "Drake's childhood"}
    [0m[1;3;34mObservation: Drake was raised in two neighborhoods. He lived on Weston Road in Toronto's working-class west end until grade six and attended Weston Memorial Junior Public School until grade four. He played minor hockey with the Weston Red Wings, reaching the Upper Canada College hockey camp before leaving due to a vicious cross-check to his neck during a game. At age 10, Drake appeared in a comedic sketch which aired during the 1997 NHL Awards.
    [0m[1;3;38;5;200mThought: I have information about both Kendrick and Drake's childhood, so I can answer the question without using any more tools.
    Answer: Kendrick Lamar grew up in Compton, California, as the child of a former gang hustler and a hairdresser, while Drake was raised in two neighborhoods in Toronto, Ontario, Canada, and had a brief experience in minor hockey before pursuing a career in entertainment.
    [0mKendrick Lamar grew up in Compton, California, as the child of a former gang hustler and a hairdresser, while Drake was raised in two neighborhoods in Toronto, Ontario, Canada, and had a brief experience in minor hockey before pursuing a career in entertainment.
    
