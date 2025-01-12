<a target="_blank" href="https://colab.research.google.com/drive/1Vy4tYxP_rlbkAKi4pGpaDRV7hnSQeG2d?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# RAG example for LangChain Agent + Fireworks + MongoDB + Nomic model

## Introduction
We just went through the Fireworks tutorial for MongoDB and nomic embedding, with all the data indexed. Now we are going to dig into using how to use an agent framework to drive the interaction, since you may not want to just use embedding to do the work sometimes

## Setting Up Your Environment
Before we dive into the code, make sure to set up your environment. This involves installing necessary packages like pymongo and openai. Run the following command in your notebook to install these packages:


```python
!pip install pymongo openai tqdm langchain openai langchain_openai langchainhub numexpr
```

    Requirement already satisfied: pymongo in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (4.6.1)
    Requirement already satisfied: openai in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (1.12.0)
    Requirement already satisfied: tqdm in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (4.66.1)
    Requirement already satisfied: langchain in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (0.1.8)
    Requirement already satisfied: langchain_openai in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (0.0.6)
    Requirement already satisfied: langchainhub in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (0.1.14)
    Requirement already satisfied: numexpr in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (2.9.0)
    Requirement already satisfied: dnspython<3.0.0,>=1.16.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from pymongo) (2.5.0)
    Requirement already satisfied: anyio<5,>=3.5.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from openai) (4.2.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from openai) (1.9.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from openai) (0.25.2)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from openai) (2.5.3)
    Requirement already satisfied: sniffio in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from openai) (1.3.0)
    Requirement already satisfied: typing-extensions<5,>=4.7 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from openai) (4.9.0)
    Requirement already satisfied: PyYAML>=5.3 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (6.0.1)
    Requirement already satisfied: SQLAlchemy<3,>=1.4 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (2.0.25)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (3.9.1)
    Requirement already satisfied: async-timeout<5.0.0,>=4.0.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (4.0.3)
    Requirement already satisfied: dataclasses-json<0.7,>=0.5.7 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (0.6.3)
    Requirement already satisfied: jsonpatch<2.0,>=1.33 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (1.33)
    Requirement already satisfied: langchain-community<0.1,>=0.0.21 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (0.0.21)
    Requirement already satisfied: langchain-core<0.2,>=0.1.24 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (0.1.25)
    Requirement already satisfied: langsmith<0.2.0,>=0.1.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (0.1.5)
    Requirement already satisfied: numpy<2,>=1 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (1.26.3)
    Requirement already satisfied: requests<3,>=2 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (2.31.0)
    Requirement already satisfied: tenacity<9.0.0,>=8.1.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain) (8.2.3)
    Requirement already satisfied: tiktoken<1,>=0.5.2 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain_openai) (0.5.2)
    Requirement already satisfied: types-requests<3.0.0.0,>=2.31.0.2 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchainhub) (2.31.0.20240218)
    Requirement already satisfied: attrs>=17.3.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (23.2.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.0.4)
    Requirement already satisfied: yarl<2.0,>=1.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.9.4)
    Requirement already satisfied: frozenlist>=1.1.1 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.4.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.1)
    Requirement already satisfied: idna>=2.8 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from anyio<5,>=3.5.0->openai) (3.6)
    Requirement already satisfied: exceptiongroup>=1.0.2 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from anyio<5,>=3.5.0->openai) (1.2.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain) (3.20.2)
    Requirement already satisfied: typing-inspect<1,>=0.4.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain) (0.9.0)
    Requirement already satisfied: certifi in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from httpx<1,>=0.23.0->openai) (2023.11.17)
    Requirement already satisfied: httpcore==1.* in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from httpx<1,>=0.23.0->openai) (1.0.2)
    Requirement already satisfied: h11<0.15,>=0.13 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
    Requirement already satisfied: jsonpointer>=1.9 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from jsonpatch<2.0,>=1.33->langchain) (2.4)
    Requirement already satisfied: packaging<24.0,>=23.2 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from langchain-core<0.2,>=0.1.24->langchain) (23.2)
    Requirement already satisfied: annotated-types>=0.4.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from pydantic<3,>=1.9.0->openai) (0.6.0)
    Requirement already satisfied: pydantic-core==2.14.6 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from pydantic<3,>=1.9.0->openai) (2.14.6)
    Requirement already satisfied: charset-normalizer<4,>=2 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from requests<3,>=2->langchain) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from requests<3,>=2->langchain) (2.1.0)
    Requirement already satisfied: greenlet!=0.4.17 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from SQLAlchemy<3,>=1.4->langchain) (3.0.3)
    Requirement already satisfied: regex>=2022.1.18 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from tiktoken<1,>=0.5.2->langchain_openai) (2023.12.25)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /home/bchen/cookbook/.venv/lib/python3.10/site-packages (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain) (1.0.0)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.3.2[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    

## Initializing Fireworks and MongoDB Clients
To interact with Fireworks and MongoDB, we need to initialize their respective clients. Replace "YOUR FIREWORKS API KEY" and "YOUR MONGO URL" with your actual credentials.


```python
import openai
import pymongo

mongo_url = input()
client = pymongo.MongoClient(mongo_url)
```


```python
fw_api_key = input()
fw_client = openai.OpenAI(
  api_key=fw_api_key,
  base_url="https://api.fireworks.ai/inference/v1"
)
```

# Picking an agent framework you prefer
There are many agent frameworks on the market for you to choose from. We will use LangChain as the tool to drive MongoDB integration here. There are a few steps we need to take here
- A tool that fetches the MongoDB schema, so our function calling model can know what field to filter on
- A tool that fetches embeddings given the user query
- A tool that executes MongoDB queries, with both the filter and the vector embeddings

We will first begin with basic LangChain tool setup


```python
import os

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Optional, Type
from langchain.globals import set_debug
from langchain.globals import set_verbose


llm = ChatOpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=fw_api_key,
    model="accounts/fireworks/models/firefunction-v1",
    temperature=0.0,
    max_tokens=256,
)

```


```python
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class FetchSchemaInput(BaseModel):
    db: str = Field(description="name of the database")
    collection: str = Field(description="name of the collection")

class FetchSchemaTool(BaseTool):
    name: str = "FetchSchemaTool"
    description: str = "Get the schema for the corresponding db and collection"
    args_schema: Type[BaseModel] = FetchSchemaInput

    def _run(self, db: str, collection: str) -> Dict[str, Any]:
        """Fetch the schema from the specified db and collection

        Args:
            db: name of the mongodb database
            collection: name of the mongodb collection
        """
        return {
            "schema": dict([(x, type(y)) for x, y in client[db][collection].find_one().items()]),
        }


class MongoDBAtlasSearchInput(BaseModel):
    user_query: str = Field(description="faith representation of user query to generate the embeddings from")
    db: str = Field(description="name of the database")
    collection: str = Field(description="name of the collection")
    filter: Dict[str, Any] = Field(description="mongodb filter for the given query")
    relevant_fields: List[str] = Field(description="relevant fields from the schema that needed to be pulled to respond to user query")

class MongoDBAtlasSearchTool(BaseTool):
    name: str = "MongoDBAtlasSearchTool"
    description: str = "Look up into MongoDB Altas based on filters and embeddings"
    args_schema: Type[BaseModel] = MongoDBAtlasSearchInput

    def _run(self, user_query: str, db: str, collection: str, filter: Dict[str, Any], relevant_fields: List[str]) -> Dict[str, Any]:
        """Fetch the schema from the specified db and collection
        """
        embedding = [x.embedding for x in 
            fw_client.embeddings.create(
            input=[user_query],
            model="nomic-ai/nomic-embed-text-v1.5"
        ).data][0]
        return [[(x, y) for x, y in doc.items() if x in relevant_fields] for doc in client[db][collection].aggregate([{
            # vector search fields are hard coded right now to make things cleaner, but we can also inject into
            # the system prompt and let the model handle it
            "$vectorSearch": {
                "index": "movie_index",
                "path": "embedding_2k_movies_fw_nomic_1_5",
                "queryVector": embedding,
                "numCandidates": 200,
                "limit": 20,
                "filter": filter
            }
        }])]

tools = [
  FetchSchemaTool(),
  MongoDBAtlasSearchTool(),
]

system_prompt = """
You are an expert MongoDB user that turns user query into model requests.
Whenever you are faced with a user query, you should first fetch the mongoDB schema for database "sample_mflix" and collection "movies".
Then, you respond to the original user query by looking into MongoDB Atlas
- with filter that best match the user query given the schema. The filters will be constructed from the schema you fetched earlier.
- as well as the embeddings.
After getting the result from MongoDB Atlas, only then respond with your movies recommendations given the fetch results. Do not prematurely respond with any movies recommendations.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm, tools, prompt)

agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
```


```python
print(agent.invoke({"input": "I like Spiderman, any similar movies you can recommend after the year 2000?"}))
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `FetchSchemaTool` with `{'db': 'sample_mflix', 'collection': 'movies'}`
    
    
    [0m[36;1m[1;3m{'schema': {'_id': <class 'bson.objectid.ObjectId'>, 'plot': <class 'str'>, 'genres': <class 'list'>, 'runtime': <class 'int'>, 'cast': <class 'list'>, 'num_mflix_comments': <class 'int'>, 'poster': <class 'str'>, 'title': <class 'str'>, 'fullplot': <class 'str'>, 'languages': <class 'list'>, 'released': <class 'datetime.datetime'>, 'directors': <class 'list'>, 'writers': <class 'list'>, 'awards': <class 'dict'>, 'lastupdated': <class 'str'>, 'year': <class 'int'>, 'imdb': <class 'dict'>, 'countries': <class 'list'>, 'type': <class 'str'>, 'tomatoes': <class 'dict'>, 'embedding_2k_movies_fw_e5_mistral': <class 'list'>}}[0m[32;1m[1;3m
    Invoking: `MongoDBAtlasSearchTool` with `{'user_query': 'Spiderman', 'db': 'sample_mflix', 'collection': 'movies', 'filter': {'year': {'$gte': 2000}}, 'relevant_fields': ['title', 'plot', 'genres', 'poster', 'imdb.rating', 'tomatoes.viewer.rating']}`
    
    
    [0m[33;1m[1;3m[[('plot', 'When bitten by a genetically modified spider, a nerdy, shy, and awkward high school student gains spider-like abilities that he eventually must use to fight evil as a superhero after tragedy befalls his family.'), ('genres', ['Action', 'Adventure']), ('poster', 'https://m.media-amazon.com/images/M/MV5BZDEyN2NhMjgtMjdhNi00MmNlLWE5YTgtZGE4MzNjMTRlMGEwXkEyXkFqcGdeQXVyNDUyOTg3Njg@._V1_SY1000_SX677_AL_.jpg'), ('title', 'Spider-Man')], [('plot', 'Two mutants come to a private academy for their kind whose resident superhero team must oppose a terrorist organization with similar powers.'), ('genres', ['Action', 'Adventure', 'Sci-Fi']), ('title', 'X-Men'), ('poster', 'https://m.media-amazon.com/images/M/MV5BZmIyMDk5NGYtYjQ5NS00ZWQxLTg2YzQtZDk1ZmM4ZDBlN2E3XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SY1000_SX677_AL_.jpg')], [('plot', "A congressman's daughter under Secret Service protection is kidnapped from a private school by an insider who calls Det. Alex Cross, sucking him into the case even though he's recovering from the loss of his partner."), ('genres', ['Thriller']), ('title', 'Along Came a Spider'), ('poster', 'https://m.media-amazon.com/images/M/MV5BOTVlY2VhMWEtYmRlOC00YWVhLWEzMDktZWJlYzNiMWJmZTIwXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SY1000_SX677_AL_.jpg')], [('plot', 'A group of astronauts gain superpowers after a cosmic radiation exposure and must use them to oppose the plans of their enemy, Doctor Victor Von Doom.'), ('genres', ['Action', 'Adventure', 'Sci-Fi']), ('title', 'Fantastic Four'), ('poster', 'https://m.media-amazon.com/images/M/MV5BNWU1ZjFjMTctYjA5ZC00YTBkLTkzZjUtZWEyMjgxY2MxYWM4XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SY1000_SX677_AL_.jpg')], [('plot', 'Scientists discover how to make people invisible, but their test subject becomes an insane killer who stalks them.'), ('genres', ['Sci-Fi', 'Thriller']), ('title', 'Hollow Man'), ('poster', 'https://m.media-amazon.com/images/M/MV5BZTM1MjM2Y2QtOTMyOS00ZTE4LWFiZTEtMjBjMjg3NmE0ZjMxXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SY1000_SX677_AL_.jpg')], [('plot', 'A young man learns that he has to find a hidden Earth ship before an enemy alien species does in order to secure the survival of humanity.'), ('genres', ['Animation', 'Action', 'Adventure']), ('poster', 'https://m.media-amazon.com/images/M/MV5BYTVkYTAyZDMtYzNmYi00OTkyLTg3M2ItNmE0MDZkNmY3Y2I3XkEyXkFqcGdeQXVyNjU2MDUzMjY@._V1_SY1000_SX677_AL_.jpg'), ('title', 'Titan A.E.')], [('plot', 'A secret agent is sent to Sydney, to find and destroy a genetically modified disease called "Chimera".'), ('genres', ['Action', 'Adventure', 'Thriller']), ('poster', 'https://m.media-amazon.com/images/M/MV5BN2RkYWVkZDQtNTMxMi00NWQ4LWE2ODctNmQzOWM2NjQzYzdlXkEyXkFqcGdeQXVyMjUzOTY1NTc@._V1_SY1000_SX677_AL_.jpg'), ('title', 'Mission: Impossible II')], [('plot', "Mysterious reclusive magician the Great Zamboni and his pet jaguar Shadow reside in the Spooky House, an old mansion that's rigged with magic tricks and hidden chambers. Young orphan Max ..."), ('genres', ['Family']), ('poster', 'https://m.media-amazon.com/images/M/MV5BMjAzMjM3NDgyN15BMl5BanBnXkFtZTcwODkzNDQyMQ@@._V1_SY1000_SX677_AL_.jpg'), ('title', 'Spooky House')], [('plot', 'Three women, detectives with a mysterious boss, retrieve stolen voice-ID software, using martial arts, tech skills, and sex appeal.'), ('genres', ['Action', 'Adventure', 'Comedy']), ('title', "Charlie's Angels"), ('poster', 'https://m.media-amazon.com/images/M/MV5BMGI1NTM5MTItNDcxMS00YTIwLTg1NjUtNzE5ZGE5NDkxMmU4XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SY1000_SX677_AL_.jpg')], [('plot', "When a Roman general is betrayed and his family murdered by an emperor's corrupt son, he comes to Rome as a gladiator to seek revenge."), ('genres', ['Action', 'Drama']), ('title', 'Gladiator'), ('poster', 'https://m.media-amazon.com/images/M/MV5BMDliMmNhNDEtODUyOS00MjNlLTgxODEtN2U3NzIxMGVkZTA1L2ltYWdlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SY1000_SX677_AL_.jpg')], [('plot', 'A special military unit fights a powerful, out-of-control supercomputer and hundreds of scientists who have mutated into flesh-eating creatures after a laboratory accident.'), ('genres', ['Action', 'Fantasy', 'Horror']), ('title', 'Resident Evil'), ('poster', 'https://m.media-amazon.com/images/M/MV5BN2Y2MTljNjMtMDRlNi00ZWNhLThmMWItYTlmZjYyZDk4NzYxXkEyXkFqcGdeQXVyNjQ2MjQ5NzM@._V1_SY1000_SX677_AL_.jpg')], [('plot', 'Agent J needs help so he is sent to find Agent K and restore his memory.'), ('genres', ['Action', 'Adventure', 'Comedy']), ('title', 'Men in Black II'), ('poster', 'https://m.media-amazon.com/images/M/MV5BMTMxNDA0NTMxMV5BMl5BanBnXkFtZTYwMDE2NzY2._V1_SY1000_SX677_AL_.jpg')], [('plot', 'In the future, a man struggles to keep his lunar nightclub out of the hands of the mafia.'), ('genres', ['Comedy', 'Action', 'Sci-Fi']), ('title', 'The Adventures of Pluto Nash'), ('poster', 'https://m.media-amazon.com/images/M/MV5BMTQ4NTMwODQwNV5BMl5BanBnXkFtZTYwNTg0Nzk2._V1_SY1000_SX677_AL_.jpg')], [('plot', "An unemployed single mother becomes a legal assistant and almost single-handedly brings down a California power company accused of polluting a city's water supply."), ('genres', ['Biography', 'Drama']), ('title', 'Erin Brockovich'), ('poster', 'https://m.media-amazon.com/images/M/MV5BYTA1NWRkNTktNDMxNS00NjE4LWEzMDAtNzA3YzlhYzRhNDA4L2ltYWdlXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SY1000_SX677_AL_.jpg')], [('plot', 'A cybernetic warrior from a post-apocalyptic future travels back in time to protect a 19-year old drifter and his future wife from a most advanced robotic assassin and to ensure they both survive a nuclear attack.'), ('genres', ['Action', 'Sci-Fi']), ('poster', 'https://m.media-amazon.com/images/M/MV5BMTk5NzM1ODgyN15BMl5BanBnXkFtZTcwMzA5MjAzMw@@._V1_SY1000_SX677_AL_.jpg'), ('title', 'Terminator 3: Rise of the Machines')], [('plot', 'In order to power the city, monsters have to scare children so that they scream. However, the children are toxic to the monsters, and after a child gets through, two monsters realize things may not be what they think.'), ('genres', ['Animation', 'Adventure', 'Comedy']), ('title', 'Monsters, Inc.'), ('poster', 'https://m.media-amazon.com/images/M/MV5BMTY1NTI0ODUyOF5BMl5BanBnXkFtZTgwNTEyNjQ0MDE@._V1_SY1000_SX677_AL_.jpg')], [('plot', 'In a coma, a cartoonist finds himself trapped within his own underground creation and must find a way to get back, while racing against his popular but treacherous character, Monkeybone.'), ('genres', ['Animation', 'Comedy', 'Fantasy']), ('title', 'Monkeybone'), ('poster', 'https://m.media-amazon.com/images/M/MV5BZjZiODk2MDMtNjI2OS00MTEzLWFlY2UtMzNhODE5MzNlM2Y0L2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyNTAyODkwOQ@@._V1_SY1000_SX677_AL_.jpg')], [('plot', 'A small New Mexican village discovers a severed hand that is considered a miracle of God, when it actually belongs to a murdered spouse with a husband in search of it.'), ('genres', ['Comedy', 'Fantasy']), ('poster', 'https://m.media-amazon.com/images/M/MV5BYzhmMzEyYjUtYTBiNi00ODgyLWFhY2YtZTdlNGQxNDAxNjExXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SY1000_SX677_AL_.jpg'), ('title', 'Picking Up the Pieces')], [('plot', 'A demon, raised from infancy after being conjured by and rescued from the Nazis, grows up to become a defender against the forces of darkness.'), ('genres', ['Action', 'Fantasy', 'Sci-Fi']), ('title', 'Hellboy'), ('poster', 'https://m.media-amazon.com/images/M/MV5BNTc1ZWY0ZTEtZTVmNi00MTg0LTg4NmQtZTI4OWNiMmQ0MWZkXkEyXkFqcGdeQXVyNTIzOTk5ODM@._V1_SY1000_SX677_AL_.jpg')], [('plot', 'Nick is a struggling dentist in Canada. A new neighbor moves in, and he discovers that it is Jimmy "The Tulip" Teduski. His wife convinces him to go to Chicago and inform the mob boss who wants Jimmy dead.'), ('genres', ['Comedy', 'Crime']), ('poster', 'https://m.media-amazon.com/images/M/MV5BMjA4Nzk4NzU4MF5BMl5BanBnXkFtZTYwNzU0MDc3._V1_SY1000_SX677_AL_.jpg'), ('title', 'The Whole Nine Yards')]][0m[32;1m[1;3mBased on your preferences, I have found some movies that you might enjoy. Here are a few recommendations:
    
    1. Spider-Man (2002) - A nerdy, shy, and awkward high school student gains spider-like abilities after being bitten by a genetically modified spider. He must use his powers to fight evil as a superhero after tragedy befalls his family.
    
    2. X-Men (2000) - Two mutants come to a private academy for their kind whose resident superhero team must oppose a terrorist organization with similar powers.
    
    3. Fantastic Four (2005) - Scientists gain superpowers after a cosmic radiation exposure and must use them to oppose the plans of their enemy, Doctor Victor Von Doom.
    
    4. Hollow Man (2000) - Scientists discover how to make people invisible, but their test subject becomes an insane killer who stalks them.
    
    5. Titan A.E. (2000) - A young man learns that he has to find a hidden Earth ship before an enemy alien species does in order to secure the survival of humanity.
    
    [0m
    
    [1m> Finished chain.[0m
    {'input': 'I like Spiderman, any similar movies you can recommend after the year 2000?', 'output': 'Based on your preferences, I have found some movies that you might enjoy. Here are a few recommendations:\n\n1. Spider-Man (2002) - A nerdy, shy, and awkward high school student gains spider-like abilities after being bitten by a genetically modified spider. He must use his powers to fight evil as a superhero after tragedy befalls his family.\n\n2. X-Men (2000) - Two mutants come to a private academy for their kind whose resident superhero team must oppose a terrorist organization with similar powers.\n\n3. Fantastic Four (2005) - Scientists gain superpowers after a cosmic radiation exposure and must use them to oppose the plans of their enemy, Doctor Victor Von Doom.\n\n4. Hollow Man (2000) - Scientists discover how to make people invisible, but their test subject becomes an insane killer who stalks them.\n\n5. Titan A.E. (2000) - A young man learns that he has to find a hidden Earth ship before an enemy alien species does in order to secure the survival of humanity.\n\n'}
    


```python
print(agent.invoke({"input": "I like Spiderman, any similar movies you can recommend after 2010?"}))
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `FetchSchemaTool` with `{'db': 'sample_mflix', 'collection': 'movies'}`
    
    
    [0m[36;1m[1;3m{'schema': {'_id': <class 'bson.objectid.ObjectId'>, 'plot': <class 'str'>, 'genres': <class 'list'>, 'runtime': <class 'int'>, 'cast': <class 'list'>, 'num_mflix_comments': <class 'int'>, 'poster': <class 'str'>, 'title': <class 'str'>, 'fullplot': <class 'str'>, 'languages': <class 'list'>, 'released': <class 'datetime.datetime'>, 'directors': <class 'list'>, 'writers': <class 'list'>, 'awards': <class 'dict'>, 'lastupdated': <class 'str'>, 'year': <class 'int'>, 'imdb': <class 'dict'>, 'countries': <class 'list'>, 'type': <class 'str'>, 'tomatoes': <class 'dict'>, 'embedding_2k_movies_fw_e5_mistral': <class 'list'>}}[0m[32;1m[1;3m
    Invoking: `MongoDBAtlasSearchTool` with `{'user_query': 'Spiderman', 'db': 'sample_mflix', 'collection': 'movies', 'filter': {'released': {'$gte': '2010-01-01'}}, 'relevant_fields': ['title', 'plot', 'genres', 'poster', 'imdb.rating', 'tomatoes.viewer.rating']}`
    
    
    [0m[33;1m[1;3m[][0m[32;1m[1;3mBased on your preferences, I have found some movies that are similar to Spiderman and were released after 2010. However, I couldn't find any movies that match your criteria. Please try again with a different query.[0m
    
    [1m> Finished chain.[0m
    {'input': 'I like Spiderman, any similar movies you can recommend after 2010?', 'output': "Based on your preferences, I have found some movies that are similar to Spiderman and were released after 2010. However, I couldn't find any movies that match your criteria. Please try again with a different query."}
    
