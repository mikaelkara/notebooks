# Lantern

>[Lantern](https://github.com/lanterndata/lantern) is an open-source vector similarity search for `Postgres`

It supports:
- Exact and approximate nearest neighbor search
- L2 squared distance, hamming distance, and cosine distance

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use the Postgres vector database (`Lantern`).

See the [installation instruction](https://github.com/lanterndata/lantern#-quick-install).

We want to use `OpenAIEmbeddings` so we have to get the OpenAI API Key.

# Pip install necessary package
!pip install openai
!pip install psycopg2-binary
!pip install tiktoken


```python
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
```

    OpenAI API Key: ········
    


```python
## Loading Environment Variables
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()
```




    False




```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Lantern
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
```


```python
loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
```


```python
# Lantern needs the connection string to the database.
# Example postgresql://postgres:postgres@localhost:5432/postgres
CONNECTION_STRING = getpass.getpass("DB Connection String:")

# # Alternatively, you can create it from environment variables.
# import os

# CONNECTION_STRING = Lantern.connection_string_from_db_params(
#     driver=os.environ.get("LANTERN_DRIVER", "psycopg2"),
#     host=os.environ.get("LANTERN_HOST", "localhost"),
#     port=int(os.environ.get("LANTERN_PORT", "5432")),
#     database=os.environ.get("LANTERN_DATABASE", "postgres"),
#     user=os.environ.get("LANTERN_USER", "postgres"),
#     password=os.environ.get("LANTERN_PASSWORD", "postgres"),
# )

# or you can pass it via `LANTERN_CONNECTION_STRING` env variable
```

    DB Connection String: ········
    

## Similarity Search with Cosine Distance (Default)


```python
# The Lantern Module will try to create a table with the name of the collection.
# So, make sure that the collection name is unique and the user has the permission to create a table.

COLLECTION_NAME = "state_of_the_union_test"

db = Lantern.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True,
)
```


```python
query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)
```


```python
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
```

    --------------------------------------------------------------------------------
    Score:  0.18440479
    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 
    
    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 
    
    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 
    
    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Score:  0.21727282
    A former top litigator in private practice. A former federal public defender. And from a family of public school educators and police officers. A consensus builder. Since she’s been nominated, she’s received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democrats and Republicans. 
    
    And if we are to advance liberty and justice, we need to secure the Border and fix the immigration system. 
    
    We can do both. At our border, we’ve installed new technology like cutting-edge scanners to better detect drug smuggling.  
    
    We’ve set up joint patrols with Mexico and Guatemala to catch more human traffickers.  
    
    We’re putting in place dedicated immigration judges so families fleeing persecution and violence can have their cases heard faster. 
    
    We’re securing commitments and supporting partners in South and Central America to host more refugees and secure their own borders.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Score:  0.22621095
    And for our LGBTQ+ Americans, let’s finally get the bipartisan Equality Act to my desk. The onslaught of state laws targeting transgender Americans and their families is wrong. 
    
    As I said last year, especially to our younger transgender Americans, I will always have your back as your President, so you can be yourself and reach your God-given potential. 
    
    While it often appears that we never agree, that isn’t true. I signed 80 bipartisan bills into law last year. From preventing government shutdowns to protecting Asian-Americans from still-too-common hate crimes to reforming military justice. 
    
    And soon, we’ll strengthen the Violence Against Women Act that I first wrote three decades ago. It is important for us to show the nation that we can come together and do big things. 
    
    So tonight I’m offering a Unity Agenda for the Nation. Four big things we can do together.  
    
    First, beat the opioid epidemic.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Score:  0.22654456
    Tonight, I’m announcing a crackdown on these companies overcharging American businesses and consumers. 
    
    And as Wall Street firms take over more nursing homes, quality in those homes has gone down and costs have gone up.  
    
    That ends on my watch. 
    
    Medicare is going to set higher standards for nursing homes and make sure your loved ones get the care they deserve and expect. 
    
    We’ll also cut costs and keep the economy going strong by giving workers a fair shot, provide more training and apprenticeships, hire them based on their skills not degrees. 
    
    Let’s pass the Paycheck Fairness Act and paid leave.  
    
    Raise the minimum wage to $15 an hour and extend the Child Tax Credit, so no one has to raise a family in poverty. 
    
    Let’s increase Pell Grants and increase our historic support of HBCUs, and invest in what Jill—our First Lady who teaches full-time—calls America’s best-kept secret: community colleges.
    --------------------------------------------------------------------------------
    

## Maximal Marginal Relevance Search (MMR)
Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.


```python
docs_with_score = db.max_marginal_relevance_search_with_score(query)
```


```python
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
```

    --------------------------------------------------------------------------------
    Score:  0.18440479
    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 
    
    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 
    
    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 
    
    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Score:  0.23515457
    We can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. 
    
    I recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. 
    
    They were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. 
    
    Officer Mora was 27 years old. 
    
    Officer Rivera was 22. 
    
    Both Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. 
    
    I spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves. 
    
    I’ve worked on these issues a long time. 
    
    I know what works: Investing in crime prevention and community police officers who’ll walk the beat, who’ll know the neighborhood, and who can restore trust and safety.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Score:  0.24478757
    One was stationed at bases and breathing in toxic smoke from “burn pits” that incinerated wastes of war—medical and hazard material, jet fuel, and more. 
    
    When they came home, many of the world’s fittest and best trained warriors were never the same. 
    
    Headaches. Numbness. Dizziness. 
    
    A cancer that would put them in a flag-draped coffin. 
    
    I know. 
    
    One of those soldiers was my son Major Beau Biden. 
    
    We don’t know for sure if a burn pit was the cause of his brain cancer, or the diseases of so many of our troops. 
    
    But I’m committed to finding out everything we can. 
    
    Committed to military families like Danielle Robinson from Ohio. 
    
    The widow of Sergeant First Class Heath Robinson.  
    
    He was born a soldier. Army National Guard. Combat medic in Kosovo and Iraq. 
    
    Stationed near Baghdad, just yards from burn pits the size of football fields. 
    
    Heath’s widow Danielle is here with us tonight. They loved going to Ohio State football games. He loved building Legos with their daughter.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Score:  0.25137997
    And I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. 
    
    Tonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  
    
    America will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  
    
    These steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. 
    
    But I want you to know that we are going to be okay. 
    
    When the history of this era is written Putin’s war on Ukraine will have left Russia weaker and the rest of the world stronger. 
    
    While it shouldn’t have taken something so terrible for people around the world to see what’s at stake now everyone sees it clearly.
    --------------------------------------------------------------------------------
    

## Working with vectorstore

Above, we created a vectorstore from scratch. However, often times we want to work with an existing vectorstore.
In order to do that, we can initialize it directly.


```python
store = Lantern(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)
```

### Add documents
We can add documents to the existing vectorstore.


```python
store.add_documents([Document(page_content="foo")])
```




    ['f8164598-aa28-11ee-a037-acde48001122']




```python
docs_with_score = db.similarity_search_with_score("foo")
```


```python
docs_with_score[0]
```




    (Document(page_content='foo'), -1.1920929e-07)




```python
docs_with_score[1]
```




    (Document(page_content='And let’s pass the PRO Act when a majority of workers want to form a union—they shouldn’t be stopped.  \n\nWhen we invest in our workers, when we build the economy from the bottom up and the middle out together, we can do something we haven’t done in a long time: build a better America. \n\nFor more than two years, COVID-19 has impacted every decision in our lives and the life of the nation. \n\nAnd I know you’re tired, frustrated, and exhausted. \n\nBut I also know this. \n\nBecause of the progress we’ve made, because of your resilience and the tools we have, tonight I can say  \nwe are moving forward safely, back to more normal routines.  \n\nWe’ve reached a new moment in the fight against COVID-19, with severe cases down to a level not seen since last July.  \n\nJust a few days ago, the Centers for Disease Control and Prevention—the CDC—issued new mask guidelines. \n\nUnder these new guidelines, most Americans in most of the country can now be mask free.', metadata={'source': '../../how_to/state_of_the_union.txt'}),
     0.24038416)



### Overriding a vectorstore

If you have an existing collection, you override it by doing `from_documents` and setting `pre_delete_collection` = True 
This will delete the collection before re-populating it


```python
db = Lantern.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True,
)
```


```python
docs_with_score = db.similarity_search_with_score("foo")
```


```python
docs_with_score[0]
```




    (Document(page_content='And let’s pass the PRO Act when a majority of workers want to form a union—they shouldn’t be stopped.  \n\nWhen we invest in our workers, when we build the economy from the bottom up and the middle out together, we can do something we haven’t done in a long time: build a better America. \n\nFor more than two years, COVID-19 has impacted every decision in our lives and the life of the nation. \n\nAnd I know you’re tired, frustrated, and exhausted. \n\nBut I also know this. \n\nBecause of the progress we’ve made, because of your resilience and the tools we have, tonight I can say  \nwe are moving forward safely, back to more normal routines.  \n\nWe’ve reached a new moment in the fight against COVID-19, with severe cases down to a level not seen since last July.  \n\nJust a few days ago, the Centers for Disease Control and Prevention—the CDC—issued new mask guidelines. \n\nUnder these new guidelines, most Americans in most of the country can now be mask free.', metadata={'source': '../../how_to/state_of_the_union.txt'}),
     0.2403456)



### Using a VectorStore as a Retriever


```python
retriever = store.as_retriever()
```


```python
print(retriever)
```

    tags=['Lantern', 'OpenAIEmbeddings'] vectorstore=<langchain_community.vectorstores.lantern.Lantern object at 0x11d02f9d0>
    


```python

```
