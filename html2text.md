# HTML to text

>[html2text](https://github.com/Alir3z4/html2text/) is a Python package that converts a page of `HTML` into clean, easy-to-read plain `ASCII text`. 

The ASCII also happens to be a valid `Markdown` (a text-to-HTML format).


```python
%pip install --upgrade --quiet html2text
```


```python
from langchain_community.document_loaders import AsyncHtmlLoader

urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()
```

    USER_AGENT environment variable not set, consider setting it to identify your requests.
    Fetching pages: 100%|##########| 2/2 [00:00<00:00, 14.74it/s]
    


```python
from langchain_community.document_transformers import Html2TextTransformer

urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

print(docs_transformed[0].page_content[1000:2000])

print(docs_transformed[1].page_content[1000:2000])
```

    ## Fantasy
    
      * Football
    
      * Baseball
    
      * Basketball
    
      * Hockey
    
    ## ESPN Sites
    
      * ESPN Deportes
    
      * Andscape
    
      * espnW
    
      * ESPNFC
    
      * X Games
    
      * SEC Network
    
    ## ESPN Apps
    
      * ESPN
    
      * ESPN Fantasy
    
      * Tournament Challenge
    
    ## Follow ESPN
    
      * Facebook
    
      * X/Twitter
    
      * Instagram
    
      * Snapchat
    
      * TikTok
    
      * YouTube
    
    ## Fresh updates to our NBA mock draft: Everything we're hearing hours before
    Round 1
    
    With hours until Round 1 begins (8 p.m. ET on ESPN and ABC), ESPN draft
    insiders Jonathan Givony and Jeremy Woo have new intel on lottery picks and
    more.
    
    2hJonathan Givony and Jeremy Woo
    
    Illustration by ESPN
    
    ## From No. 1 to 100: Ranking the 2024 NBA draft prospects
    
    Who's No. 1? Where do the Kentucky, Duke and UConn players rank? Here's our
    final Top 100 Big Board.
    
    6hJonathan Givony and Jeremy Woo
    
      * Full draft order: All 58 picks over two rounds
      * Trade tracker: Details for all deals
    
      * Betting buzz: Lakers favorites to draft Bronny
      * Use our NBA draft simu
    ent system, LLM functions as the agent's brain,
    complemented by several key components:
    
      * **Planning**
        * Subgoal and decomposition: The agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.
        * Reflection and refinement: The agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps, thereby improving the quality of final results.
      * **Memory**
        * Short-term memory: I would consider all the in-context learning (See Prompt Engineering) as utilizing short-term memory of the model to learn.
        * Long-term memory: This provides the agent with the capability to retain and recall (infinite) information over extended periods, often by leveraging an external vector store and fast retrieval.
      * **Tool use**
        * The agent learns to call external APIs for extra information that is missing from the model weights (often hard to change after pre-training), including 
    
