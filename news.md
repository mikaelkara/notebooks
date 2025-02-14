# News URL

This covers how to load HTML news articles from a list of URLs into a document format that we can use downstream.


```python
from langchain_community.document_loaders import NewsURLLoader
```


```python
urls = [
    "https://www.bbc.com/news/world-us-canada-66388172",
    "https://www.bbc.com/news/entertainment-arts-66384971",
]
```

Pass in urls to load them into Documents


```python
loader = NewsURLLoader(urls=urls)
data = loader.load()
print("First article: ", data[0])
print("\nSecond article: ", data[1])
```

    First article:  page_content='In testimony to the congressional committee examining the 6 January riot, Mrs Powell said she did not review all of the many claims of election fraud she made, telling them that "no reasonable person" would view her claims as fact. Neither she nor her representatives have commented.' metadata={'title': 'Donald Trump indictment: What do we know about the six co-conspirators?', 'link': 'https://www.bbc.com/news/world-us-canada-66388172', 'authors': [], 'language': 'en', 'description': 'Six people accused of helping Mr Trump undermine the election have been described by prosecutors.', 'publish_date': None}
    
    Second article:  page_content='Ms Williams added: "If there\'s anything that I can do in my power to ensure that dancers or singers or whoever decides to work with her don\'t have to go through that same experience, I\'m going to do that."' metadata={'title': "Lizzo dancers Arianna Davis and Crystal Williams: 'No one speaks out, they are scared'", 'link': 'https://www.bbc.com/news/entertainment-arts-66384971', 'authors': [], 'language': 'en', 'description': 'The US pop star is being sued for sexual harassment and fat-shaming but has yet to comment.', 'publish_date': None}
    

Use nlp=True to run nlp analysis and generate keywords + summary


```python
loader = NewsURLLoader(urls=urls, nlp=True)
data = loader.load()
print("First article: ", data[0])
print("\nSecond article: ", data[1])
```

    First article:  page_content='In testimony to the congressional committee examining the 6 January riot, Mrs Powell said she did not review all of the many claims of election fraud she made, telling them that "no reasonable person" would view her claims as fact. Neither she nor her representatives have commented.' metadata={'title': 'Donald Trump indictment: What do we know about the six co-conspirators?', 'link': 'https://www.bbc.com/news/world-us-canada-66388172', 'authors': [], 'language': 'en', 'description': 'Six people accused of helping Mr Trump undermine the election have been described by prosecutors.', 'publish_date': None, 'keywords': ['powell', 'know', 'donald', 'trump', 'review', 'indictment', 'telling', 'view', 'reasonable', 'person', 'testimony', 'coconspirators', 'riot', 'representatives', 'claims'], 'summary': 'In testimony to the congressional committee examining the 6 January riot, Mrs Powell said she did not review all of the many claims of election fraud she made, telling them that "no reasonable person" would view her claims as fact.\nNeither she nor her representatives have commented.'}
    
    Second article:  page_content='Ms Williams added: "If there\'s anything that I can do in my power to ensure that dancers or singers or whoever decides to work with her don\'t have to go through that same experience, I\'m going to do that."' metadata={'title': "Lizzo dancers Arianna Davis and Crystal Williams: 'No one speaks out, they are scared'", 'link': 'https://www.bbc.com/news/entertainment-arts-66384971', 'authors': [], 'language': 'en', 'description': 'The US pop star is being sued for sexual harassment and fat-shaming but has yet to comment.', 'publish_date': None, 'keywords': ['davis', 'lizzo', 'singers', 'experience', 'crystal', 'ensure', 'arianna', 'theres', 'williams', 'power', 'going', 'dancers', 'im', 'speaks', 'work', 'ms', 'scared'], 'summary': 'Ms Williams added: "If there\'s anything that I can do in my power to ensure that dancers or singers or whoever decides to work with her don\'t have to go through that same experience, I\'m going to do that."'}
    


```python
data[0].metadata["keywords"]
```




    ['powell',
     'know',
     'donald',
     'trump',
     'review',
     'indictment',
     'telling',
     'view',
     'reasonable',
     'person',
     'testimony',
     'coconspirators',
     'riot',
     'representatives',
     'claims']




```python
data[0].metadata["summary"]
```




    'In testimony to the congressional committee examining the 6 January riot, Mrs Powell said she did not review all of the many claims of election fraud she made, telling them that "no reasonable person" would view her claims as fact.\nNeither she nor her representatives have commented.'


