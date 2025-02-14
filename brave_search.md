# Brave Search


>[Brave Search](https://en.wikipedia.org/wiki/Brave_Search) is a search engine developed by Brave Software.
> - `Brave Search` uses its own web index. As of May 2022, it covered over 10 billion pages and was used to serve 92% 
> of search results without relying on any third-parties, with the remainder being retrieved 
> server-side from the Bing API or (on an opt-in basis) client-side from Google. According 
> to Brave, the index was kept "intentionally smaller than that of Google or Bing" in order to 
> help avoid spam and other low-quality content, with the disadvantage that "Brave Search is 
> not yet as good as Google in recovering long-tail queries."
>- `Brave Search Premium`: As of April 2023 Brave Search is an ad-free website, but it will 
> eventually switch to a new model that will include ads and premium users will get an ad-free experience.
> User data including IP addresses won't be collected from its users by default. A premium account 
> will be required for opt-in data-collection.


## Installation and Setup

To get access to the Brave Search API, you need to [create an account and get an API key](https://api.search.brave.com/app/dashboard).



```python
api_key = "..."
```


```python
from langchain_community.document_loaders import BraveSearchLoader
```

## Example


```python
loader = BraveSearchLoader(
    query="obama middle name", api_key=api_key, search_kwargs={"count": 3}
)
docs = loader.load()
len(docs)
```




    3




```python
[doc.metadata for doc in docs]
```




    [{'title': "Obama's Middle Name -- My Last Name -- is 'Hussein.' So?",
      'link': 'https://www.cair.com/cair_in_the_news/obamas-middle-name-my-last-name-is-hussein-so/'},
     {'title': "What's up with Obama's middle name? - Quora",
      'link': 'https://www.quora.com/Whats-up-with-Obamas-middle-name'},
     {'title': 'Barack Obama | Biography, Parents, Education, Presidency, Books, ...',
      'link': 'https://www.britannica.com/biography/Barack-Obama'}]




```python
[doc.page_content for doc in docs]
```




    ['I wasn’t sure whether to laugh or cry a few days back listening to radio talk show host Bill Cunningham repeatedly scream Barack <strong>Obama</strong>’<strong>s</strong> <strong>middle</strong> <strong>name</strong> — my last <strong>name</strong> — as if he had anti-Muslim Tourette’s. “Hussein,” Cunningham hissed like he was beckoning Satan when shouting the ...',
     'Answer (1 of 15): A better question would be, “What’s up with <strong>Obama</strong>’s first <strong>name</strong>?” President Barack Hussein <strong>Obama</strong>’s father’s <strong>name</strong> was Barack Hussein <strong>Obama</strong>. He was <strong>named</strong> after his father. Hussein, <strong>Obama</strong>’<strong>s</strong> <strong>middle</strong> <strong>name</strong>, is a very common Arabic <strong>name</strong>, meaning &quot;good,&quot; &quot;handsome,&quot; or ...',
     'Barack <strong>Obama</strong>, in full Barack Hussein <strong>Obama</strong> II, (born August 4, 1961, Honolulu, Hawaii, U.S.), 44th president of the United States (2009–17) and the first African American to hold the office. Before winning the presidency, <strong>Obama</strong> represented Illinois in the U.S.']




```python

```
