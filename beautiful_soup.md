# Beautiful Soup

>[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) is a Python package for parsing 
> HTML and XML documents (including having malformed markup, i.e. non-closed tags, so named after tag soup). 
> It creates a parse tree for parsed pages that can be used to extract data from HTML,[3] which 
> is useful for web scraping.

`Beautiful Soup` offers fine-grained control over HTML content, enabling specific tag extraction, removal, and content cleaning. 

It's suited for cases where you want to extract specific information and clean up the HTML content according to your needs.

For example, we can scrape text content within `<p>, <li>, <div>, and <a>` tags from the HTML content:

* `<p>`: The paragraph tag. It defines a paragraph in HTML and is used to group together related sentences and/or phrases.
 
* `<li>`: The list item tag. It is used within ordered (`<ol>`) and unordered (`<ul>`) lists to define individual items within the list.
 
* `<div>`: The division tag. It is a block-level element used to group other inline or block-level elements.
 
* `<a>`: The anchor tag. It is used to define hyperlinks.


```python
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# Load HTML
loader = AsyncChromiumLoader(["https://www.wsj.com"])
html = loader.load()
```


```python
# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    html, tags_to_extract=["p", "li", "div", "a"]
)
```


```python
docs_transformed[0].page_content[0:500]
```




    'Conservative legal activists are challenging Amazon, Comcast and others using many of the same tools that helped kill affirmative-action programs in colleges.1,2099 min read U.S. stock indexes fell and government-bond prices climbed, after Moody’s lowered credit ratings for 10 smaller U.S. banks and said it was reviewing ratings for six larger ones. The Dow industrials dropped more than 150 points.3 min read Penn Entertainment’s Barstool Sportsbook app will be rebranded as ESPN Bet this fall as '


