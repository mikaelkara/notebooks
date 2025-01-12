# How to split by HTML sections
## Description and motivation
Similar in concept to the [HTMLHeaderTextSplitter](/docs/how_to/HTML_header_metadata_splitter), the `HTMLSectionSplitter` is a "structure-aware" chunker that splits text at the element level and adds metadata for each header "relevant" to any given chunk.

It can return chunks element by element or combine elements with the same metadata, with the objectives of (a) keeping related text grouped (more or less) semantically and (b) preserving context-rich information encoded in document structures.

Use `xslt_path` to provide an absolute path to transform the HTML so that it can detect sections based on provided tags. The default is to use the `converting_to_header.xslt` file in the `data_connection/document_transformers` directory. This is for converting the html to a format/layout that is easier to detect sections. For example, `span` based on their font size can be converted to header tags to be detected as a section.

## Usage examples
### 1) How to split HTML strings:


```python
from langchain_text_splitters import HTMLSectionSplitter

html_string = """
    <!DOCTYPE html>
    <html>
    <body>
        <div>
            <h1>Foo</h1>
            <p>Some intro text about Foo.</p>
            <div>
                <h2>Bar main section</h2>
                <p>Some intro text about Bar.</p>
                <h3>Bar subsection 1</h3>
                <p>Some text about the first subtopic of Bar.</p>
                <h3>Bar subsection 2</h3>
                <p>Some text about the second subtopic of Bar.</p>
            </div>
            <div>
                <h2>Baz</h2>
                <p>Some text about Baz</p>
            </div>
            <br>
            <p>Some concluding text about Foo</p>
        </div>
    </body>
    </html>
"""

headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]

html_splitter = HTMLSectionSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits
```




    [Document(page_content='Foo \n Some intro text about Foo.', metadata={'Header 1': 'Foo'}),
     Document(page_content='Bar main section \n Some intro text about Bar. \n Bar subsection 1 \n Some text about the first subtopic of Bar. \n Bar subsection 2 \n Some text about the second subtopic of Bar.', metadata={'Header 2': 'Bar main section'}),
     Document(page_content='Baz \n Some text about Baz \n \n \n Some concluding text about Foo', metadata={'Header 2': 'Baz'})]



### 2) How to constrain chunk sizes:

`HTMLSectionSplitter` can be used with other text splitters as part of a chunking pipeline. Internally, it uses the `RecursiveCharacterTextSplitter` when the section size is larger than the chunk size. It also considers the font size of the text to determine whether it is a section or not based on the determined font size threshold.


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

html_string = """
    <!DOCTYPE html>
    <html>
    <body>
        <div>
            <h1>Foo</h1>
            <p>Some intro text about Foo.</p>
            <div>
                <h2>Bar main section</h2>
                <p>Some intro text about Bar.</p>
                <h3>Bar subsection 1</h3>
                <p>Some text about the first subtopic of Bar.</p>
                <h3>Bar subsection 2</h3>
                <p>Some text about the second subtopic of Bar.</p>
            </div>
            <div>
                <h2>Baz</h2>
                <p>Some text about Baz</p>
            </div>
            <br>
            <p>Some concluding text about Foo</p>
        </div>
    </body>
    </html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]

html_splitter = HTMLSectionSplitter(headers_to_split_on)

html_header_splits = html_splitter.split_text(html_string)

chunk_size = 500
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# Split
splits = text_splitter.split_documents(html_header_splits)
splits
```




    [Document(page_content='Foo \n Some intro text about Foo.', metadata={'Header 1': 'Foo'}),
     Document(page_content='Bar main section \n Some intro text about Bar.', metadata={'Header 2': 'Bar main section'}),
     Document(page_content='Bar subsection 1 \n Some text about the first subtopic of Bar.', metadata={'Header 3': 'Bar subsection 1'}),
     Document(page_content='Bar subsection 2 \n Some text about the second subtopic of Bar.', metadata={'Header 3': 'Bar subsection 2'}),
     Document(page_content='Baz \n Some text about Baz \n \n \n Some concluding text about Foo', metadata={'Header 2': 'Baz'})]


