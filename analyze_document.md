# Analyze a single long document

The AnalyzeDocumentChain takes in a single document, splits it up, and then runs it through a CombineDocumentsChain.


```python
with open("../docs/docs/modules/state_of_the_union.txt") as f:
    state_of_the_union = f.read()
```


```python
from langchain.chains import AnalyzeDocumentChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```


```python
from langchain.chains.question_answering import load_qa_chain

qa_chain = load_qa_chain(llm, chain_type="map_reduce")
```


```python
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
```


```python
qa_document_chain.run(
    input_document=state_of_the_union,
    question="what did the president say about justice breyer?",
)
```




    'The President said, "Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service."'


