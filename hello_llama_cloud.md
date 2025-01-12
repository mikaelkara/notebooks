## This demo app shows:
* How to run Llama 3 in the cloud hosted on OctoAI
* How to use LangChain to ask Llama general questions and follow up questions
* How to use LangChain to load a recent PDF doc - the Llama paper pdf - and chat about it. This is the well known RAG (Retrieval Augmented Generation) method to let LLM such as Llama be able to answer questions about your own data. RAG is one way to prevent LLM's hallucination

**Note** We will be using OctoAI to run the examples here. You will need to first sign into [OctoAI](https://octoai.cloud/) with your Github or Google account, then create a free API token [here](https://octo.ai/docs/getting-started/how-to-create-an-octoai-access-token) that you can use for a while (a month or $10 in OctoAI credits, whichever one runs out first).
After the free trial ends, you will need to enter billing info to continue to use Llama 3 hosted on OctoAI.

Let's start by installing the necessary packages:
- sentence-transformers for text embeddings
- chromadb gives us database capabilities
- langchain provides necessary RAG tools for this demo

And setting up the OctoAI token.


```python
%pip install langchain==0.1.19 octoai-sdk==0.10.1 openai sentence-transformers chromadb pypdf
```


```python
from getpass import getpass
import os

OCTOAI_API_TOKEN = getpass()
os.environ["OCTOAI_API_TOKEN"] = OCTOAI_API_TOKEN
```

Next we call the Llama 3 model from OctoAI. In this example we will use the Llama 3 8b instruct model. You can find more on Llama models on the [OctoAI text generation solution page](https://octoai.cloud/text).

At the time of writing this notebook the following Llama models are available on OctoAI:
* meta-llama-3-8b-instruct
* meta-llama-3-70b-instruct
* codellama-7b-instruct
* codellama-13b-instruct
* codellama-34b-instruct
* llama-2-13b-chat
* llama-2-70b-chat
* llamaguard-7b


```python
from langchain.llms.octoai_endpoint import OctoAIEndpoint

llama3_8b = "meta-llama-3-8b-instruct"
llm = OctoAIEndpoint(
    model=llama3_8b,
    max_tokens=500,
    temperature=0.01
)
```

With the model set up, you are now ready to ask some questions. Here is an example of the simplest way to ask the model some general questions.


```python
question = "who wrote the book Innovator's dilemma?"
answer = llm.invoke(question)
print(answer)
```

We will then try to follow up the response with a question asking for more information on the book. 

Since the chat history is not passed on Llama doesn't have the context and doesn't know this is more about the book thus it treats this as new query.



```python
# chat history not passed so Llama doesn't have the context and doesn't know this is more about the book
followup = "tell me more"
followup_answer = llm.invoke(followup)
print(followup_answer)
```

To get around this we will need to provide the model with history of the chat. 

To do this, we will use  [`ConversationBufferMemory`](https://python.langchain.com/docs/modules/memory/types/buffer) to pass the chat history to the model and give it the capability to handle follow up questions.


```python
# using ConversationBufferMemory to pass memory (chat history) for follow up questions
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=False
)
```

Once this is set up, let us repeat the steps from before and ask the model a simple question.

Then we pass the question and answer back into the model for context along with the follow up question.


```python
# restart from the original question
answer = conversation.predict(input=question)
print(answer)
```


```python
# pass context (previous question and answer) along with the follow up "tell me more" to Llama who now knows more of what
memory.save_context({"input": question},
                    {"output": answer})
followup_answer = conversation.predict(input=followup)
print(followup_answer)
```

Next, let's explore using Llama 3 to answer questions using documents for context. 
This gives us the ability to update Llama 3's knowledge thus giving it better context without needing to finetune. 

We will use the PyPDFLoader to load in a pdf, in this case, the Llama paper.


```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("https://arxiv.org/pdf/2307.09288.pdf")
docs = loader.load()
```


```python
# check docs length and content
print(len(docs), docs[0].page_content[0:300])
```

We need to store our documents. There are more than 30 vector stores (DBs) supported by LangChain.
For this example we will use [Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma) which is light-weight and in memory so it's easy to get started with.
For other vector stores especially if you need to store a large amount of data - see https://python.langchain.com/docs/integrations/vectorstores

We will also import the OctoAIEmbeddings and RecursiveCharacterTextSplitter to assist in storing the documents.


```python
from langchain.vectorstores import Chroma

# embeddings are numerical representations of the question and answer text
from langchain_community.embeddings import OctoAIEmbeddings

# use a common text splitter to split text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

To store the documents, we will need to split them into chunks using [`RecursiveCharacterTextSplitter`](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter) and create vector representations of these chunks using [`OctoAIEmbeddings`](https://octoai.cloud/tools/text/embeddings?mode=api&model=thenlper%2Fgte-large) on them before storing them into our vector database.

In general, you should use larger chuck sizes for highly structured text such as code and smaller size for less structured text. You may need to experiment with different chunk sizes and overlap values to find out the best numbers.


```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(docs)

# create the vector db to store all the split chunks as embeddings
embeddings = OctoAIEmbeddings(
    endpoint_url="https://text.octoai.run/v1/embeddings"
)
vectordb = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
)
```

We then use ` RetrievalQA` to retrieve the documents from the vector database and give the model more context on Llama, thereby increasing its knowledge.

For each question, LangChain performs a semantic similarity search of it in the vector db, then passes the search results as the context to Llama to answer the question.


```python
# use LangChain's RetrievalQA, to associate Llama with the loaded documents stored in the vector db
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

question = "What is llama?"
result = qa_chain({"query": question})
print(result['result'])
```

Now, lets bring it all together by incorporating follow up questions.

First we ask a follow up questions without giving the model context of the previous conversation.
Without this context, the answer we get does not relate to our original question.


```python
# no context passed so Llama doesn't have enough context to answer so it lets its imagination go wild
result = qa_chain({"query": "what are its use cases?"})
print(result['result'])
```

As we did before, let us use the `ConversationalRetrievalChain` package to give the model context of our previous question so we can add follow up questions.


```python
# use ConversationalRetrievalChain to pass chat history for follow up questions
from langchain.chains import ConversationalRetrievalChain
chat_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), return_source_documents=True)
```


```python
# let's ask the original question "What is llama?" again
result = chat_chain({"question": question, "chat_history": []})
print(result['answer'])
```


```python
# this time we pass chat history along with the follow up so good things should happen
chat_history = [(question, result["answer"])]
followup = "what are its use cases?"
followup_answer = chat_chain({"question": followup, "chat_history": chat_history})
print(followup_answer['answer'])
```

Further follow ups can be made possible by updating chat_history.

Note that results can get cut off. You may set "max_new_tokens" in the OctoAIEndpoint call above to a larger number (like shown below) to avoid the cut off.

```python
model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens": 1000}
```


```python
# further follow ups can be made possible by updating chat_history like this:
chat_history.append((followup, followup_answer["answer"]))
more_followup = "what tasks can it assist with?"
more_followup_answer = chat_chain({"question": more_followup, "chat_history": chat_history})
print(more_followup_answer['answer'])
```
