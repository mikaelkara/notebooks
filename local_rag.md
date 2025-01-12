# Build a Local RAG Application

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Chat Models](/docs/concepts/chat_models)
- [Chaining runnables](/docs/how_to/sequence/)
- [Embeddings](/docs/concepts/embedding_models)
- [Vector stores](/docs/concepts/vectorstores)
- [Retrieval-augmented generation](/docs/tutorials/rag/)

:::

The popularity of projects like [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://github.com/ollama/ollama), and [llamafile](https://github.com/Mozilla-Ocho/llamafile) underscore the importance of running LLMs locally.

LangChain has integrations with [many open-source LLM providers](/docs/how_to/local_llms) that can be run locally.

This guide will show how to run `LLaMA 3.1` via one provider, [Ollama](/docs/integrations/providers/ollama/) locally (e.g., on your laptop) using local embeddings and a local LLM. However, you can set up and swap in other local providers, such as [LlamaCPP](/docs/integrations/chat/llamacpp/) if you prefer.

**Note:** This guide uses a [chat model](/docs/concepts/chat_models) wrapper that takes care of formatting your input prompt for the specific local model you're using. However, if you are prompting local models directly with a [text-in/text-out LLM](/docs/concepts/text_llms) wrapper, you may need to use a prompt tailed for your specific model. This will often [require the inclusion of special tokens](https://huggingface.co/blog/llama2#how-to-prompt-llama-2). [Here's an example for LLaMA 2](https://smith.langchain.com/hub/rlm/rag-prompt-llama).

## Setup

First we'll need to set up Ollama.

The instructions [on their GitHub repo](https://github.com/ollama/ollama) provide details, which we summarize here:

- [Download](https://ollama.com/download) and run their desktop app
- From command line, fetch models from [this list of options](https://ollama.com/library). For this guide, you'll need:
  - A general purpose model like `llama3.1:8b`, which you can pull with something like `ollama pull llama3.1:8b`
  - A [text embedding model](https://ollama.com/search?c=embedding) like `nomic-embed-text`, which you can pull with something like `ollama pull nomic-embed-text`
- When the app is running, all models are automatically served on `localhost:11434`
- Note that your model choice will depend on your hardware capabilities

Next, install packages needed for local embeddings, vector storage, and inference.


```python
# Document loading, retrieval methods and text splitting
%pip install -qU langchain langchain_community

# Local vector store via Chroma
%pip install -qU langchain_chroma

# Local inference and embeddings via Ollama
%pip install -qU langchain_ollama

# Web Loader
%pip install -qU beautifulsoup4
```

You can also [see this page](/docs/integrations/text_embedding/) for a full list of available embeddings models

## Document Loading

Now let's load and split an example document.

We'll use a [blog post](https://lilianweng.github.io/posts/2023-06-23-agent/) by Lilian Weng on agents as an example.


```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
```

Next, the below steps will initialize your vector store. We use [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text), but you can explore other providers or options as well:


```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
```

And now we have a working vector store! Test that similarity search is working:


```python
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
len(docs)
```




    4




```python
docs[0]
```




    Document(metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"}, page_content='Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.')



Next, set up a model. We use Ollama with `llama3.1:8b` here, but you can [explore other providers](/docs/how_to/local_llms/) or [model options depending on your hardware setup](https://ollama.com/library):


```python
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.1:8b",
)
```

Test it to make sure you've set everything up properly:


```python
response_message = model.invoke(
    "Simulate a rap battle between Stephen Colbert and John Oliver"
)

print(response_message.content)
```

    **The scene is set: a packed arena, the crowd on their feet. In the blue corner, we have Stephen Colbert, aka "The O'Reilly Factor" himself. In the red corner, the challenger, John Oliver. The judges are announced as Tina Fey, Larry Wilmore, and Patton Oswalt. The crowd roars as the two opponents face off.**
    
    **Stephen Colbert (aka "The Truth with a Twist"):**
    Yo, I'm the king of satire, the one they all fear
    My show's on late, but my jokes are clear
    I skewer the politicians, with precision and might
    They tremble at my wit, day and night
    
    **John Oliver:**
    Hold up, Stevie boy, you may have had your time
    But I'm the new kid on the block, with a different prime
    Time to wake up from that 90s coma, son
    My show's got bite, and my facts are never done
    
    **Stephen Colbert:**
    Oh, so you think you're the one, with the "Last Week" crown
    But your jokes are stale, like the ones I wore down
    I'm the master of absurdity, the lord of the spin
    You're just a British import, trying to fit in
    
    **John Oliver:**
    Stevie, my friend, you may have been the first
    But I've got the skill and the wit, that's never blurred
    My show's not afraid, to take on the fray
    I'm the one who'll make you think, come what may
    
    **Stephen Colbert:**
    Well, it's time for a showdown, like two old friends
    Let's see whose satire reigns supreme, till the very end
    But I've got a secret, that might just seal your fate
    My humor's contagious, and it's already too late!
    
    **John Oliver:**
    Bring it on, Stevie! I'm ready for you
    I'll take on your jokes, and show them what to do
    My sarcasm's sharp, like a scalpel in the night
    You're just a relic of the past, without a fight
    
    **The judges deliberate, weighing the rhymes and the flow. Finally, they announce their decision:**
    
    Tina Fey: I've got to go with John Oliver. His jokes were sharper, and his delivery was smoother.
    
    Larry Wilmore: Agreed! But Stephen Colbert's still got that old-school charm.
    
    Patton Oswalt: You know what? It's a tie. Both of them brought the heat!
    
    **The crowd goes wild as both opponents take a bow. The rap battle may be over, but the satire war is just beginning...
    

## Using in a chain

We can create a summarization chain with either model by passing in retrieved docs and a simple prompt.

It formats the prompt template using the input key values provided and passes the formatted string to the specified model:


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)


# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | model | StrOutputParser()

question = "What are the approaches to Task Decomposition?"

docs = vectorstore.similarity_search(question)

chain.invoke(docs)
```




    'The main themes in these documents are:\n\n1. **Task Decomposition**: The process of breaking down complex tasks into smaller, manageable subgoals is crucial for efficient task handling.\n2. **Autonomous Agent System**: A system powered by Large Language Models (LLMs) that can perform planning, reflection, and refinement to improve the quality of final results.\n3. **Challenges in Planning and Decomposition**:\n\t* Long-term planning and task decomposition are challenging for LLMs.\n\t* Adjusting plans when faced with unexpected errors is difficult for LLMs.\n\t* Humans learn from trial and error, making them more robust than LLMs in certain situations.\n\nOverall, the documents highlight the importance of task decomposition and planning in autonomous agent systems powered by LLMs, as well as the challenges that still need to be addressed.'



## Q&A

You can also perform question-answering with your local model and vector store. Here's an example with a simple string prompt:


```python
from langchain_core.runnables import RunnablePassthrough

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)

question = "What are the approaches to Task Decomposition?"

docs = vectorstore.similarity_search(question)

# Run
chain.invoke({"context": docs, "question": question})
```




    'Task decomposition can be done through (1) simple prompting using LLM, (2) task-specific instructions, or (3) human inputs. This approach helps break down large tasks into smaller, manageable subgoals for efficient handling of complex tasks. It enables agents to plan ahead and improve the quality of final results through reflection and refinement.'



## Q&A with retrieval

Finally, instead of manually passing in docs, you can automatically retrieve them from our vector store based on the user question:


```python
retriever = vectorstore.as_retriever()

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)
```


```python
question = "What are the approaches to Task Decomposition?"

qa_chain.invoke(question)
```




    'Task decomposition can be done through (1) simple prompting in Large Language Models (LLM), (2) using task-specific instructions, or (3) with human inputs. This process involves breaking down large tasks into smaller, manageable subgoals for efficient handling of complex tasks.'



## Next steps

You've now seen how to build a RAG application using all local components. RAG is a very deep topic, and you might be interested in the following guides that discuss and demonstrate additional techniques:

- [Video: Reliable, fully local RAG agents with LLaMA 3](https://www.youtube.com/watch?v=-ROS6gfYIts) for an agentic approach to RAG with local models
- [Video: Building Corrective RAG from scratch with open-source, local LLMs](https://www.youtube.com/watch?v=E2shqsYwxck)
- [Conceptual guide on retrieval](/docs/concepts/retrieval) for an overview of various retrieval techniques you can apply to improve performance
- [How to guides on RAG](/docs/how_to/#qa-with-rag) for a deeper dive into different specifics around of RAG
- [How to run models locally](/docs/how_to/local_llms/) for different approaches to setting up different providers
