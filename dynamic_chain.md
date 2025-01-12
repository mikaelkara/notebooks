# How to create a dynamic (self-constructing) chain

:::info Prerequisites

This guide assumes familiarity with the following:
- [LangChain Expression Language (LCEL)](/docs/concepts/lcel)
- [How to turn any function into a runnable](/docs/how_to/functions)

:::

Sometimes we want to construct parts of a chain at runtime, depending on the chain inputs ([routing](/docs/how_to/routing/) is the most common example of this). We can create dynamic chains like this using a very useful property of RunnableLambda's, which is that if a RunnableLambda returns a Runnable, that Runnable is itself invoked. Let's see an example.

import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs
  customVarName="llm"
/>



```python
# | echo: false

from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229")
```


```python
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, chain

contextualize_instructions = """Convert the latest user question into a standalone question given the chat history. Don't answer the question, return the question and nothing else (no descriptive text)."""
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_instructions),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ]
)
contextualize_question = contextualize_prompt | llm | StrOutputParser()

qa_instructions = (
    """Answer the user question given the following context:\n\n{context}."""
)
qa_prompt = ChatPromptTemplate.from_messages(
    [("system", qa_instructions), ("human", "{question}")]
)


@chain
def contextualize_if_needed(input_: dict) -> Runnable:
    if input_.get("chat_history"):
        # NOTE: This is returning another Runnable, not an actual output.
        return contextualize_question
    else:
        return RunnablePassthrough() | itemgetter("question")


@chain
def fake_retriever(input_: dict) -> str:
    return "egypt's population in 2024 is about 111 million"


full_chain = (
    RunnablePassthrough.assign(question=contextualize_if_needed).assign(
        context=fake_retriever
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

full_chain.invoke(
    {
        "question": "what about egypt",
        "chat_history": [
            ("human", "what's the population of indonesia"),
            ("ai", "about 276 million"),
        ],
    }
)
```




    "According to the context provided, Egypt's population in 2024 is estimated to be about 111 million."



The key here is that `contextualize_if_needed` returns another Runnable and not an actual output. This returned Runnable is itself run when the full chain is executed.

Looking at the trace we can see that, since we passed in chat_history, we executed the contextualize_question chain as part of the full chain: https://smith.langchain.com/public/9e0ae34c-4082-4f3f-beed-34a2a2f4c991/r

Note that the streaming, batching, etc. capabilities of the returned Runnable are all preserved


```python
for chunk in contextualize_if_needed.stream(
    {
        "question": "what about egypt",
        "chat_history": [
            ("human", "what's the population of indonesia"),
            ("ai", "about 276 million"),
        ],
    }
):
    print(chunk)
```

    What
     is
     the
     population
     of
     Egypt
    ?
    
