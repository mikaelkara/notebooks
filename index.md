---
sidebar_position: 1
---
# How to migrate from v0.0 chains

LangChain has evolved since its initial release, and many of the original "Chain" classes 
have been deprecated in favor of the more flexible and powerful frameworks of LCEL and LangGraph. 

This guide will help you migrate your existing v0.0 chains to the new abstractions.

:::info How deprecated implementations work
Even though many of these implementations are deprecated, they are **still supported** in the codebase. 
However, they are not recommended for new development, and we recommend re-implementing them using the following guides!

To see the planned removal version for each deprecated implementation, check their API reference.
:::

:::info Prerequisites

These guides assume some familiarity with the following concepts:
- [LangChain Expression Language](/docs/concepts/lcel)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
:::

LangChain maintains a number of legacy abstractions. Many of these can be reimplemented via short combinations of LCEL and LangGraph primitives.

### LCEL
[LCEL](/docs/concepts/lcel) is designed to streamline the process of building useful apps with LLMs and combining related components. It does this by providing:

1. **A unified interface**: Every LCEL object implements the `Runnable` interface, which defines a common set of invocation methods (`invoke`, `batch`, `stream`, `ainvoke`, ...). This makes it possible to also automatically and consistently support useful operations like streaming of intermediate steps and batching, since every chain composed of LCEL objects is itself an LCEL object.
2. **Composition primitives**: LCEL provides a number of primitives that make it easy to compose chains, parallelize components, add fallbacks, dynamically configure chain internals, and more.

### LangGraph
[LangGraph](https://langchain-ai.github.io/langgraph/), built on top of LCEL, allows for performant orchestrations of application components while maintaining concise and readable code. It includes built-in persistence, support for cycles, and prioritizes controllability.
If LCEL grows unwieldy for larger or more complex chains, they may benefit from a LangGraph implementation.

### Advantages
Using these frameworks for existing v0.0 chains confers some advantages:

- The resulting chains typically implement the full `Runnable` interface, including streaming and asynchronous support where appropriate;
- The chains may be more easily extended or modified;
- The parameters of the chain are typically surfaced for easier customization (e.g., prompts) over previous versions, which tended to be subclasses and had opaque parameters and internals.
- If using LangGraph, the chain supports built-in persistence, allowing for conversational experiences via a "memory" of the chat history.
- If using LangGraph, the steps of the chain can be streamed, allowing for greater control and customizability.


The below pages assist with migration from various specific chains to LCEL and LangGraph:

- [LLMChain](./llm_chain.ipynb)
- [ConversationChain](./conversation_chain.ipynb)
- [RetrievalQA](./retrieval_qa.ipynb)
- [ConversationalRetrievalChain](./conversation_retrieval_chain.ipynb)
- [StuffDocumentsChain](./stuff_docs_chain.ipynb)
- [MapReduceDocumentsChain](./map_reduce_chain.ipynb)
- [MapRerankDocumentsChain](./map_rerank_docs_chain.ipynb)
- [RefineDocumentsChain](./refine_docs_chain.ipynb)
- [LLMRouterChain](./llm_router_chain.ipynb)
- [MultiPromptChain](./multi_prompt_chain.ipynb)
- [LLMMathChain](./llm_math_chain.ipynb)
- [ConstitutionalChain](./constitutional_chain.ipynb)

Check out the [LCEL conceptual docs](/docs/concepts/lcel) and [LangGraph docs](https://langchain-ai.github.io/langgraph/) for more background information.
