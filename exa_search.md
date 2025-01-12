# Exa

>[Exa](https://exa.ai/) is a knowledge API for AI and developers.
>

## Installation and Setup

`Exa` integration exists in its own [partner package](https://pypi.org/project/langchain-exa/). You can install it with:


```python
%pip install -qU langchain-exa
```

In order to use the package, you will also need to set the `EXA_API_KEY` environment variable to your Exa API key.

## Retriever

You can use the [`ExaSearchRetriever`](/docs/integrations/tools/exa_search#using-exasearchretriever) in a standard retrieval pipeline. You can import it as follows.

See a [usage example](/docs/integrations/tools/exa_search).



```python
from langchain_exa import ExaSearchRetriever
```

## Tools

You can use Exa as an agent tool as described in the [Exa tool calling docs](/docs/integrations/tools/exa_search#using-the-exa-sdk-as-langchain-agent-tools).

See a [usage example](/docs/integrations/tools/exa_search).

### ExaFindSimilarResults

A tool that queries the Metaphor Search API and gets back JSON.


```python
from langchain_exa.tools import ExaFindSimilarResults
```

### ExaSearchResults

Exa Search tool.


```python
from langchain_exa.tools import ExaSearchResults
```
