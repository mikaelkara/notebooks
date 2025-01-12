```python
%pip install -qU langchain-airbyte langchain_chroma
```

    Note: you may need to restart the kernel to use updated packages.
    


```python
import getpass

GITHUB_TOKEN = getpass.getpass()
```


```python
from langchain_airbyte import AirbyteLoader
from langchain_core.prompts import PromptTemplate

loader = AirbyteLoader(
    source="source-github",
    stream="pull_requests",
    config={
        "credentials": {"personal_access_token": GITHUB_TOKEN},
        "repositories": ["langchain-ai/langchain"],
    },
    template=PromptTemplate.from_template(
        """# {title}
by {user[login]}

{body}"""
    ),
    include_metadata=False,
)
docs = loader.load()
```


```python
print(docs[-2].page_content)
```

    # Updated partners/ibm README
    by williamdevena
    
    ## PR title
    partners: changed the README file for the IBM Watson AI integration in the libs/partners/ibm folder.
    
    ## PR message
    Description: Changed the README file of partners/ibm following the docs on https://python.langchain.com/docs/integrations/llms/ibm_watsonx
    
    The README includes:
    
    - Brief description
    - Installation
    - Setting-up instructions (API key, project id, ...)
    - Basic usage:
      - Loading the model
      - Direct inference
      - Chain invoking
      - Streaming the model output
      
    Issue: https://github.com/langchain-ai/langchain/issues/17545
    
    Dependencies: None
    
    Twitter handle: None
    


```python
len(docs)
```




    10283




```python
import tiktoken
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

enc = tiktoken.get_encoding("cl100k_base")

vectorstore = Chroma.from_documents(
    docs,
    embedding=OpenAIEmbeddings(
        disallowed_special=(enc.special_tokens_set - {"<|endofprompt|>"})
    ),
)
```


```python
retriever = vectorstore.as_retriever()
```


```python
retriever.invoke("pull requests related to IBM")
```




    [Document(page_content='# Updated partners/ibm README\nby williamdevena\n\n## PR title\r\npartners: changed the README file for the IBM Watson AI integration in the libs/partners/ibm folder.\r\n\r\n## PR message\r\nDescription: Changed the README file of partners/ibm following the docs on https://python.langchain.com/docs/integrations/llms/ibm_watsonx\r\n\r\nThe README includes:\r\n\r\n- Brief description\r\n- Installation\r\n- Setting-up instructions (API key, project id, ...)\r\n- Basic usage:\r\n  - Loading the model\r\n  - Direct inference\r\n  - Chain invoking\r\n  - Streaming the model output\r\n  \r\nIssue: https://github.com/langchain-ai/langchain/issues/17545\r\n\r\nDependencies: None\r\n\r\nTwitter handle: None'),
     Document(page_content='# Updated partners/ibm README\nby williamdevena\n\n## PR title\r\npartners: changed the README file for the IBM Watson AI integration in the `libs/partners/ibm` folder. \r\n\r\n\r\n\r\n## PR message\r\n- **Description:** Changed the README file of partners/ibm following the docs on https://python.langchain.com/docs/integrations/llms/ibm_watsonx\r\n\r\n    The README includes:\r\n    - Brief description\r\n    - Installation\r\n    - Setting-up instructions (API key, project id, ...)\r\n    - Basic usage:\r\n        - Loading the model\r\n        - Direct inference\r\n        - Chain invoking\r\n        - Streaming the model output\r\n\r\n\r\n- **Issue:** #17545\r\n- **Dependencies:** None\r\n- **Twitter handle:** None'),
     Document(page_content='# IBM: added partners package `langchain_ibm`, added llm\nby MateuszOssGit\n\n  - **Description:** Added `langchain_ibm` as an langchain partners package of IBM [watsonx.ai](https://www.ibm.com/products/watsonx-ai) LLM provider (`WatsonxLLM`)\r\n  - **Dependencies:** [ibm-watsonx-ai](https://pypi.org/project/ibm-watsonx-ai/),\r\n  - **Tag maintainer:** : \r\n\r\nPlease make sure your PR is passing linting and testing before submitting. Run `make format`, `make lint` and `make test` to check this locally. ✅'),
     Document(page_content='# Add WatsonX support\nby baptistebignaud\n\nIt is a connector to use a LLM from WatsonX.\r\nIt requires python SDK "ibm-generative-ai"\r\n\r\n(It might not be perfect since it is my first PR on a public repository 😄)')]




```python

```
