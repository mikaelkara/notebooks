[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1huPsNm9l4OcJvIcu63u0FFWF8X2J7zW3?usp=sharing)


```python
!pip3 install langchain openai langchain_openai langchainhub numexpr
```

    /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/pty.py:95: DeprecationWarning: This process (pid=28845) is multi-threaded, use of forkpty() may lead to deadlocks in the child.
      pid, fd = os.forkpty()
    

    Requirement already satisfied: langchain in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.1.20)
    Requirement already satisfied: openai in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (1.47.0)
    Requirement already satisfied: langchain_openai in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.1.7)
    Requirement already satisfied: langchainhub in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.1.21)
    Requirement already satisfied: numexpr in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (2.10.1)
    Requirement already satisfied: PyYAML>=5.3 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (6.0.2)
    Requirement already satisfied: SQLAlchemy<3,>=1.4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (2.0.35)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (3.10.5)
    Requirement already satisfied: dataclasses-json<0.7,>=0.5.7 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (0.6.7)
    Requirement already satisfied: langchain-community<0.1,>=0.0.38 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (0.0.38)
    Requirement already satisfied: langchain-core<0.2.0,>=0.1.52 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (0.1.52)
    Requirement already satisfied: langchain-text-splitters<0.1,>=0.0.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (0.0.2)
    Requirement already satisfied: langsmith<0.2.0,>=0.1.17 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (0.1.125)
    Requirement already satisfied: numpy<2,>=1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (1.26.4)
    Requirement already satisfied: pydantic<3,>=1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (2.9.2)
    Requirement already satisfied: requests<3,>=2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (2.32.3)
    Requirement already satisfied: tenacity<9.0.0,>=8.1.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain) (8.5.0)
    Requirement already satisfied: anyio<5,>=3.5.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (4.6.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (1.9.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (0.27.2)
    Requirement already satisfied: jiter<1,>=0.4.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (0.4.2)
    Requirement already satisfied: sniffio in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (1.3.1)
    Requirement already satisfied: tqdm>4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (4.66.5)
    Requirement already satisfied: typing-extensions<5,>=4.11 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (4.12.2)
    Requirement already satisfied: tiktoken<1,>=0.7 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain_openai) (0.7.0)
    Requirement already satisfied: packaging<25,>=23.2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchainhub) (23.2)
    Requirement already satisfied: types-requests<3.0.0.0,>=2.31.0.2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchainhub) (2.32.0.20240914)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (2.4.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (24.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.1.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.11.1)
    Requirement already satisfied: idna>=2.8 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from anyio<5,>=3.5.0->openai) (3.10)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain) (3.22.0)
    Requirement already satisfied: typing-inspect<1,>=0.4.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain) (0.9.0)
    Requirement already satisfied: certifi in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx<1,>=0.23.0->openai) (2024.8.30)
    Requirement already satisfied: httpcore==1.* in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx<1,>=0.23.0->openai) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
    Requirement already satisfied: jsonpatch<2.0,>=1.33 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-core<0.2.0,>=0.1.52->langchain) (1.33)
    Requirement already satisfied: orjson<4.0.0,>=3.9.14 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langsmith<0.2.0,>=0.1.17->langchain) (3.10.7)
    Requirement already satisfied: annotated-types>=0.6.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic<3,>=1->langchain) (0.7.0)
    Requirement already satisfied: pydantic-core==2.23.4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic<3,>=1->langchain) (2.23.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from requests<3,>=2->langchain) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from requests<3,>=2->langchain) (2.2.3)
    Requirement already satisfied: regex>=2022.1.18 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from tiktoken<1,>=0.7->langchain_openai) (2023.12.25)
    Requirement already satisfied: jsonpointer>=1.9 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from jsonpatch<2.0,>=1.33->langchain-core<0.2.0,>=0.1.52->langchain) (3.0.0)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain) (1.0.0)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.2[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip3 install --upgrade pip[0m
    

# Introduction - Fireworks x Langchain

In this notebook, we demonstrate how we can use Fireworks Function Calling model as a router across multiple models with specialized capabilities. Function calling models have seen rapid rise in usage because of their abilities to use external tools easily. One such powerful tool is other LLMs. We have quite a number of specialized OSS LLMs for [Coding](https://www.deepseek.com/), [Chatting in Certain Language](https://github.com/QwenLM/Qwen) or just plain [HF Assistants](https://huggingface.co/chat/assistants).

Function Calling model allows us to
1. Analyze the user query for intent
2. Find the best model to answer the request. And it could be the FC model itself!
3. Construct the right query for the choosen LLM.
4. Profit!

For this notebook we are going to use [LangChain](https://www.langchain.com/) framework to construct an agent chain which is capable of chit chatting & solving math equations using a calculator tool.

This agent chain will take in [custom defined tools](https://python.langchain.com/docs/modules/agents/tools/custom_tools) which are capable of executing a Math Query using a LLM. The main routing LLM is going to be Fireworks function calling model.

## Setup Deps

In order to accomplish the task in this notebook we are goign to import some dependencies from langchain along with LLMMathChain. To read more about what this chain does, check this [documentation](https://api.python.langchain.com/en/latest/chains/langchain.chains.llm_math.base.LLMMathChain.html).

For solving our math equations, we are going to use the recently released [Mixtral MoE](https://mistral.ai/news/mixtral-of-experts/). For all of our inferencing needs we are going to use the [Fireworks Inference Service](https://fireworks.ai/models).

In order to use the Fireworks AI function calling model, you must first obtain Fireworks API Keys. If you don't already have one, you can one by following the instructions [here](https://readme.fireworks.ai/docs/quickstart). Replace `YOUR_FW_API_KEY` with your obtained key.

**NOTE:** It's important to set temperature to 0.0 for the function calling model because we want reliable behaviour in routing.


```python
import os

from langchain.chains import LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Optional, Type

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")

llm = ChatOpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key="YOUR_FW_API_KEY",
    model="accounts/fireworks/models/firefunction-v2",
    temperature=0.0,
    max_tokens=256,
)

math_llm = ChatOpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key="fYOUR_FW_API_KEY",
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    temperature=0.0,
)
```

    /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/langchain/hub.py:86: DeprecationWarning: The `langchainhub sdk` is deprecated.
    Please use the `langsmith sdk` instead:
      pip install langsmith
    Use the `pull_prompt` method.
      res_dict = client.pull_repo(owner_repo_commit)
    

## Custom Tools

In order to seamlessly use function calling ability of the models, we can use [Custom Tools](https://python.langchain.com/docs/modules/agents/tools/custom_tools) functionality built into LangChain. This allows us to seamlessly define the schema of the functions our model can access along with the execution logic. It simplifies the JSON function specification construction.

For this notebook, we are going to construct a `CustomCalculatorTool` which will use LLM to answer & execute math queries. It takes in a single parameter `query` which has to be valid mathemetical expression.


```python

class CalculatorInput(BaseModel):
    query: str = Field(description="should be a math expression")

class CustomCalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Tool to evaluate mathemetical expressions"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, query: str) -> str:
        """Use the tool."""
        return LLMMathChain(llm=math_llm, verbose=True).run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("not support async")

tools = [
  CustomCalculatorTool()
]

agent = create_openai_tools_agent(llm, tools, prompt)

agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### Test Chit Chat Ability

As we outlined in the beginning, the model should be able to both chit-chat, route queries to external tools when necessary or answer from internal knowledge.

Let's first start with a question that can be answered from internal knowledge.


```python
print(agent.invoke({"input": "What is the capital of USA?"}))
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mThe capital of the United States of America is Washington D.C.[0m
    
    [1m> Finished chain.[0m
    {'input': 'What is the capital of USA?', 'output': 'The capital of the United States of America is Washington D.C.'}
    

## Test Calculator

Now let's test it's ability to detect a mathematical question & route the query accordingly.


```python
print(agent.invoke({"input": "What is 100 divided by 25?"}))
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `Calculator` with `{'query': '100/25'}`
    
    
    [0m
    
    [1m> Entering new LLMMathChain chain...[0m
    100/25

    /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/langchain/chains/llm_math/base.py:57: UserWarning: Directly instantiating an LLMMathChain with an llm is deprecated. Please instantiate with llm_chain argument or using the from_llm class method.
      warnings.warn(
    

    [32;1m[1;3m```text
    100/25
    ```
    ...numexpr.evaluate("100/25")...
    [0m
    Answer: [33;1m[1;3m4.0[0m
    [1m> Finished chain.[0m
    [36;1m[1;3mAnswer: 4.0[0m[32;1m[1;3mThe answer is 4.0.[0m
    
    [1m> Finished chain.[0m
    {'input': 'What is 100 divided by 25?', 'output': 'The answer is 4.0.'}
    

# Conclusion

The fireworks function calling model can route request to external tools or internal knowledge appropriately - thus helping developers build co-operative agents.
