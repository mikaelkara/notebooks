# Google Finance

This notebook goes over how to use the Google Finance Tool to get information from the Google Finance page

To get an SerpApi key key, sign up at: https://serpapi.com/users/sign_up.

Then install google-search-results with the command: 

pip install google-search-results

Then set the environment variable SERPAPI_API_KEY to your SerpApi key

Or pass the key in as a argument to the wrapper serp_api_key="your secret key"

Use the Tool


```python
%pip install --upgrade --quiet  google-search-results langchain-community
```


```python
import os

from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper

os.environ["SERPAPI_API_KEY"] = ""
tool = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper())
```


```python
tool.run("Google")
```

Using it with Langchain


```python
import os

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERP_API_KEY"] = ""
llm = OpenAI()
tools = load_tools(["google-scholar", "google-finance"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run("what is google's stock")
```
