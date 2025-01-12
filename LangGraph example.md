# LangGraph example

In this guide, we will see how to integrate Literal AI in a LangGraph workflow.

- [Necessary imports](#imports)
- [Define tools](#define-available-tools)
- [Agent logic](#agent-logic)
- [Run agent](#run-agent)

<a id="imports"></a>
## Necessary imports

Make sure to define the `LITERAL_API_KEY`, `OPENAI_API_KEY` and `TAVILY_API_KEY` in your `.env`.

If you have a prompt template, check https://docs.literalai.com/guides/prompt-management#convert-to-langchain-chat-prompt


```python
from typing import Annotated
from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from langchain.schema.runnable.config import RunnableConfig
import os
from literalai import LiteralClient
```

# Check environment variables


```python
load_dotenv()
# print(os.getenv("LITERAL_API_KEY"))
# print(os.getenv("TAVILY_API_KEY"))
# print(os.getenv("OPENAI_API_KEY"))
```




    True



<a id="define-available-tools"></a>
## Define available tools

We will use Tavily as a search tool. `tools` is a list of available tools  (here, we only have one tool, the TavilySearchResults tool).


```python
load_dotenv()
literalai_client = LiteralClient()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Define the tool (TavilySearchResults tool to search the web)
tool = TavilySearchResults(max_results=2, k=2)
tools = [tool]

# Define the LLM (chatGPT 4o-mini)
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
```

<a id="agent-logic"></a>
## Agent logic

For the agent logic, we simply repeat the following pattern (max. 5 times):
- ask the user question to the LLM, making the tools available
- execute tools if LLM asks for it, otherwise return message


```python
# Define the chatbot logic
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Add the tool node to the graph
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

```

<a id="run-agent"></a>
## Run agent against a question

The agent has a pre-set user question (What is the weather in Paris?). It is run in a separate thread to log the output in literal, and then prints the result.


```python
# wait for user input and then run the graph
with literalai_client.thread(name="Weather in Paris") as thread:
    user_input = "What is the weather in Paris?"
    cb = literalai_client.langchain_callback()
    res = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=RunnableConfig(callbacks=[cb]))
    print(res["messages"][-1].content)
```

    The current weather in Paris is as follows:
    
    - **Temperature**: 8.3°C (46.9°F)
    - **Condition**: Overcast
    - **Humidity**: 87%
    - **Wind**: 12.6 kph (7.8 mph) from the northeast
    - **Pressure**: 1032.0 mb (30.47 in)
    - **Visibility**: 8.0 km (4.0 miles)
    
    The weather feels like 6.2°C (43.1°F) due to wind chill. For more detailed weather updates, you can check [Weather API](https://www.weatherapi.com/).
    


```python
# EXPERIMENT

def run_agent(input: str):
    # Use the Runnable
    cb = literalai_client.langchain_callback()
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=input)]},
        config={"callbacks": [cb]}
    )
    return final_state



experiment = literalai_client.api.create_experiment(
    name="LangGraph", params=[]  # optional
)

def score_output(input, output, expected_output=None):
    # Faking the scoring
    return [{"name": "context_relevancy", "type": "AI", "value": 0.6}]


@literalai_client.experiment_item_run
def run_and_eval(input, expected_output=None):
    lc_output = run_agent(input)
    output = cb.process_content(lc_output)
    experiment_item = {
      "scores": score_output(input, output, expected_output),
      "input": {"question": input},
      "output": {"answer": output}
    }
    experiment.log(experiment_item)



def run_experiment(inputs):
    for input in inputs:
        run_and_eval(input)


run_experiment(["What is the weather in SF?", "What is the weather in Paris?"])


```


```python
![Experiment from Literal AI]
```
