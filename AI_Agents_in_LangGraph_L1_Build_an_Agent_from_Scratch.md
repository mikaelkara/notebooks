<a href="https://colab.research.google.com/github/meta-llama/llama-recipes/blob/main/recipes/quickstart/agents/dlai/AI_Agents_in_LangGraph_L1_Build_an_Agent_from_Scratch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook ports the DeepLearning.AI short course [AI Agents in LangGraph Lesson 1 Build an Agent from Scratch](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/2/build-an-agent-from-scratch) to using Llama 3, with a bonus section that ports the agent from scratch code to using LangGraph, introduced in [Lession 2 LangGraph Components](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/3/langgraph-components) of the course. 

You should take the course, especially the first two lessons, before or after going through this notebook, to have a deeper understanding.


```python
!pip install groq
```


```python
import os 
from groq import Groq

os.environ['GROQ_API_KEY'] = 'your_groq_api_key' # get a free key at https://console.groq.com/keys
```


```python
# a quick sanity test of calling Llama 3 70b on Groq 
# see https://console.groq.com/docs/text-chat for more info
client = Groq()
chat_completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[{"role": "user", "content": "what are the words Charlotte wrote for the pig?"}]
)
print(chat_completion.choices[0].message.content)
```

### ReAct Agent from Sractch


```python
client = Groq()
model = "llama3-8b-8192" # this model works with the prompt below only for the first simpler example; you'll see how to modify the prompt to make it work for a more complicated question
#model = "llama3-70b-8192" # this model works with the prompt below for both example questions 

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
                        model=model,
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content
    
```


```python
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()
```


```python
def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
```


```python
abot = Agent(prompt)
```


```python
result = abot("How much does a toy poodle weigh?")
print(result)
```


```python
abot.messages
```


```python
# manually call the exeternal func (tool) for now
result = average_dog_weight("Toy Poodle")
```


```python
result
```


```python
next_prompt = "Observation: {}".format(result)
```


```python
abot(next_prompt)
```


```python
abot.messages
```


```python
abot = Agent(prompt)
```


```python
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
abot(question)

```


```python
abot.messages
```


```python
next_prompt = "Observation: {}".format(average_dog_weight("Border Collie"))
print(next_prompt)
```


```python
abot(next_prompt)
```


```python
abot.messages
```


```python
next_prompt = "Observation: {}".format(average_dog_weight("Scottish Terrier"))
print(next_prompt)
```


```python
abot(next_prompt)
```


```python
abot.messages
```


```python
next_prompt = "Observation: {}".format(eval("37 + 20"))
print(next_prompt)
```


```python
abot(next_prompt)
```


```python
abot.messages
```

### Automate the ReAct action execution


```python
import re

# automate the action execution above to make the whole ReAct (Thought - Action- Observsation) process fully automated
action_re = re.compile('^Action: (\w+): (.*)$')   # python regular expression to selection action
```


```python
def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt) # set system prompt
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))

            # key to make the agent process fully automated:
            # programtically call the external func with arguments, with the info returned by LLM
            observation = known_actions[action](action_input) 

            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return
```

#### Using model "llama3-8b-8192", the code below will cause an invalid syntax error because the Action returned is calculate: (average_dog_weight: Border Collie) + (average_dog_weight: Scottish Terrier), instead of the expected "Action: average_dog_weight: Border Collie".


```python
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
query(question)
```

#### Prompt engineering in action:
REPLACE "Use Thought to describe your thoughts about the question you have been asked. Use Action to run one of the actions available to you - then return PAUSE." with 
"First, use Thought to describe your thoughts about the question you have been asked, and generate Action to run one of the actions available to you, then return PAUSE."


```python
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
First, use Thought to describe your thoughts about the question you have been asked, and generate Action to run one of the actions available to you, then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()
```


```python
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
query(question)
```

### Bonus: Port the Agent Implementation to LangGraph


```python
!pip install langchain
!pip install langgraph
!pip install langchain_openai
!pip install langchain_community
!pip install httpx
!pip install langchain-groq
```


```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
```


```python
from langchain_groq import ChatGroq

model = ChatGroq(temperature=0, model_name="llama3-8b-8192")
```


```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def calculate(what):
    """Runs a calculation and returns the number."""
    return eval(what)

@tool
def average_dog_weight(name):
    """Returns the average weight of a dog."""
    if name in "Scottish Terrier":
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

```


```python
prompt = """
You run in a loop of Thought, Action, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()
```


```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
```


```python
class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_llm(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
```


```python
abot = Agent(model, [calculate, average_dog_weight], system=prompt)
```


```python
messages = [HumanMessage(content="How much does a Toy Poodle weigh?")]
result = abot.graph.invoke({"messages": messages})
result['messages'], result['messages'][-1].content

# the code above will cause an error because Llama 3 8B incorrectly returns an extra "calculate" tool call
```


```python
# using the Llama 3 70B will fix the error
model = ChatGroq(temperature=0, model_name="llama3-70b-8192")
abot = Agent(model, [calculate, average_dog_weight], system=prompt)
```


```python
# Toy Poodle case sensitive here - can be fixed easily by modifying def average_dog_weight
messages = [HumanMessage(content="How much does a Toy Poodle weigh?")]
result = abot.graph.invoke({"messages": messages})
result['messages'], result['messages'][-1].content
```


```python
messages = [HumanMessage(content="I have 2 dogs, a border collie and a scottish terrier. What are their average weights? Total weight?")]
result = abot.graph.invoke({"messages": messages})
```


```python
result['messages'], result['messages'][-1].content
```
