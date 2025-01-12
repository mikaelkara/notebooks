# Language Agent Tree Search

[Language Agent Tree Search](https://arxiv.org/abs/2310.04406) (LATS), by Zhou, et. al, is a general LLM agent search algorithm that combines reflection/evaluation and search (specifically monte-carlo trees search) to get achieve better overall task performance compared to similar techniques like ReACT, Reflexion, or Tree of Thoughts.

![LATS diagram](969d281d-0b01-4252-acc1-b98efa936324.png)

It has four main steps:

1. Select: pick the best next actions based on the aggregate rewards from step (2). Either respond (if a solution is found or the max search depth is reached) or continue searching.
2. Expand and simulate: select the "best" 5 potential actions to take and execute them in parallel.
3. Reflect + Evaluate: observe the outcomes of these actions and score the decisions based on reflection (and possibly external feedback)
4. Backpropagate: update the scores of the root trajectories based on the outcomes.

## Setup

Install `langgraph` (for the framework), `langchain_openai` (for the LLM), and `langchain` + `tavily-python` (for the search engine).

We will use tavily search as a tool. You can get an API key [here](https://app.tavily.com/sign-in) or replace with a different tool of your choosing.


```python
%%capture --no-stderr
%pip install -U --quiet langchain langgraph langchain_openai
%pip install -U --quiet tavily-python
```


```python
import getpass
import os


def _set_if_undefined(var: str) -> None:
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var)


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>   

## Graph State

LATS is based on a  (greedy) Monte-Carlo tree search. For each search steps, it picks the node with the highest "upper confidence bound", which is a metric that balances exploitation (highest average reward) and exploration (lowest visits). Starting from that node, it generates N (5 in this case) new candidate actions to take, and adds them to the tree. It stops searching either when it has generated a valid solution OR when it has reached the maximum number of rollouts (search tree depth).

![Tree Diagram](9d9d2775-494e-4a53-bf7e-e95da29ce902.png)

Our LangGraph state will be composed of two items:
1. The root of the search tree
2. The user input


```python
import math
from collections import deque
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class Node:
    def __init__(
        self,
        messages: list[BaseMessage],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent
```

#### The graph state itself

The main component is the tree, represented by the root node.


```python
from typing_extensions import TypedDict


class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str
```

## Define Language Agent

Our agent will have three primary LLM-powered processes:
1. Reflect: score the action based on the tool response.
2. Initial response: to create the root node and start the search.
3. Expand: generate 5 candidate "next steps" from the best spot in the current tree

For more "Grounded" tool applications (such as code synthesis), you could integrate code execution into the reflection/reward step. This type of external feedback is very useful (though adds complexity to an already complicated example notebook).


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
```

#### Tools

For our example, we will give the language agent a search engine.


```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.prebuilt import ToolNode

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tools = [tavily_tool]
tool_node = ToolNode(tools=tools)
```

### Reflection

The reflection chain will score agent outputs based on the decision and the tool responses.
We will call this within the other two nodes.


```python
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain as as_runnable

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Reflect and grade the assistant response to the user question below.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)

reflection_llm_chain = (
    prompt
    | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
        run_name="Reflection"
    )
    | PydanticToolsParser(tools=[Reflection])
)


@as_runnable
def reflection_chain(inputs) -> Reflection:
    tool_choices = reflection_llm_chain.invoke(inputs)
    reflection = tool_choices[0]
    if not isinstance(inputs["candidate"][-1], AIMessage):
        reflection.found_solution = False
    return reflection
```

### Initial Response

We start with a single root node, generated by this first step. It responds to the user input either with a tool invocation or a response.


```python
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


initial_answer_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
    run_name="GenerateInitialCandidate"
)


parser = JsonOutputToolsParser(return_id=True)
```


```python
initial_response = initial_answer_chain.invoke(
    {"input": "Write a research report on lithium pollution."}
)
initial_response
```




    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_xRFx5hZJNyfurW9kWrPAWx15', 'function': {'arguments': '{"query":"lithium pollution research 2023"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 25, 'prompt_tokens': 93, 'total_tokens': 118, 'completion_tokens_details': {'reasoning_tokens': 0}}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a5d11b2ef2', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-448238e0-f2a7-4be0-b21d-03beb7d22121-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'lithium pollution research 2023'}, 'id': 'call_xRFx5hZJNyfurW9kWrPAWx15', 'type': 'tool_call'}], usage_metadata={'input_tokens': 93, 'output_tokens': 25, 'total_tokens': 118})



#### Starting Node

We will package up the candidate generation and reflection in a single node of our graph. This is represented by the following function:


```python
# Define the node we will add to the graph
def generate_initial_response(state: TreeState) -> dict:
    """Generate the initial candidate response."""
    res = initial_answer_chain.invoke({"input": state["input"]})
    parsed = parser.invoke(res)
    tool_responses = [
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": r["type"], "args": r["args"], "id": r["id"]}
                        ],
                    )
                ]
            }
        )
        for r in parsed
    ]
    output_messages = [res] + [tr["messages"][0] for tr in tool_responses]
    reflection = reflection_chain.invoke(
        {"input": state["input"], "candidate": output_messages}
    )
    root = Node(output_messages, reflection=reflection)
    return {
        **state,
        "root": root,
    }
```

### Candidate Generation

The following code prompts the same LLM to generate N additional candidates to check.


```python
# This generates N candidate values
# for a single input to sample actions from the environment


def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
    n = config["configurable"].get("N", 5)
    bound_kwargs = llm.bind_tools(tools=tools).kwargs
    chat_result = llm.generate(
        [messages.to_messages()],
        n=n,
        callbacks=config["callbacks"],
        run_name="GenerateCandidates",
        **bound_kwargs,
    )
    return [gen.message for gen in chat_result.generations[0]]


expansion_chain = prompt_template | generate_candidates
```


```python
res = expansion_chain.invoke({"input": "Write a research report on lithium pollution."})
res
```




    [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_rf2Ns2CW2LppxuUFI4irvRhM', 'function': {'arguments': '{"query":"lithium pollution research 2023"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None}, id='run-dc7c2f76-1eaf-4c65-8803-7ccededfcf0e-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'lithium pollution research 2023'}, 'id': 'call_rf2Ns2CW2LppxuUFI4irvRhM', 'type': 'tool_call'}], usage_metadata={'input_tokens': 93, 'output_tokens': 123, 'total_tokens': 216}),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_rf2Ns2CW2LppxuUFI4irvRhM', 'function': {'arguments': '{"query":"lithium pollution research report 2023"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None}, id='run-dc7c2f76-1eaf-4c65-8803-7ccededfcf0e-1', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'lithium pollution research report 2023'}, 'id': 'call_rf2Ns2CW2LppxuUFI4irvRhM', 'type': 'tool_call'}], usage_metadata={'input_tokens': 93, 'output_tokens': 123, 'total_tokens': 216}),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_rf2Ns2CW2LppxuUFI4irvRhM', 'function': {'arguments': '{"query":"lithium pollution research report"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None}, id='run-dc7c2f76-1eaf-4c65-8803-7ccededfcf0e-2', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'lithium pollution research report'}, 'id': 'call_rf2Ns2CW2LppxuUFI4irvRhM', 'type': 'tool_call'}], usage_metadata={'input_tokens': 93, 'output_tokens': 123, 'total_tokens': 216}),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_rf2Ns2CW2LppxuUFI4irvRhM', 'function': {'arguments': '{"query":"lithium pollution research report"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None}, id='run-dc7c2f76-1eaf-4c65-8803-7ccededfcf0e-3', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'lithium pollution research report'}, 'id': 'call_rf2Ns2CW2LppxuUFI4irvRhM', 'type': 'tool_call'}], usage_metadata={'input_tokens': 93, 'output_tokens': 123, 'total_tokens': 216}),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_rf2Ns2CW2LppxuUFI4irvRhM', 'function': {'arguments': '{"query":"lithium pollution research report 2023"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None}, id='run-dc7c2f76-1eaf-4c65-8803-7ccededfcf0e-4', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'lithium pollution research report 2023'}, 'id': 'call_rf2Ns2CW2LppxuUFI4irvRhM', 'type': 'tool_call'}], usage_metadata={'input_tokens': 93, 'output_tokens': 123, 'total_tokens': 216})]



#### Candidate generation node

We will package the candidate generation and reflection steps in the following "expand" node.
We do all the operations as a batch process to speed up execution.


```python
from collections import defaultdict


def select(root: Node) -> dict:
    """Starting from the root node a child node is selected at each tree level until a leaf node is reached."""

    if not root.children:
        return root

    node = root
    while node.children:
        max_child = max(node.children, key=lambda child: child.upper_confidence_bound())
        node = max_child

    return node


def expand(state: TreeState, config: RunnableConfig) -> dict:
    """Starting from the "best" node in the tree, generate N candidates for the next step."""
    root = state["root"]
    best_candidate: Node = select(root)
    messages = best_candidate.get_trajectory()
    # Generate N candidates from the single child candidate
    new_candidates = expansion_chain.invoke(
        {"input": state["input"], "messages": messages}, config
    )
    parsed = parser.batch(new_candidates)
    flattened = [
        (i, tool_call)
        for i, tool_calls in enumerate(parsed)
        for tool_call in tool_calls
    ]
    tool_responses = [
        (
            i,
            tool_node.invoke(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": tool_call["type"],
                                    "args": tool_call["args"],
                                    "id": tool_call["id"],
                                }
                            ],
                        )
                    ]
                }
            ),
        )
        for i, tool_call in flattened
    ]
    collected_responses = defaultdict(list)
    for i, resp in tool_responses:
        collected_responses[i].append(resp["messages"][0])
    output_messages = []
    for i, candidate in enumerate(new_candidates):
        output_messages.append([candidate] + collected_responses[i])

    # Reflect on each candidate
    # For tasks with external validation, you'd add that here.
    reflections = reflection_chain.batch(
        [{"input": state["input"], "candidate": msges} for msges in output_messages],
        config,
    )
    # Grow tree
    child_nodes = [
        Node(cand, parent=best_candidate, reflection=reflection)
        for cand, reflection in zip(output_messages, reflections)
    ]
    best_candidate.children.extend(child_nodes)
    # We have already extended the tree directly, so we just return the state
    return state
```

## Create Graph

With those two nodes defined, we are ready to define the graph. After each agent step, we have the option of finishing.


```python
from typing import Literal

from langgraph.graph import END, StateGraph, START


def should_loop(state: TreeState):
    """Determine whether to continue the tree search."""
    root = state["root"]
    if root.is_solved:
        return END
    if root.height > 5:
        return END
    return "expand"


builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand)
builder.add_edge(START, "start")


builder.add_conditional_edges(
    "start",
    # Either expand/rollout or finish
    should_loop,
    ["expand", END],
)
builder.add_conditional_edges(
    "expand",
    # Either continue to rollout or finish
    should_loop,
    ["expand", END],
)

graph = builder.compile()
```


```python
from IPython.display import Image

Image(graph.get_graph().draw_mermaid_png())
```




    
![jpeg](output_27_0.jpg)
    



## Invoke


```python
question = "Generate a table with the average size and weight, as well as the oldest recorded instance for each of the top 5 most common birds."
last_step = None
for step in graph.stream({"input": question}):
    last_step = step
    step_name, step_state = next(iter(step.items()))
    print(step_name)
    print("rolled out: ", step_state["root"].height)
    print("---")
```

    start
    rolled out:  1
    ---
    expand
    rolled out:  2
    ---
    


```python
solution_node = last_step["expand"]["root"].get_best_solution()
best_trajectory = solution_node.get_trajectory(include_reflections=False)
print(best_trajectory[-1].content)
```

    Let's synthesize the information into a coherent table summarizing the average size, weight, and the oldest recorded instance for each of the top 5 most common birds.
    
    ### Top 5 Most Common Birds
    Based on the search results, the top 5 most common birds are:
    1. Domestic Chicken
    2. House Sparrow
    3. European Starling
    4. Ring-billed Gull
    5. Barn Swallow
    
    ### Table: Average Size, Weight, and Oldest Recorded Instance
    
    | Bird               | Average Size (cm) | Average Weight (g) | Oldest Recorded Instance |
    |--------------------|-------------------|--------------------|-------------------------|
    | Domestic Chicken   | 40-50             | 1,200-2,500        | ~16 years (Pet record)  |
    | House Sparrow      | 14-18             | 24-40              | 13 years                |
    | European Starling  | 20-23             | 58-100             | 15 years                |
    | Ring-billed Gull   | 48-53             | 300-700            | 23 years                |
    | Barn Swallow       | 15-20             | 17-20              | 16 years                |
    
    ### Additional Details
    - **Domestic Chicken**: The average size and weight can vary significantly based on breed and diet. The oldest recorded pet chicken lived up to 16 years.
    - **House Sparrow**: Commonly found in urban areas, with an average lifespan significantly shorter in the wild.
    - **European Starling**: Known for their adaptability, starlings have a notable lifespan when not exposed to predators or harsh conditions.
    - **Ring-billed Gull**: These gulls are common in North America and have a relatively long lifespan compared to other birds.
    - **Barn Swallow**: Known for their migratory habits, these birds have relatively high longevity given their size.
    
    This table now provides a structured and comprehensive summary of the requested information.
    


```python
question = "Write out magnus carlson series of moves in his game against Alireza Firouzja and propose an alternate strategy"
last_step = None
for step in graph.stream({"input": question}):
    last_step = step
    step_name, step_state = next(iter(step.items()))
    print(step_name)
    print("rolled out: ", step_state["root"].height)
    print("---")
```

    start
    rolled out:  1
    ---
    expand
    rolled out:  2
    ---
    expand
    rolled out:  3
    ---
    expand
    rolled out:  3
    ---
    expand
    rolled out:  3
    ---
    


```python
solution_node = last_step["expand"]["root"].get_best_solution()
best_trajectory = solution_node.get_trajectory(include_reflections=False)
print(best_trajectory[-1].content)
```

    It appears that the specific game moves between Magnus Carlsen and Alireza Firouzja are not readily available in the search results. However, I can provide a general idea of what a typical game between high-level players like Carlsen and Firouzja might look like and propose an alternate strategy based on common chess principles.
    
    ### Example Game Moves (Hypothetical)
    Here's a hypothetical sequence of moves in a game between Magnus Carlsen and Alireza Firouzja:
    
    1. e4 e5
    2. Nf3 Nc6
    3. Bb5 a6
    4. Ba4 Nf6
    5. O-O Be7
    6. Re1 b5
    7. Bb3 d6
    8. c3 O-O
    9. h3 Nb8
    10. d4 Nbd7
    11. Nbd2 Bb7
    12. Bc2 Re8
    13. Nf1 Bf8
    14. Ng3 g6
    15. a4 c5
    16. d5 c4
    17. Be3 Qc7
    18. Qd2 Nc5
    19. Nh2 Bg7
    20. Ng4 Nxg4
    21. hxg4 Qd7
    22. f3 f6
    23. Kf2 Qf7
    24. Rh1 Rad8
    25. Rh3 Bc8
    26. Rah1 h6
    27. Bxh6 Bxh6
    28. Rxh6 Qg7
    29. g5 f5
    30. exf5 Bxf5
    31. Bxf5 gxf5
    32. Nh5 Qf7
    33. Nf6+ Kf8
    34. Rh8+ Ke7
    35. Rxe8+ Rxe8
    36. Nxe8 Qxe8
    37. Rh7+ Kd8
    38. g6 Qg8
    39. Qg5+ Kc8
    40. Qe7 Qd8
    41. Qxd8+ Kxd8
    42. g7 Kc7
    43. g8=Q+ Kb6
    44. Qb8+ Ka5
    45. Qd8+ Kxa4
    46. g4 fxg4
    47. fxg4 Kb3
    48. g5 Kxb2
    49. Qb6 Kxc3
    50. Qxc5 dxc5
    51. d6 b4
    52. d7 b3
    53. d8=Q b2
    54. Qd1 b1=Q
    55. Rxb1 Kxc4
    56. Qc1+ Kd5
    57. Qxc3 c4
    58. Ke3 Kc6
    59. Kd4 Kc7
    60. Qxc4+ Kd6
    61. Qc5+ Ke6
    62. Rb6+ Kf7
    63. Qc7+ Ke8
    64. Rb8#
    
    ### Alternate Strategy
    
    If we consider that Magnus Carlsen played the white pieces and used a typical Ruy Lopez opening, an alternate strategy could involve a different opening or a variation within the Ruy Lopez itself. For instance:
    
    1. **Alternative Opening: The Italian Game**
       - 1. e4 e5
       - 2. Nf3 Nc6
       - 3. Bc4 Bc5
       - 4. c3 Nf6
       - 5. d4 exd4
       - 6. cxd4 Bb4+
       - 7. Nc3 Nxe4
       - 8. O-O Bxc3
       - 9. d5 Ne7
       - 10. Qd3 f5
       - 11. bxc3 d6
       - 12. Nd4 O-O
       - 13. f3 Nc5
       - 14. Qc2 f4
       - 15. Re1 Ng6
       - 16. Ba3 Qg5
       - 17. Bxc5 dxc5
       - 18. Ne6 Bxe6
       - 19. dxe6 Ne7
       - 20. Rad1 Rad8
       - 21. Rd7 Rxd7
       - 22. exd7+ Kh8
       - 23. Qe4 Nc6
       - 24. Bd3 g6
       - 25. Qe8 Kg7
       - 26. Bb5 Nd8
       - 27. Re7+ Kh6
       - 28. Qxf8+ Kh5
       - 29. Rxh7#
    
    2. **Variation in the Ruy Lopez:**
       - Instead of the main lines, White could opt for the "Cozy Variation" or the "Deferred Steinitz Defense."
       - For example, after the initial moves:
         - 1. e4 e5
         - 2. Nf3 Nc6
         - 3. Bb5 a6
         - 4. Ba4 d6 (Deferred Steinitz Defense)
         - 5. c3 Bg4
         - 6. h3 Bh5
         - 7. d4 exd4
         - 8. cxd4 Be7
         - 9. Nc3 Nf6
         - 10. O-O O-O
    
    By varying the opening or the approach within a given opening, Carlsen could potentially avoid deep preparation by Firouzja and steer the game into less familiar territory for his opponent.
    

## Conclusion

Congrats on implementing LATS! This is a technique that can be reasonably fast and effective at solving complex reasoning tasks. A few notes that you probably observed above:
1. While effective , the tree rollout can take additional compute time. If you wanted to include this in a production app, you'd either want to ensure that intermediate steps are streamed (so the user sees the thinking process/has access to intermediate results) or use it for fine-tuning data to improve the single-shot accuracy and avoid long rollouts.
2. The candidate selection process is only as good as the reward you generate. Here we are using self-reflection exclusively, but if you have an external source of feedback (such as code test execution), that should be incorporated in the locations mentioned above.
