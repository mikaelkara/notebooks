# Code generation with self-correction

AlphaCodium presented an approach for code generation that uses control flow.

Main idea: [construct an answer to a coding question iteratively.](https://x.com/karpathy/status/1748043513156272416?s=20). 

[AlphaCodium](https://github.com/Codium-ai/AlphaCodium) iteravely tests and improves an answer on public and AI-generated tests for a particular question. 

We will implement some of these ideas from scratch using [LangGraph](https://langchain-ai.github.io/langgraph/):

1. We show how to route user questions to different types of documentation
2. We we will show how to perform inline unit tests to confirm imports and code execution work
3. We will show how to use LangGraph to orchestrate this

![Screenshot 2024-05-23 at 2.17.51 PM.png](15d3ac32-cdf3-4800-a30c-f26d828d69c8.png)


```python
! pip install -U langchain_community langchain-mistralai langchain langgraph
```

### LLM

We'll use the Mistral API and `Codestral` instruct model, which support tool use!


```python
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
mistral_api_key = os.getenv("MISTRAL_API_KEY")  # Ensure this is set
```

### Tracing

Optionally, we'll use LangSmith for tracing.


```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<your-api-key>"
os.environ["LANGCHAIN_PROJECT"] = "Mistral-code-gen-testing"
```

## Code Generation

Test with structured output.


```python
# Select LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_mistralai import ChatMistralAI

mistral_model = "mistral-large-latest"
llm = ChatMistralAI(model=mistral_model, temperature=0)

# Prompt
code_gen_prompt_claude = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant. Ensure any code you provide can be executed with all required imports and variables \n
            defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block.
            \n Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)


# Data model
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description = "Schema for code solutions to questions about LCEL."


# LLM
code_gen_chain = llm.with_structured_output(code, include_raw=False)
```


```python
question = "Write a function for fibonacci."
messages = [("user", question)]
```


```python
# Test
result = code_gen_chain.invoke(messages)
result
```




    code(prefix='A function to calculate the nth Fibonacci number.', imports='', code='def fibonacci(n):\n    if n <= 0:\n        return "Input should be positive integer"\n    elif n == 1:\n        return 0\n    elif n == 2:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range(2, n):\n            a, b = b, a + b\n        return b', description='Schema for code solutions to questions about LCEL.')



## State


```python
from typing import Annotated, TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int
```

## Graph


```python
import uuid

from langchain_core.pydantic_v1 import BaseModel, Field

### Parameters
max_iterations = 3


### Nodes
def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]

    # Solution
    code_solution = code_gen_chain.invoke(messages)
    messages += [
        (
            "assistant",
            f"Here is my attempt to solve the problem: {code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def code_check(state: GraphState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [
            (
                "user",
                f"Your solution failed the import test. Here is the error: {e}. Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check execution
    try:
        combined_code = f"{imports}\n{code}"
        print(f"CODE TO TEST: {combined_code}")
        # Use a shared scope for exec
        global_scope = {}
        exec(combined_code, global_scope)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [
            (
                "user",
                f"Your solution failed the code execution test: {e}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


### Conditional edges


def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


### Utilities


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
```


```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START

builder = StateGraph(GraphState)

# Define the nodes
builder.add_node("generate", generate)  # generation solution
builder.add_node("check_code", code_check)  # check code

# Build graph
builder.add_edge(START, "generate")
builder.add_edge("generate", "check_code")
builder.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate",
    },
)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![jpeg](output_15_0.jpg)
    



```python
_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

question = "Write a Python program that prints 'Hello, World!' to the console."
events = graph.stream(
    {"messages": [("user", question)], "iterations": 0}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)
```

`Trace:`

https://smith.langchain.com/public/53bcdaab-e3c5-4423-9908-c44595325c38/r


```python
_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

question = """Create a Python program that checks if a given string is a palindrome. A palindrome is a word, phrase, number, or other sequence of characters that reads the same forward and backward (ignoring spaces, punctuation, and capitalization).

Requirements:
The program should define a function is_palindrome(s) that takes a string s as input.
The function should return True if the string is a palindrome and False otherwise.
Ignore spaces, punctuation, and case differences when checking for palindromes.

Give an example of it working on an example input word."""

events = graph.stream(
    {"messages": [("user", question)], "iterations": 0}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)
```

Trace:

https://smith.langchain.com/public/e749936d-7746-49de-b980-c41b17986e79/r


```python
_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

question = """Write a program that prints the numbers from 1 to 100. 
But for multiples of three, print "Fizz" instead of the number, and for the multiples of five, print "Buzz". 
For numbers which are multiples of both three and five, print "FizzBuzz"."""

events = graph.stream(
    {"messages": [("user", question)], "iterations": 0}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)
```

Trace: 

https://smith.langchain.com/public/f5c19708-7592-4512-9f00-9696ab34a9eb/r


```python
import uuid

_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

question = """I want to vectorize a function

        frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for i, val1 in enumerate(rows):
            for j, val2 in enumerate(cols):
                for j, val3 in enumerate(ch):
                    # Assuming you want to store the pair as tuples in the matrix
                    frame[i, j, k] = image[val1, val2, val3]

        out.write(np.array(frame))

with a simple numpy function that does something like this what is it called. Show me a test case with this working."""

events = graph.stream(
    {"messages": [("user", question)], "iterations": 0}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)
```

Trace w/ good example of self-correction:

https://smith.langchain.com/public/b54778a0-d267-4f09-bc28-71761201c522/r


```python
_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

question = """Create a Python program that allows two players to play a game of Tic-Tac-Toe. The game should be played on a 3x3 grid. The program should:

- Allow players to take turns to input their moves.
- Check for invalid moves (e.g., placing a marker on an already occupied space).
- Determine and announce the winner or if the game ends in a draw.

Requirements:
- Use a 2D list to represent the Tic-Tac-Toe board.
- Use functions to modularize the code.
- Validate player input.
- Check for win conditions and draw conditions after each move."""

events = graph.stream(
    {"messages": [("user", question)], "iterations": 0}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)
```

Trace w/ good example of failure to correct:

https://smith.langchain.com/public/871ae736-2f77-44d4-b0da-a600d8f5377d/r


```python

```
