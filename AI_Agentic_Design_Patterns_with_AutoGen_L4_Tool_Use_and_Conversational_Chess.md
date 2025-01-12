<a href="https://colab.research.google.com/github/meta-llama/llama-recipes/blob/main/recipes/quickstart/agents/dlai/AI_Agentic_Design_Patterns_with_AutoGen_L4_Tool_Use_and_Conversational_Chess.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook ports the DeepLearning.AI short course [AI Agentic Design Patterns with AutoGen Lesson 4 Tool Use and Conversational Chess](https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen/lesson/5/tool-use-and-conversational-chess) to using Llama 3. 

You should take the course before or after going through this notebook to have a deeper understanding.


```python
!pip install chess
!pip install pyautogen
```


```python
import chess
import chess.svg
from typing_extensions import Annotated
```


```python
board = chess.Board()

made_move = False
```


```python
def get_legal_moves(
    
) -> Annotated[str, "A list of legal moves in UCI format"]:
    return "Possible moves are: " + ",".join(
        [str(move) for move in board.legal_moves]
    )
```


```python
from IPython.display import SVG

def make_move(
    move: Annotated[str, "A move in UCI format."]
) -> Annotated[str, "Result of the move."]:
    move = chess.Move.from_uci(move)
    board.push_uci(str(move))
    global made_move
    made_move = True
    
    svg_str = chess.svg.board(
            board,
            arrows=[(move.from_square, move.to_square)],
            fill={move.from_square: "gray"},
            size=200
        )
    display(
        SVG(data=svg_str)
    )
    
    # Get the piece name.
    piece = board.piece_at(move.to_square)
    piece_symbol = piece.unicode_symbol()
    piece_name = (
        chess.piece_name(piece.piece_type).capitalize()
        if piece_symbol.isupper()
        else chess.piece_name(piece.piece_type)
    )
    return f"Moved {piece_name} ({piece_symbol}) from "\
    f"{chess.SQUARE_NAMES[move.from_square]} to "\
    f"{chess.SQUARE_NAMES[move.to_square]}."
```


```python
# base url from https://console.groq.com/docs/openai
config_list = [
    {
        "model": "llama3-70b-8192",
        "base_url": "https://api.groq.com/openai/v1",
        'api_key': 'your_groq_api_key', # get a free key at https://console.groq.com/keys
    },
]
```


```python
from autogen import ConversableAgent

# Player white agent
player_white = ConversableAgent(
    name="Player White",
    system_message="You are a chess player and you play as white. "
    "First call get_legal_moves(), to get a list of legal moves in UCI format. "
    "Then call make_move(move) to make a move. Finally, tell the proxy what you have moved and ask the black to move", # added "Finally..." to make the agents work
    llm_config={"config_list": config_list,
                "temperature": 0,
               },
)
```


```python
# Player black agent
player_black = ConversableAgent(
    name="Player Black",
    system_message="You are a chess player and you play as black. "
    "First call get_legal_moves(), to get a list of legal moves in UCI format. "
    "Then call make_move(move) to make a move. Finally, tell the proxy what you have moved and ask the white to move", # added "Finally..." to make the agents work
    llm_config={"config_list": config_list,
                "temperature": 0,
               },)
```


```python
def check_made_move(msg):
    global made_move
    if made_move:
        made_move = False
        return True
    else:
        return False

```


```python
board_proxy = ConversableAgent(
    name="Board Proxy",
    llm_config=False,
    is_termination_msg=check_made_move,
    default_auto_reply="Please make a move.",
    human_input_mode="NEVER",
)
```


```python
from autogen import register_function
```


```python
for caller in [player_white, player_black]:
    register_function(
        get_legal_moves,
        caller=caller,
        executor=board_proxy,
        name="get_legal_moves",
        description="Call this tool to get all legal moves in UCI format.",
    )
    
    register_function(
        make_move,
        caller=caller,
        executor=board_proxy,
        name="make_move",
        description="Call this tool to make a move.",
    )
```


```python
player_black.llm_config["tools"]
```


```python
player_white.register_nested_chats(
    trigger=player_black,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_white,
            "summary_method": "last_msg",
        }
    ],
)

player_black.register_nested_chats(
    trigger=player_white,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_black,
            "summary_method": "last_msg",
        }
    ],
)
```


```python
board = chess.Board()

chat_result = player_black.initiate_chat(
    player_white,
    message="Let's play chess! Your move.",
    max_turns=3,
)
```
