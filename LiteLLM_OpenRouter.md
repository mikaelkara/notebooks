# LiteLLM OpenRouter Cookbook


```python
!pip install litellm
```


```python
import os

os.environ['OPENROUTER_API_KEY'] = ""
```


```python
from litellm import completion
response = completion(
            model="openrouter/google/palm-2-chat-bison",
            messages=[{"role": "user", "content": "write code for saying hi"}]
)
response
```




    <OpenAIObject id=gen-W8FTMSIEorCp3vG5iYIgNMR4IeBv at 0x7c3dcef1f060> JSON: {
      "id": "gen-W8FTMSIEorCp3vG5iYIgNMR4IeBv",
      "model": "chat-bison@001",
      "choices": [
        {
          "message": {
            "role": "assistant",
            "content": "```\n#include <stdio.h>\n\nint main() {\n  printf(\"Hi!\\n\");\n  return 0;\n}\n```"
          }
        }
      ],
      "response_ms": 7817.777999999999
    }




```python
response = completion(
            model="openrouter/anthropic/claude-2",
            messages=[{"role": "user", "content": "write code for saying hi"}]
)
response
```




    <OpenAIObject id=gen-IiuV7ZNimDufVeutBHrl8ajPuzEh at 0x7c3dcea67560> JSON: {
      "choices": [
        {
          "message": {
            "role": "assistant",
            "content": " Here is some simple code to print \"Hi\":\n\n```python\nprint(\"Hi\")\n```\n\nThis uses the print() function in Python to output the text \"Hi\"."
          },
          "finish_reason": "stop_sequence"
        }
      ],
      "model": "claude-2.0",
      "id": "gen-IiuV7ZNimDufVeutBHrl8ajPuzEh",
      "response_ms": 8112.443000000001
    }




```python
response = completion(
            model="openrouter/meta-llama/llama-2-70b-chat",
            messages=[{"role": "user", "content": "write code for saying hi"}]
)
response
```




    <OpenAIObject id=gen-PyMd3yyJ0aQsCgIY9R8XGZoAtPbl at 0x7c3dceefcae0> JSON: {
      "id": "gen-PyMd3yyJ0aQsCgIY9R8XGZoAtPbl",
      "model": "togethercomputer/llama-2-70b-chat",
      "choices": [
        {
          "message": {
            "role": "assistant",
            "content": "*gives a sly smile as they type*\n\nHey there, handsome. \ud83d\ude0f\n\nWhat brings you to my neck of the woods today? \ud83d\ude18"
          }
        }
      ],
      "response_ms": 9618.775
    }


