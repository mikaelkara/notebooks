# MESSAGE_COERCION_FAILURE

Instead of always requiring instances of `BaseMessage`, several modules in LangChain take `MessageLikeRepresentation`, which is defined as:


```python
from typing import Union

from langchain_core.prompts.chat import (
    BaseChatPromptTemplate,
    BaseMessage,
    BaseMessagePromptTemplate,
)

MessageLikeRepresentation = Union[
    Union[BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate],
    tuple[
        Union[str, type],
        Union[str, list[dict], list[object]],
    ],
    str,
]
```

These include OpenAI style message objects (`{ role: "user", content: "Hello world!" }`),
tuples, and plain strings (which are converted to [`HumanMessages`](/docs/concepts/messages/#humanmessage)).

If one of these modules receives a value outside of one of these formats, you will receive an error like the following:


```python
from langchain_anthropic import ChatAnthropic

uncoercible_message = {"role": "HumanMessage", "random_field": "random value"}

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

model.invoke([uncoercible_message])
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File ~/langchain/oss-py/libs/core/langchain_core/messages/utils.py:318, in _convert_to_message(message)
        317     # None msg content is not allowed
    --> 318     msg_content = msg_kwargs.pop("content") or ""
        319 except KeyError as e:
    

    KeyError: 'content'

    
    The above exception was the direct cause of the following exception:
    

    ValueError                                Traceback (most recent call last)

    Cell In[5], line 10
          3 uncoercible_message = {
          4     "role": "HumanMessage",
          5     "random_field": "random value"
          6 }
          8 model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    ---> 10 model.invoke([uncoercible_message])
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:287, in BaseChatModel.invoke(self, input, config, stop, **kwargs)
        275 def invoke(
        276     self,
        277     input: LanguageModelInput,
       (...)
        281     **kwargs: Any,
        282 ) -> BaseMessage:
        283     config = ensure_config(config)
        284     return cast(
        285         ChatGeneration,
        286         self.generate_prompt(
    --> 287             [self._convert_input(input)],
        288             stop=stop,
        289             callbacks=config.get("callbacks"),
        290             tags=config.get("tags"),
        291             metadata=config.get("metadata"),
        292             run_name=config.get("run_name"),
        293             run_id=config.pop("run_id", None),
        294             **kwargs,
        295         ).generations[0][0],
        296     ).message
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:267, in BaseChatModel._convert_input(self, input)
        265     return StringPromptValue(text=input)
        266 elif isinstance(input, Sequence):
    --> 267     return ChatPromptValue(messages=convert_to_messages(input))
        268 else:
        269     msg = (
        270         f"Invalid input type {type(input)}. "
        271         "Must be a PromptValue, str, or list of BaseMessages."
        272     )
    

    File ~/langchain/oss-py/libs/core/langchain_core/messages/utils.py:348, in convert_to_messages(messages)
        346 if isinstance(messages, PromptValue):
        347     return messages.to_messages()
    --> 348 return [_convert_to_message(m) for m in messages]
    

    File ~/langchain/oss-py/libs/core/langchain_core/messages/utils.py:348, in <listcomp>(.0)
        346 if isinstance(messages, PromptValue):
        347     return messages.to_messages()
    --> 348 return [_convert_to_message(m) for m in messages]
    

    File ~/langchain/oss-py/libs/core/langchain_core/messages/utils.py:321, in _convert_to_message(message)
        319     except KeyError as e:
        320         msg = f"Message dict must contain 'role' and 'content' keys, got {message}"
    --> 321         raise ValueError(msg) from e
        322     _message = _create_message_from_message_type(
        323         msg_type, msg_content, **msg_kwargs
        324     )
        325 else:
    

    ValueError: Message dict must contain 'role' and 'content' keys, got {'role': 'HumanMessage', 'random_field': 'random value'}


## Troubleshooting

The following may help resolve this error:

- Ensure that all inputs to chat models are an array of LangChain message classes or a supported message-like.
  - Check that there is no stringification or other unexpected transformation occuring.
- Check the error's stack trace and add log or debugger statements.


