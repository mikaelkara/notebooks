# OUTPUT_PARSING_FAILURE

An [output parser](/docs/concepts/output_parsers) was unable to handle model output as expected.

To illustrate this, let's say you have an output parser that expects a chat model to output JSON surrounded by a markdown code tag (triple backticks). Here would be an example of good input:


```python
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser

message = AIMessage(content='```\n{"foo": "bar"}\n```')
output_parser = JsonOutputParser()
output_parser.invoke(message)
```




    {'foo': 'bar'}



Internally, our JSON parser stripped out the markdown fence and newlines and then ran `json.loads`.

If instead the chat model generated an output with malformed JSON, we will get an error:


```python
message = AIMessage(content='```\n{{"foo":\n```')
output_parser = JsonOutputParser()
output_parser.invoke(message)
```


    ---------------------------------------------------------------------------

    JSONDecodeError                           Traceback (most recent call last)

    File ~/langchain/oss-py/libs/core/langchain_core/output_parsers/json.py:83, in JsonOutputParser.parse_result(self, result, partial)
         82 try:
    ---> 83     return parse_json_markdown(text)
         84 except JSONDecodeError as e:
    

    File ~/langchain/oss-py/libs/core/langchain_core/utils/json.py:144, in parse_json_markdown(json_string, parser)
        143     json_str = json_string if match is None else match.group(2)
    --> 144 return _parse_json(json_str, parser=parser)
    

    File ~/langchain/oss-py/libs/core/langchain_core/utils/json.py:160, in _parse_json(json_str, parser)
        159 # Parse the JSON string into a Python dictionary
    --> 160 return parser(json_str)
    

    File ~/langchain/oss-py/libs/core/langchain_core/utils/json.py:118, in parse_partial_json(s, strict)
        115 # If we got here, we ran out of characters to remove
        116 # and still couldn't parse the string as JSON, so return the parse error
        117 # for the original string.
    --> 118 return json.loads(s, strict=strict)
    

    File ~/.pyenv/versions/3.11.4/lib/python3.11/json/__init__.py:359, in loads(s, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
        358     kw['parse_constant'] = parse_constant
    --> 359 return cls(**kw).decode(s)
    

    File ~/.pyenv/versions/3.11.4/lib/python3.11/json/decoder.py:337, in JSONDecoder.decode(self, s, _w)
        333 """Return the Python representation of ``s`` (a ``str`` instance
        334 containing a JSON document).
        335 
        336 """
    --> 337 obj, end = self.raw_decode(s, idx=_w(s, 0).end())
        338 end = _w(s, end).end()
    

    File ~/.pyenv/versions/3.11.4/lib/python3.11/json/decoder.py:353, in JSONDecoder.raw_decode(self, s, idx)
        352 try:
    --> 353     obj, end = self.scan_once(s, idx)
        354 except StopIteration as err:
    

    JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)

    
    The above exception was the direct cause of the following exception:
    

    OutputParserException                     Traceback (most recent call last)

    Cell In[9], line 3
          1 message = AIMessage(content='```\n{{"foo":\n```')
          2 output_parser = JsonOutputParser()
    ----> 3 output_parser.invoke(message)
    

    File ~/langchain/oss-py/libs/core/langchain_core/output_parsers/base.py:193, in BaseOutputParser.invoke(self, input, config, **kwargs)
        186 def invoke(
        187     self,
        188     input: Union[str, BaseMessage],
        189     config: Optional[RunnableConfig] = None,
        190     **kwargs: Any,
        191 ) -> T:
        192     if isinstance(input, BaseMessage):
    --> 193         return self._call_with_config(
        194             lambda inner_input: self.parse_result(
        195                 [ChatGeneration(message=inner_input)]
        196             ),
        197             input,
        198             config,
        199             run_type="parser",
        200         )
        201     else:
        202         return self._call_with_config(
        203             lambda inner_input: self.parse_result([Generation(text=inner_input)]),
        204             input,
        205             config,
        206             run_type="parser",
        207         )
    

    File ~/langchain/oss-py/libs/core/langchain_core/runnables/base.py:1927, in Runnable._call_with_config(self, func, input, config, run_type, serialized, **kwargs)
       1923     context = copy_context()
       1924     context.run(_set_config_context, child_config)
       1925     output = cast(
       1926         Output,
    -> 1927         context.run(
       1928             call_func_with_variable_args,  # type: ignore[arg-type]
       1929             func,  # type: ignore[arg-type]
       1930             input,  # type: ignore[arg-type]
       1931             config,
       1932             run_manager,
       1933             **kwargs,
       1934         ),
       1935     )
       1936 except BaseException as e:
       1937     run_manager.on_chain_error(e)
    

    File ~/langchain/oss-py/libs/core/langchain_core/runnables/config.py:396, in call_func_with_variable_args(func, input, config, run_manager, **kwargs)
        394 if run_manager is not None and accepts_run_manager(func):
        395     kwargs["run_manager"] = run_manager
    --> 396 return func(input, **kwargs)
    

    File ~/langchain/oss-py/libs/core/langchain_core/output_parsers/base.py:194, in BaseOutputParser.invoke.<locals>.<lambda>(inner_input)
        186 def invoke(
        187     self,
        188     input: Union[str, BaseMessage],
        189     config: Optional[RunnableConfig] = None,
        190     **kwargs: Any,
        191 ) -> T:
        192     if isinstance(input, BaseMessage):
        193         return self._call_with_config(
    --> 194             lambda inner_input: self.parse_result(
        195                 [ChatGeneration(message=inner_input)]
        196             ),
        197             input,
        198             config,
        199             run_type="parser",
        200         )
        201     else:
        202         return self._call_with_config(
        203             lambda inner_input: self.parse_result([Generation(text=inner_input)]),
        204             input,
        205             config,
        206             run_type="parser",
        207         )
    

    File ~/langchain/oss-py/libs/core/langchain_core/output_parsers/json.py:86, in JsonOutputParser.parse_result(self, result, partial)
         84 except JSONDecodeError as e:
         85     msg = f"Invalid json output: {text}"
    ---> 86     raise OutputParserException(msg, llm_output=text) from e
    

    OutputParserException: Invalid json output: ```
    {{"foo":
    ```


Note that some prebuilt constructs like [legacy LangChain agents](/docs/how_to/agent_executor) and chains may use output parsers internally,
so you may see this error even if you're not visibly instantiating and using an output parser.

## Troubleshooting

The following may help resolve this error:

- Consider using [tool calling or other structured output techniques](/docs/how_to/structured_output/) if possible without an output parser to reliably output parseable values.
  - If you are using a prebuilt chain or agent, use [LangGraph](https://langchain-ai.github.io/langgraph/) to compose your logic explicitly instead.
- Add more precise formatting instructions to your prompt. In the above example, adding `"You must always return valid JSON fenced by a markdown code block. Do not return any additional text."` to your input may help steer the model to returning the expected format.
- If you are using a smaller or less capable model, try using a more capable one.
- Add [LLM-powered retries](/docs/how_to/output_parser_fixing/).
