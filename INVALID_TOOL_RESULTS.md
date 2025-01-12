# INVALID_TOOL_RESULTS

You are passing too many, too few, or mismatched [`ToolMessages`](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.ToolMessage.html#toolmessage) to a model.

When [using a model to call tools](/docs/concepts/tool_calling), the [`AIMessage`](https://api.js.langchain.com/classes/_langchain_core.messages.AIMessage.html)
the model responds with will contain a `tool_calls` array. To continue the flow, the next messages you pass back to the model must
be exactly one `ToolMessage` for each item in that array containing the result of that tool call. Each `ToolMessage` must have a `tool_call_id` field
that matches one of the `tool_calls` on the `AIMessage`.

For example, given the following response from a model:


```python
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")


@tool
def foo_tool() -> str:
    """
    A dummy tool that returns 'action complete!'
    """
    return "action complete!"


model_with_tools = model.bind_tools([foo_tool])

chat_history: List[BaseMessage] = [
    HumanMessage(content='Call tool "foo" twice with no arguments')
]

response_message = model_with_tools.invoke(chat_history)

print(response_message.tool_calls)
```

    [{'name': 'foo_tool', 'args': {}, 'id': 'call_dq9O0eGHrryBwDRCnk0deHK4', 'type': 'tool_call'}, {'name': 'foo_tool', 'args': {}, 'id': 'call_mjLuNyXNHoUIXHiBtXhaWdxN', 'type': 'tool_call'}]
    

Calling the model with only one tool response would result in an error:


```python
from langchain_core.messages import AIMessage, ToolMessage

tool_call = response_message.tool_calls[0]
tool_response = foo_tool.invoke(tool_call)

chat_history.append(
    AIMessage(
        content=response_message.content,
        additional_kwargs=response_message.additional_kwargs,
    )
)
chat_history.append(
    ToolMessage(content=str(tool_response), tool_call_id=tool_call.get("id"))
)

final_response = model_with_tools.invoke(chat_history)
print(final_response)
```


    ---------------------------------------------------------------------------

    BadRequestError                           Traceback (most recent call last)

    Cell In[3], line 9
          6 chat_history.append(AIMessage(content=response_message.content, additional_kwargs=response_message.additional_kwargs))
          7 chat_history.append(ToolMessage(content=str(tool_response), tool_call_id=tool_call.get('id')))
    ----> 9 final_response = model_with_tools.invoke(chat_history)
         10 print(final_response)
    

    File ~/langchain/oss-py/libs/core/langchain_core/runnables/base.py:5354, in RunnableBindingBase.invoke(self, input, config, **kwargs)
       5348 def invoke(
       5349     self,
       5350     input: Input,
       5351     config: Optional[RunnableConfig] = None,
       5352     **kwargs: Optional[Any],
       5353 ) -> Output:
    -> 5354     return self.bound.invoke(
       5355         input,
       5356         self._merge_configs(config),
       5357         **{**self.kwargs, **kwargs},
       5358     )
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:286, in BaseChatModel.invoke(self, input, config, stop, **kwargs)
        275 def invoke(
        276     self,
        277     input: LanguageModelInput,
       (...)
        281     **kwargs: Any,
        282 ) -> BaseMessage:
        283     config = ensure_config(config)
        284     return cast(
        285         ChatGeneration,
    --> 286         self.generate_prompt(
        287             [self._convert_input(input)],
        288             stop=stop,
        289             callbacks=config.get("callbacks"),
        290             tags=config.get("tags"),
        291             metadata=config.get("metadata"),
        292             run_name=config.get("run_name"),
        293             run_id=config.pop("run_id", None),
        294             **kwargs,
        295         ).generations[0][0],
        296     ).message
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:786, in BaseChatModel.generate_prompt(self, prompts, stop, callbacks, **kwargs)
        778 def generate_prompt(
        779     self,
        780     prompts: list[PromptValue],
       (...)
        783     **kwargs: Any,
        784 ) -> LLMResult:
        785     prompt_messages = [p.to_messages() for p in prompts]
    --> 786     return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:643, in BaseChatModel.generate(self, messages, stop, callbacks, tags, metadata, run_name, run_id, **kwargs)
        641         if run_managers:
        642             run_managers[i].on_llm_error(e, response=LLMResult(generations=[]))
    --> 643         raise e
        644 flattened_outputs = [
        645     LLMResult(generations=[res.generations], llm_output=res.llm_output)  # type: ignore[list-item]
        646     for res in results
        647 ]
        648 llm_output = self._combine_llm_outputs([res.llm_output for res in results])
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:633, in BaseChatModel.generate(self, messages, stop, callbacks, tags, metadata, run_name, run_id, **kwargs)
        630 for i, m in enumerate(messages):
        631     try:
        632         results.append(
    --> 633             self._generate_with_cache(
        634                 m,
        635                 stop=stop,
        636                 run_manager=run_managers[i] if run_managers else None,
        637                 **kwargs,
        638             )
        639         )
        640     except BaseException as e:
        641         if run_managers:
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:851, in BaseChatModel._generate_with_cache(self, messages, stop, run_manager, **kwargs)
        849 else:
        850     if inspect.signature(self._generate).parameters.get("run_manager"):
    --> 851         result = self._generate(
        852             messages, stop=stop, run_manager=run_manager, **kwargs
        853         )
        854     else:
        855         result = self._generate(messages, stop=stop, **kwargs)
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/langchain_openai/chat_models/base.py:686, in BaseChatOpenAI._generate(self, messages, stop, run_manager, **kwargs)
        684     generation_info = {"headers": dict(raw_response.headers)}
        685 else:
    --> 686     response = self.client.create(**payload)
        687 return self._create_chat_result(response, generation_info)
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_utils/_utils.py:274, in required_args.<locals>.inner.<locals>.wrapper(*args, **kwargs)
        272             msg = f"Missing required argument: {quote(missing[0])}"
        273     raise TypeError(msg)
    --> 274 return func(*args, **kwargs)
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/resources/chat/completions.py:742, in Completions.create(self, messages, model, frequency_penalty, function_call, functions, logit_bias, logprobs, max_completion_tokens, max_tokens, metadata, n, parallel_tool_calls, presence_penalty, response_format, seed, service_tier, stop, store, stream, stream_options, temperature, tool_choice, tools, top_logprobs, top_p, user, extra_headers, extra_query, extra_body, timeout)
        704 @required_args(["messages", "model"], ["messages", "model", "stream"])
        705 def create(
        706     self,
       (...)
        739     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        740 ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        741     validate_response_format(response_format)
    --> 742     return self._post(
        743         "/chat/completions",
        744         body=maybe_transform(
        745             {
        746                 "messages": messages,
        747                 "model": model,
        748                 "frequency_penalty": frequency_penalty,
        749                 "function_call": function_call,
        750                 "functions": functions,
        751                 "logit_bias": logit_bias,
        752                 "logprobs": logprobs,
        753                 "max_completion_tokens": max_completion_tokens,
        754                 "max_tokens": max_tokens,
        755                 "metadata": metadata,
        756                 "n": n,
        757                 "parallel_tool_calls": parallel_tool_calls,
        758                 "presence_penalty": presence_penalty,
        759                 "response_format": response_format,
        760                 "seed": seed,
        761                 "service_tier": service_tier,
        762                 "stop": stop,
        763                 "store": store,
        764                 "stream": stream,
        765                 "stream_options": stream_options,
        766                 "temperature": temperature,
        767                 "tool_choice": tool_choice,
        768                 "tools": tools,
        769                 "top_logprobs": top_logprobs,
        770                 "top_p": top_p,
        771                 "user": user,
        772             },
        773             completion_create_params.CompletionCreateParams,
        774         ),
        775         options=make_request_options(
        776             extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
        777         ),
        778         cast_to=ChatCompletion,
        779         stream=stream or False,
        780         stream_cls=Stream[ChatCompletionChunk],
        781     )
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_base_client.py:1270, in SyncAPIClient.post(self, path, cast_to, body, options, files, stream, stream_cls)
       1256 def post(
       1257     self,
       1258     path: str,
       (...)
       1265     stream_cls: type[_StreamT] | None = None,
       1266 ) -> ResponseT | _StreamT:
       1267     opts = FinalRequestOptions.construct(
       1268         method="post", url=path, json_data=body, files=to_httpx_files(files), **options
       1269     )
    -> 1270     return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_base_client.py:947, in SyncAPIClient.request(self, cast_to, options, remaining_retries, stream, stream_cls)
        944 else:
        945     retries_taken = 0
    --> 947 return self._request(
        948     cast_to=cast_to,
        949     options=options,
        950     stream=stream,
        951     stream_cls=stream_cls,
        952     retries_taken=retries_taken,
        953 )
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_base_client.py:1051, in SyncAPIClient._request(self, cast_to, options, retries_taken, stream, stream_cls)
       1048         err.response.read()
       1050     log.debug("Re-raising status error")
    -> 1051     raise self._make_status_error_from_response(err.response) from None
       1053 return self._process_response(
       1054     cast_to=cast_to,
       1055     options=options,
       (...)
       1059     retries_taken=retries_taken,
       1060 )
    

    BadRequestError: Error code: 400 - {'error': {'message': "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. The following tool_call_ids did not have response messages: call_mjLuNyXNHoUIXHiBtXhaWdxN", 'type': 'invalid_request_error', 'param': 'messages', 'code': None}}


If we add a second response, the call will succeed as expected because we now have one tool response per tool call:


```python
tool_response_2 = foo_tool.invoke(response_message.tool_calls[1])

chat_history.append(tool_response_2)

model_with_tools.invoke(chat_history)
```




    AIMessage(content='Both calls to the tool "foo" have been completed successfully. The output for each call is "action complete!".', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 24, 'prompt_tokens': 137, 'total_tokens': 161, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_e2bde53e6e', 'finish_reason': 'stop', 'logprobs': None}, id='run-b5ac3c54-4e26-4da4-853a-d0ab1cba90e0-0', usage_metadata={'input_tokens': 137, 'output_tokens': 24, 'total_tokens': 161, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})



But if we add a duplicate, extra tool response, the call will fail again:


```python
duplicate_tool_response_2 = foo_tool.invoke(response_message.tool_calls[1])

chat_history.append(duplicate_tool_response_2)

await model_with_tools.invoke(chat_history)
```


    ---------------------------------------------------------------------------

    BadRequestError                           Traceback (most recent call last)

    Cell In[7], line 5
          1 duplicate_tool_response_2 = foo_tool.invoke(response_message.tool_calls[1])
          3 chat_history.append(duplicate_tool_response_2)
    ----> 5 await model_with_tools.invoke(chat_history)
    

    File ~/langchain/oss-py/libs/core/langchain_core/runnables/base.py:5354, in RunnableBindingBase.invoke(self, input, config, **kwargs)
       5348 def invoke(
       5349     self,
       5350     input: Input,
       5351     config: Optional[RunnableConfig] = None,
       5352     **kwargs: Optional[Any],
       5353 ) -> Output:
    -> 5354     return self.bound.invoke(
       5355         input,
       5356         self._merge_configs(config),
       5357         **{**self.kwargs, **kwargs},
       5358     )
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:286, in BaseChatModel.invoke(self, input, config, stop, **kwargs)
        275 def invoke(
        276     self,
        277     input: LanguageModelInput,
       (...)
        281     **kwargs: Any,
        282 ) -> BaseMessage:
        283     config = ensure_config(config)
        284     return cast(
        285         ChatGeneration,
    --> 286         self.generate_prompt(
        287             [self._convert_input(input)],
        288             stop=stop,
        289             callbacks=config.get("callbacks"),
        290             tags=config.get("tags"),
        291             metadata=config.get("metadata"),
        292             run_name=config.get("run_name"),
        293             run_id=config.pop("run_id", None),
        294             **kwargs,
        295         ).generations[0][0],
        296     ).message
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:786, in BaseChatModel.generate_prompt(self, prompts, stop, callbacks, **kwargs)
        778 def generate_prompt(
        779     self,
        780     prompts: list[PromptValue],
       (...)
        783     **kwargs: Any,
        784 ) -> LLMResult:
        785     prompt_messages = [p.to_messages() for p in prompts]
    --> 786     return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:643, in BaseChatModel.generate(self, messages, stop, callbacks, tags, metadata, run_name, run_id, **kwargs)
        641         if run_managers:
        642             run_managers[i].on_llm_error(e, response=LLMResult(generations=[]))
    --> 643         raise e
        644 flattened_outputs = [
        645     LLMResult(generations=[res.generations], llm_output=res.llm_output)  # type: ignore[list-item]
        646     for res in results
        647 ]
        648 llm_output = self._combine_llm_outputs([res.llm_output for res in results])
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:633, in BaseChatModel.generate(self, messages, stop, callbacks, tags, metadata, run_name, run_id, **kwargs)
        630 for i, m in enumerate(messages):
        631     try:
        632         results.append(
    --> 633             self._generate_with_cache(
        634                 m,
        635                 stop=stop,
        636                 run_manager=run_managers[i] if run_managers else None,
        637                 **kwargs,
        638             )
        639         )
        640     except BaseException as e:
        641         if run_managers:
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:851, in BaseChatModel._generate_with_cache(self, messages, stop, run_manager, **kwargs)
        849 else:
        850     if inspect.signature(self._generate).parameters.get("run_manager"):
    --> 851         result = self._generate(
        852             messages, stop=stop, run_manager=run_manager, **kwargs
        853         )
        854     else:
        855         result = self._generate(messages, stop=stop, **kwargs)
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/langchain_openai/chat_models/base.py:686, in BaseChatOpenAI._generate(self, messages, stop, run_manager, **kwargs)
        684     generation_info = {"headers": dict(raw_response.headers)}
        685 else:
    --> 686     response = self.client.create(**payload)
        687 return self._create_chat_result(response, generation_info)
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_utils/_utils.py:274, in required_args.<locals>.inner.<locals>.wrapper(*args, **kwargs)
        272             msg = f"Missing required argument: {quote(missing[0])}"
        273     raise TypeError(msg)
    --> 274 return func(*args, **kwargs)
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/resources/chat/completions.py:742, in Completions.create(self, messages, model, frequency_penalty, function_call, functions, logit_bias, logprobs, max_completion_tokens, max_tokens, metadata, n, parallel_tool_calls, presence_penalty, response_format, seed, service_tier, stop, store, stream, stream_options, temperature, tool_choice, tools, top_logprobs, top_p, user, extra_headers, extra_query, extra_body, timeout)
        704 @required_args(["messages", "model"], ["messages", "model", "stream"])
        705 def create(
        706     self,
       (...)
        739     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        740 ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        741     validate_response_format(response_format)
    --> 742     return self._post(
        743         "/chat/completions",
        744         body=maybe_transform(
        745             {
        746                 "messages": messages,
        747                 "model": model,
        748                 "frequency_penalty": frequency_penalty,
        749                 "function_call": function_call,
        750                 "functions": functions,
        751                 "logit_bias": logit_bias,
        752                 "logprobs": logprobs,
        753                 "max_completion_tokens": max_completion_tokens,
        754                 "max_tokens": max_tokens,
        755                 "metadata": metadata,
        756                 "n": n,
        757                 "parallel_tool_calls": parallel_tool_calls,
        758                 "presence_penalty": presence_penalty,
        759                 "response_format": response_format,
        760                 "seed": seed,
        761                 "service_tier": service_tier,
        762                 "stop": stop,
        763                 "store": store,
        764                 "stream": stream,
        765                 "stream_options": stream_options,
        766                 "temperature": temperature,
        767                 "tool_choice": tool_choice,
        768                 "tools": tools,
        769                 "top_logprobs": top_logprobs,
        770                 "top_p": top_p,
        771                 "user": user,
        772             },
        773             completion_create_params.CompletionCreateParams,
        774         ),
        775         options=make_request_options(
        776             extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
        777         ),
        778         cast_to=ChatCompletion,
        779         stream=stream or False,
        780         stream_cls=Stream[ChatCompletionChunk],
        781     )
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_base_client.py:1270, in SyncAPIClient.post(self, path, cast_to, body, options, files, stream, stream_cls)
       1256 def post(
       1257     self,
       1258     path: str,
       (...)
       1265     stream_cls: type[_StreamT] | None = None,
       1266 ) -> ResponseT | _StreamT:
       1267     opts = FinalRequestOptions.construct(
       1268         method="post", url=path, json_data=body, files=to_httpx_files(files), **options
       1269     )
    -> 1270     return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_base_client.py:947, in SyncAPIClient.request(self, cast_to, options, remaining_retries, stream, stream_cls)
        944 else:
        945     retries_taken = 0
    --> 947 return self._request(
        948     cast_to=cast_to,
        949     options=options,
        950     stream=stream,
        951     stream_cls=stream_cls,
        952     retries_taken=retries_taken,
        953 )
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_base_client.py:1051, in SyncAPIClient._request(self, cast_to, options, retries_taken, stream, stream_cls)
       1048         err.response.read()
       1050     log.debug("Re-raising status error")
    -> 1051     raise self._make_status_error_from_response(err.response) from None
       1053 return self._process_response(
       1054     cast_to=cast_to,
       1055     options=options,
       (...)
       1059     retries_taken=retries_taken,
       1060 )
    

    BadRequestError: Error code: 400 - {'error': {'message': "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.", 'type': 'invalid_request_error', 'param': 'messages.[4].role', 'code': None}}


You should additionally not pass `ToolMessages` back to to a model if they are not preceded by an `AIMessage` with tool calls. For example, this will fail:


```python
model_with_tools.invoke(
    [ToolMessage(content="action completed!", tool_call_id="dummy")]
)
```


    ---------------------------------------------------------------------------

    BadRequestError                           Traceback (most recent call last)

    Cell In[8], line 1
    ----> 1 model_with_tools.invoke([ToolMessage(content="action completed!", tool_call_id="dummy")])
    

    File ~/langchain/oss-py/libs/core/langchain_core/runnables/base.py:5354, in RunnableBindingBase.invoke(self, input, config, **kwargs)
       5348 def invoke(
       5349     self,
       5350     input: Input,
       5351     config: Optional[RunnableConfig] = None,
       5352     **kwargs: Optional[Any],
       5353 ) -> Output:
    -> 5354     return self.bound.invoke(
       5355         input,
       5356         self._merge_configs(config),
       5357         **{**self.kwargs, **kwargs},
       5358     )
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:286, in BaseChatModel.invoke(self, input, config, stop, **kwargs)
        275 def invoke(
        276     self,
        277     input: LanguageModelInput,
       (...)
        281     **kwargs: Any,
        282 ) -> BaseMessage:
        283     config = ensure_config(config)
        284     return cast(
        285         ChatGeneration,
    --> 286         self.generate_prompt(
        287             [self._convert_input(input)],
        288             stop=stop,
        289             callbacks=config.get("callbacks"),
        290             tags=config.get("tags"),
        291             metadata=config.get("metadata"),
        292             run_name=config.get("run_name"),
        293             run_id=config.pop("run_id", None),
        294             **kwargs,
        295         ).generations[0][0],
        296     ).message
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:786, in BaseChatModel.generate_prompt(self, prompts, stop, callbacks, **kwargs)
        778 def generate_prompt(
        779     self,
        780     prompts: list[PromptValue],
       (...)
        783     **kwargs: Any,
        784 ) -> LLMResult:
        785     prompt_messages = [p.to_messages() for p in prompts]
    --> 786     return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:643, in BaseChatModel.generate(self, messages, stop, callbacks, tags, metadata, run_name, run_id, **kwargs)
        641         if run_managers:
        642             run_managers[i].on_llm_error(e, response=LLMResult(generations=[]))
    --> 643         raise e
        644 flattened_outputs = [
        645     LLMResult(generations=[res.generations], llm_output=res.llm_output)  # type: ignore[list-item]
        646     for res in results
        647 ]
        648 llm_output = self._combine_llm_outputs([res.llm_output for res in results])
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:633, in BaseChatModel.generate(self, messages, stop, callbacks, tags, metadata, run_name, run_id, **kwargs)
        630 for i, m in enumerate(messages):
        631     try:
        632         results.append(
    --> 633             self._generate_with_cache(
        634                 m,
        635                 stop=stop,
        636                 run_manager=run_managers[i] if run_managers else None,
        637                 **kwargs,
        638             )
        639         )
        640     except BaseException as e:
        641         if run_managers:
    

    File ~/langchain/oss-py/libs/core/langchain_core/language_models/chat_models.py:851, in BaseChatModel._generate_with_cache(self, messages, stop, run_manager, **kwargs)
        849 else:
        850     if inspect.signature(self._generate).parameters.get("run_manager"):
    --> 851         result = self._generate(
        852             messages, stop=stop, run_manager=run_manager, **kwargs
        853         )
        854     else:
        855         result = self._generate(messages, stop=stop, **kwargs)
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/langchain_openai/chat_models/base.py:686, in BaseChatOpenAI._generate(self, messages, stop, run_manager, **kwargs)
        684     generation_info = {"headers": dict(raw_response.headers)}
        685 else:
    --> 686     response = self.client.create(**payload)
        687 return self._create_chat_result(response, generation_info)
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_utils/_utils.py:274, in required_args.<locals>.inner.<locals>.wrapper(*args, **kwargs)
        272             msg = f"Missing required argument: {quote(missing[0])}"
        273     raise TypeError(msg)
    --> 274 return func(*args, **kwargs)
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/resources/chat/completions.py:742, in Completions.create(self, messages, model, frequency_penalty, function_call, functions, logit_bias, logprobs, max_completion_tokens, max_tokens, metadata, n, parallel_tool_calls, presence_penalty, response_format, seed, service_tier, stop, store, stream, stream_options, temperature, tool_choice, tools, top_logprobs, top_p, user, extra_headers, extra_query, extra_body, timeout)
        704 @required_args(["messages", "model"], ["messages", "model", "stream"])
        705 def create(
        706     self,
       (...)
        739     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        740 ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        741     validate_response_format(response_format)
    --> 742     return self._post(
        743         "/chat/completions",
        744         body=maybe_transform(
        745             {
        746                 "messages": messages,
        747                 "model": model,
        748                 "frequency_penalty": frequency_penalty,
        749                 "function_call": function_call,
        750                 "functions": functions,
        751                 "logit_bias": logit_bias,
        752                 "logprobs": logprobs,
        753                 "max_completion_tokens": max_completion_tokens,
        754                 "max_tokens": max_tokens,
        755                 "metadata": metadata,
        756                 "n": n,
        757                 "parallel_tool_calls": parallel_tool_calls,
        758                 "presence_penalty": presence_penalty,
        759                 "response_format": response_format,
        760                 "seed": seed,
        761                 "service_tier": service_tier,
        762                 "stop": stop,
        763                 "store": store,
        764                 "stream": stream,
        765                 "stream_options": stream_options,
        766                 "temperature": temperature,
        767                 "tool_choice": tool_choice,
        768                 "tools": tools,
        769                 "top_logprobs": top_logprobs,
        770                 "top_p": top_p,
        771                 "user": user,
        772             },
        773             completion_create_params.CompletionCreateParams,
        774         ),
        775         options=make_request_options(
        776             extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
        777         ),
        778         cast_to=ChatCompletion,
        779         stream=stream or False,
        780         stream_cls=Stream[ChatCompletionChunk],
        781     )
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_base_client.py:1270, in SyncAPIClient.post(self, path, cast_to, body, options, files, stream, stream_cls)
       1256 def post(
       1257     self,
       1258     path: str,
       (...)
       1265     stream_cls: type[_StreamT] | None = None,
       1266 ) -> ResponseT | _StreamT:
       1267     opts = FinalRequestOptions.construct(
       1268         method="post", url=path, json_data=body, files=to_httpx_files(files), **options
       1269     )
    -> 1270     return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_base_client.py:947, in SyncAPIClient.request(self, cast_to, options, remaining_retries, stream, stream_cls)
        944 else:
        945     retries_taken = 0
    --> 947 return self._request(
        948     cast_to=cast_to,
        949     options=options,
        950     stream=stream,
        951     stream_cls=stream_cls,
        952     retries_taken=retries_taken,
        953 )
    

    File ~/langchain/oss-py/docs/.venv/lib/python3.11/site-packages/openai/_base_client.py:1051, in SyncAPIClient._request(self, cast_to, options, retries_taken, stream, stream_cls)
       1048         err.response.read()
       1050     log.debug("Re-raising status error")
    -> 1051     raise self._make_status_error_from_response(err.response) from None
       1053 return self._process_response(
       1054     cast_to=cast_to,
       1055     options=options,
       (...)
       1059     retries_taken=retries_taken,
       1060 )
    

    BadRequestError: Error code: 400 - {'error': {'message': "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.", 'type': 'invalid_request_error', 'param': 'messages.[0].role', 'code': None}}


See [this guide](/docs/how_to/tool_results_pass_to_model/) for more details on tool calling.

## Troubleshooting

The following may help resolve this error:

- If you are using a custom executor rather than a prebuilt one like LangGraph's [`ToolNode`](https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph_prebuilt.ToolNode.html)
  or the legacy LangChain [AgentExecutor](/docs/how_to/agent_executor), verify that you are invoking and returning the result for one tool per tool call.
- If you are using [few-shot tool call examples](/docs/how_to/tools_few_shot) with messages that you manually create, and you want to simulate a failure,
  you still need to pass back a `ToolMessage` whose content indicates that failure.

