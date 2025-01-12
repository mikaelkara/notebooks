## Use LiteLLM with Langfuse
https://docs.litellm.ai/docs/observability/langfuse_integration

## Install Dependencies


```python
%pip install litellm lunary
```

## Set Env Variables


```python
import litellm
from litellm import completion
import os

# from https://app.lunary.ai/
os.environ["LUNARY_PUBLIC_KEY"] = ""


# LLM provider keys
# You can use any of the litellm supported providers: https://docs.litellm.ai/docs/providers
os.environ['OPENAI_API_KEY'] = ""

```

## Set Lunary as a callback for sending data
## OpenAI completion call


```python
# set langfuse as a callback, litellm will send the data to langfuse
litellm.success_callback = ["lunary"]

# openai call
response = completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)

print(response)
```

    [Choices(finish_reason='stop', index=0, message=Message(content='Hello! How can I assist you today?', role='assistant'))]ModelResponse(id='chatcmpl-8xIWykI0GiJSmYtXYuB8Z363kpIBm', choices=[Choices(finish_reason='stop', index=0, message=Message(content='Hello! How can I assist you today?', role='assistant'))], created=1709143276, model='gpt-3.5-turbo-0125', object='chat.completion', system_fingerprint='fp_86156a94a0', usage=Usage(completion_tokens=9, prompt_tokens=15, total_tokens=24))
    
    [Lunary] Add event: {
        "event": "start",
        "type": "llm",
        "name": "gpt-3.5-turbo",
        "runId": "a363776a-bd07-4474-bce2-193067f01b2e",
        "timestamp": "2024-02-28T18:01:15.188153+00:00",
        "input": {
            "role": "user",
            "content": "Hi \ud83d\udc4b - i'm openai"
        },
        "extra": {},
        "runtime": "litellm",
        "metadata": {}
    }
    
    
    [Lunary] Add event: {
        "event": "end",
        "type": "llm",
        "runId": "a363776a-bd07-4474-bce2-193067f01b2e",
        "timestamp": "2024-02-28T18:01:16.846581+00:00",
        "output": {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        "runtime": "litellm",
        "tokensUsage": {
            "completion": 9,
            "prompt": 15
        }
    }
    
    
    

    --- Logging error ---
    Traceback (most recent call last):
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/urllib3/connectionpool.py", line 537, in _make_request
        response = conn.getresponse()
                   ^^^^^^^^^^^^^^^^^^
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/urllib3/connection.py", line 466, in getresponse
        httplib_response = super().getresponse()
                           ^^^^^^^^^^^^^^^^^^^^^
      File "/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/client.py", line 1423, in getresponse
        response.begin()
      File "/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/client.py", line 331, in begin
        version, status, reason = self._read_status()
                                  ^^^^^^^^^^^^^^^^^^^
      File "/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/client.py", line 292, in _read_status
        line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/lib/python3.12/socket.py", line 707, in readinto
        return self._sock.recv_into(b)
               ^^^^^^^^^^^^^^^^^^^^^^^
    TimeoutError: timed out
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/requests/adapters.py", line 486, in send
        resp = conn.urlopen(
               ^^^^^^^^^^^^^
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/urllib3/connectionpool.py", line 847, in urlopen
        retries = retries.increment(
                  ^^^^^^^^^^^^^^^^^^
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/urllib3/util/retry.py", line 470, in increment
        raise reraise(type(error), error, _stacktrace)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/urllib3/util/util.py", line 39, in reraise
        raise value
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/urllib3/connectionpool.py", line 793, in urlopen
        response = self._make_request(
                   ^^^^^^^^^^^^^^^^^^^
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/urllib3/connectionpool.py", line 539, in _make_request
        self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/urllib3/connectionpool.py", line 370, in _raise_timeout
        raise ReadTimeoutError(
    urllib3.exceptions.ReadTimeoutError: HTTPConnectionPool(host='localhost', port=3333): Read timed out. (read timeout=5)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/lunary/consumer.py", line 59, in send_batch
        response = requests.post(
                   ^^^^^^^^^^^^^^
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/requests/api.py", line 115, in post
        return request("post", url, data=data, json=json, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/requests/adapters.py", line 532, in send
        raise ReadTimeout(e, request=request)
    requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=3333): Read timed out. (read timeout=5)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/lib/python3.12/logging/__init__.py", line 1160, in emit
        msg = self.format(record)
              ^^^^^^^^^^^^^^^^^^^
      File "/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/lib/python3.12/logging/__init__.py", line 999, in format
        return fmt.format(record)
               ^^^^^^^^^^^^^^^^^^
      File "/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/lib/python3.12/logging/__init__.py", line 703, in format
        record.message = record.getMessage()
                         ^^^^^^^^^^^^^^^^^^^
      File "/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/lib/python3.12/logging/__init__.py", line 392, in getMessage
        msg = msg % self.args
              ~~~~^~~~~~~~~~~
    TypeError: not all arguments converted during string formatting
    Call stack:
      File "/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/lib/python3.12/threading.py", line 1030, in _bootstrap
        self._bootstrap_inner()
      File "/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/lib/python3.12/threading.py", line 1073, in _bootstrap_inner
        self.run()
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/lunary/consumer.py", line 24, in run
        self.send_batch()
      File "/Users/vince/Library/Caches/pypoetry/virtualenvs/litellm-7WKnDWGw-py3.12/lib/python3.12/site-packages/lunary/consumer.py", line 73, in send_batch
        logging.error("[Lunary] Error sending events", e)
    Message: '[Lunary] Error sending events'
    Arguments: (ReadTimeout(ReadTimeoutError("HTTPConnectionPool(host='localhost', port=3333): Read timed out. (read timeout=5)")),)
    

# Using LiteLLM with Lunary Templates

You can use LiteLLM seamlessly with Lunary templates to manage your prompts and completions.

Assuming you have created a template "test-template" with a variable "question", you can use it like this:


```python
import lunary
from litellm import completion

template = lunary.render_template("test-template", {"question": "Hello!"})

response = completion(**template)

print(response)
```

    [Choices(finish_reason='stop', index=0, message=Message(content='Hello! How can I assist you today?', role='assistant'))]ModelResponse(id='chatcmpl-8xIXegwpudg4YKnLB6pmpFGXqTHcH', choices=[Choices(finish_reason='stop', index=0, message=Message(content='Hello! How can I assist you today?', role='assistant'))], created=1709143318, model='gpt-4-0125-preview', object='chat.completion', system_fingerprint='fp_c8aa5a06d6', usage=Usage(completion_tokens=9, prompt_tokens=21, total_tokens=30))
    
    [Lunary] Add event: {
        "event": "start",
        "type": "llm",
        "name": "gpt-4-turbo-preview",
        "runId": "3a5b698d-cb55-4b3b-ab6d-04d2b99e40cb",
        "timestamp": "2024-02-28T18:01:56.746249+00:00",
        "input": [
            {
                "role": "system",
                "content": "You are an helpful assistant."
            },
            {
                "role": "user",
                "content": "Hi! Hello!"
            }
        ],
        "extra": {
            "temperature": 1,
            "max_tokens": 100
        },
        "runtime": "litellm",
        "metadata": {}
    }
    
    
    [Lunary] Add event: {
        "event": "end",
        "type": "llm",
        "runId": "3a5b698d-cb55-4b3b-ab6d-04d2b99e40cb",
        "timestamp": "2024-02-28T18:01:58.741244+00:00",
        "output": {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        "runtime": "litellm",
        "tokensUsage": {
            "completion": 9,
            "prompt": 21
        }
    }
    
    
    
