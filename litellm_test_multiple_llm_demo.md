```python
!pip install litellm
```


```python
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["COHERE_API_KEY"] = "cohere key"
os.environ["REPLICATE_API_KEY"] = "replicate key"
messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)

# replicate call
response = completion("replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1", messages)
```
