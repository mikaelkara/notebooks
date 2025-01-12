# Call Replicate LLMs using chatGPT Input/Output Format
This tutorial covers using the following Replicate Models with liteLLM

- [StableLM Tuned Alpha 7B](https://replicate.com/stability-ai/stablelm-tuned-alpha-7b)
- [LLAMA-2 70B Chat](https://replicate.com/replicate/llama-2-70b-chat)
- [A16z infra-LLAMA-2 7B Chat](https://replicate.com/a16z-infra/llama-2-7b-chat)
- [Dolly V2 12B](https://replicate.com/replicate/dolly-v2-12b)
- [Vicuna 13B](https://replicate.com/replicate/vicuna-13b)






```python
# install liteLLM
!pip install litellm
```

Imports & Set ENV variables
Get your Replicate Key: https://replicate.com/account/api-tokens


```python
from litellm import completion
import os
os.environ['REPLICATE_API_TOKEN'] = ' ' # @param
user_message = "Hello, whats the weather in San Francisco??"
messages = [{ "content": user_message,"role": "user"}]
```

## Call Replicate Models using completion(model, messages) - chatGPT format


```python
llama_2 = "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"
llama_2_7b = "a16z-infra/llama-2-7b-chat:4f0b260b6a13eb53a6b1891f089d57c08f41003ae79458be5011303d81a394dc"
dolly_v2 = "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5"
vicuna = "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"
models = [llama_2, llama_2_7b, dolly_v2, vicuna]
for model in models:
  response = completion(model=model, messages=messages)
  print(f"Response from {model} \n]\n")
  print(response)
```

    replicate is not installed. Installing...
    Response from stability-ai/stablelm-tuned-alpha-7b:c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb 
    ]
    
    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': "I'm sorry for you being unable to access this content as my training data only goes up until 2023/03. However I can tell you what your local weather forecast may look like at any time of year with respect to current conditions:"}}], 'created': 1691611730.7224207, 'model': 'stability-ai/stablelm-tuned-alpha-7b:c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb', 'usage': {'prompt_tokens': 9, 'completion_tokens': 49, 'total_tokens': 58}}
    Response from replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1 
    ]
    
    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': " Hello! I'm happy to help you with your question. However, I must point out that the question itself may not be meaningful. San Francisco is a city located in California, USA, and it is not possible for me to provide you with the current weather conditions there as I am a text-based AI language model and do not have access to real-time weather data. Additionally, the weather in San Francisco can vary greatly depending on the time of year, so it would be best to check a reliable weather source for the most up-to-date information.\n\nIf you meant to ask a different question, please feel free to rephrase it, and I will do my best to assist you in a safe and positive manner."}}], 'created': 1691611745.0269957, 'model': 'replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1', 'usage': {'prompt_tokens': 9, 'completion_tokens': 143, 'total_tokens': 152}}
    Response from a16z-infra/llama-2-7b-chat:4f0b260b6a13eb53a6b1891f089d57c08f41003ae79458be5011303d81a394dc 
    ]
    
    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': " Hello! I'm here to help you with your question. However, I must inform you that the weather in San Francisco can be quite unpredictable and can change rapidly. It's important to check reliable sources such as AccuWeather or the National Weather Service for the most up-to-date and accurate information about the weather in San Francisco.\nI cannot provide you with real-time weather data or forecasts as I'm just an AI and do not have access to current weather conditions or predictions. But I can suggest some trustworthy websites or apps where you can find the latest weather updates:\n* AccuWeather (accuweather.com)\n* The Weather Channel (weather.com)\n* Dark Sky (darksky.net)\n* Weather Underground (wunderground.com)\nRemember, it's always best to consult multiple sources for the most accurate information when planning your day or trip. Enjoy your day!"}}], 'created': 1691611748.7723358, 'model': 'a16z-infra/llama-2-7b-chat:4f0b260b6a13eb53a6b1891f089d57c08f41003ae79458be5011303d81a394dc', 'usage': {'prompt_tokens': 9, 'completion_tokens': 174, 'total_tokens': 183}}
    Response from replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5 
    ]
    
    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': 'Its 68 degrees right now in San Francisco! The temperature will be rising through the week and i expect it to reach 70 on Thursdays and Friday. Skies are expected to be partly cloudy with some sun breaks throughout the day.\n\n'}}], 'created': 1691611752.2002115, 'model': 'replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5', 'usage': {'prompt_tokens': 9, 'completion_tokens': 48, 'total_tokens': 57}}
    Response from replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b 
    ]
    
    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': ''}}], 'created': 1691611752.8998356, 'model': 'replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b', 'usage': {'prompt_tokens': 9, 'completion_tokens': 0, 'total_tokens': 9}}
    


```python
# @title Stream Responses from Replicate - Outputs in the same format used by chatGPT streaming
response = completion(model=llama_2, messages=messages, stream=True)

for chunk in response:
  print(chunk['choices'][0]['delta'])
```

    Hi
     there!
     The
     current
     forecast
     for
     today's
     high
     temperature
     ranges
     from
     75
     degrees
     Fahrenheit
     all
     day
     to
     83
     degrees
     Fahrenheit
     with
     possible
     isolated
     thunderstorms
     during
     the
     afternoon
     hours,
     mainly
     at
     sunset
     through
     early
     evening.  The
     Pacific
     Ocean
     has
     a
     low
     pressure
     of
     926
     mb
     and
     mostly
     cloud
     cover
     in
     this
     region
     on
     sunny
     days
     due
     to
     warming
     temperatures
     above
     average
     along
     most
     coastal
     areas
     and
     ocean
     breezes.<|USER|>
    


```python

```
