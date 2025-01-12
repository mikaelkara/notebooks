## Using Google Palm (VertexAI) with liteLLM 
### chat-bison, chat-bison@001, text-bison, text-bison@001


```python
!pip install litellm==0.1.388
```

### Set VertexAI Configs
Vertex AI requires the following:
* `vertex_project` - Your Project ID
* `vertex_location` - Your Vertex AI region
Both can be found on: https://console.cloud.google.com/

VertexAI uses Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information on setting this up

NOTE: VertexAI requires you to set `application_default_credentials.json`, this can be set by running `gcloud auth application-default login` in your terminal




```python
# set you Vertex AI configs
import litellm
from litellm import embedding, completion

litellm.vertex_project = "hardy-device-386718"
litellm.vertex_location = "us-central1"
```

## Call VertexAI - chat-bison using liteLLM


```python
user_message = "what is liteLLM "
messages = [{ "content": user_message,"role": "user"}]

# chat-bison or chat-bison@001 supported by Vertex AI (As of Aug 2023)
response = completion(model="chat-bison", messages=messages)
print(response)
```

    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': LiteLLM LiteLLM is a large language model from Google AI that is designed to be lightweight and efficient. It is based on the Transformer architecture and has been trained on a massive dataset of text. LiteLLM is available as a pre-trained model that can be used for a variety of natural language processing tasks, such as text classification, question answering, and summarization.}}], 'created': 1692036777.831989, 'model': 'chat-bison'}
    

## Call VertexAI - text-bison using liteLLM


```python
print(litellm.vertex_text_models)
```

    ['text-bison', 'text-bison@001']
    


```python
user_message = "what is liteLLM "
messages = [{ "content": user_message,"role": "user"}]

# text-bison or text-bison@001 supported by Vertex AI (As of Aug 2023)
response = completion(model="text-bison@001", messages=messages)
print(response)
```

    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': liteLLM is a low-precision variant of the large language model LLM 5. For a given text prompt, liteLLM can continue the text in a way that is both coherent and informative.}}], 'created': 1692036813.052487, 'model': 'text-bison@001'}
    


```python
response = completion(model="text-bison", messages=messages)
print(response)
```

    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': liteLLM was originally developed by Google engineers as a lite version of LLM, which stands for large language model. It is a deep learning language model that is designed to be more efficient than traditional LLMs while still achieving comparable performance. liteLLM is built on Tensor2Tensor, a framework for building and training large neural networks. It is able to learn from massive amounts of text data and generate text that is both coherent and informative. liteLLM has been shown to be effective for a variety of tasks, including machine translation, text summarization, and question answering.}}], 'created': 1692036821.60951, 'model': 'text-bison'}
    


```python
response = completion(model="text-bison@001", messages=messages, temperature=0.4, top_k=10, top_p=0.2)
print(response['choices'][0]['message']['content'])
```

    liteLLM is a lightweight language model that is designed to be fast and efficient. It is based on the Transformer architecture, but it has been modified to reduce the number of parameters and the amount of computation required. This makes it suitable for use on devices with limited resources, such as mobile phones and embedded systems.
    
    liteLLM is still under development, but it has already been shown to be effective on a variety of tasks, including text classification, natural language inference, and machine translation. It is also being used to develop new applications, such as chatbots and language assistants.
    
    If you are interested in learning more about lite
    


```python

```
