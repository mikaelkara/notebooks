# LiteLLM - Azure OpenAI + OpenAI Calls
This notebook covers the following for Azure OpenAI + OpenAI:
* Completion - Quick start
* Completion - Streaming
* Completion - Azure, OpenAI in separate threads
* Completion - Stress Test 10 requests in parallel
* Completion - Azure, OpenAI in the same thread


```python
!pip install litellm
```


```python
import os, litellm
```

## Completion - Quick start


```python
import os
from litellm import completion

# openai configs
os.environ["OPENAI_API_KEY"] = ""

# azure openai configs
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = "https://openai-gpt-4-test-v-1.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2023-05-15"


# openai call
response = completion(
    model = "gpt-3.5-turbo",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
print("Openai Response\n")
print(response)



# azure call
response = completion(
    model = "azure/your-azure-deployment",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
print("Azure Response\n")
print(response)
```

    Openai Response
    
    {
      "id": "chatcmpl-7yjVOEKCPw2KdkfIaM3Ao1tIXp8EM",
      "object": "chat.completion",
      "created": 1694708958,
      "model": "gpt-3.5-turbo-0613",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "I'm an AI, so I don't have feelings, but I'm here to help you. How can I assist you?"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 26,
        "total_tokens": 39
      }
    }
    Azure Response
    
    {
      "id": "chatcmpl-7yjVQ6m2R2HRtnKHRRFp6JzL4Fjez",
      "object": "chat.completion",
      "created": 1694708960,
      "model": "gpt-35-turbo",
      "choices": [
        {
          "index": 0,
          "finish_reason": "stop",
          "message": {
            "role": "assistant",
            "content": "Hello there! As an AI language model, I don't have feelings but I'm functioning well. How can I assist you today?"
          }
        }
      ],
      "usage": {
        "completion_tokens": 27,
        "prompt_tokens": 14,
        "total_tokens": 41
      }
    }
    

## Completion - Streaming


```python
import os
from litellm import completion

# openai configs
os.environ["OPENAI_API_KEY"] = ""

# azure openai configs
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = "https://openai-gpt-4-test-v-1.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2023-05-15"


# openai call
response = completion(
    model = "gpt-3.5-turbo",
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=True
)
print("OpenAI Streaming response")
for chunk in response:
  print(chunk)

# azure call
response = completion(
    model = "azure/your-azure-deployment",
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=True
)
print("Azure Streaming response")
for chunk in response:
  print(chunk)

```

## Completion - Azure, OpenAI in separate threads


```python
import os
import threading
from litellm import completion

# Function to make a completion call
def make_completion(model, messages):
    response = completion(
        model=model,
        messages=messages
    )

    print(f"Response for {model}: {response}")

# openai configs
os.environ["OPENAI_API_KEY"] = ""

# azure openai configs
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = "https://openai-gpt-4-test-v-1.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2023-05-15"

# Define the messages for the completions
messages = [{"content": "Hello, how are you?", "role": "user"}]

# Create threads for making the completions
thread1 = threading.Thread(target=make_completion, args=("gpt-3.5-turbo", messages))
thread2 = threading.Thread(target=make_completion, args=("azure/your-azure-deployment", messages))

# Start both threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()

print("Both completions are done.")
```

## Completion - Stress Test 10 requests in parallel




```python
import os
import threading
from litellm import completion

# Function to make a completion call
def make_completion(model, messages):
    response = completion(
        model=model,
        messages=messages
    )

    print(f"Response for {model}: {response}")

# Set your API keys
os.environ["OPENAI_API_KEY"] = ""
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = "https://openai-gpt-4-test-v-1.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2023-05-15"

# Define the messages for the completions
messages = [{"content": "Hello, how are you?", "role": "user"}]

# Create and start 10 threads for making completions
threads = []
for i in range(10):
    thread = threading.Thread(target=make_completion, args=("gpt-3.5-turbo" if i % 2 == 0 else "azure/your-azure-deployment", messages))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All completions are done.")

```

## Completion - Azure, OpenAI in the same thread


```python
import os
from litellm import completion

# Function to make both OpenAI and Azure completions
def make_completions():
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = ""

    # OpenAI completion
    openai_response = completion(
        model="gpt-3.5-turbo",
        messages=[{"content": "Hello, how are you?", "role": "user"}]
    )

    print("OpenAI Response:", openai_response)

    # Set your Azure OpenAI API key and configuration
    os.environ["AZURE_API_KEY"] = ""
    os.environ["AZURE_API_BASE"] = "https://openai-gpt-4-test-v-1.openai.azure.com/"
    os.environ["AZURE_API_VERSION"] = "2023-05-15"

    # Azure OpenAI completion
    azure_response = completion(
        model="azure/your-azure-deployment",
        messages=[{"content": "Hello, how are you?", "role": "user"}]
    )

    print("Azure OpenAI Response:", azure_response)

# Call the function to make both completions in one thread
make_completions()

```

    OpenAI Response: {
      "id": "chatcmpl-7yjzrDeOeVeSrQ00tApmTxEww3vBS",
      "object": "chat.completion",
      "created": 1694710847,
      "model": "gpt-3.5-turbo-0613",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Hello! I'm an AI, so I don't have feelings, but I'm here to help you. How can I assist you today?"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 29,
        "total_tokens": 42
      }
    }
    Azure OpenAI Response: {
      "id": "chatcmpl-7yjztAQ0gK6IMQt7cvLroMSOoXkeu",
      "object": "chat.completion",
      "created": 1694710849,
      "model": "gpt-35-turbo",
      "choices": [
        {
          "index": 0,
          "finish_reason": "stop",
          "message": {
            "role": "assistant",
            "content": "As an AI language model, I don't have feelings but I'm functioning properly. Thank you for asking! How can I assist you today?"
          }
        }
      ],
      "usage": {
        "completion_tokens": 29,
        "prompt_tokens": 14,
        "total_tokens": 43
      }
    }
    
