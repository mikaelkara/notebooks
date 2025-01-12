## Getting started with LangChain and Gemma, running locally or in the Cloud

### Installing dependencies


```python
!pip install --upgrade langchain langchain-google-vertexai
```

### Running the model

Go to the VertexAI Model Garden on Google Cloud [console](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/335), and deploy the desired version of Gemma to VertexAI. It will take a few minutes, and after the endpoint it ready, you need to copy its number.


```python
# @title Basic parameters
project: str = "PUT_YOUR_PROJECT_ID_HERE"  # @param {type:"string"}
endpoint_id: str = "PUT_YOUR_ENDPOINT_ID_HERE"  # @param {type:"string"}
location: str = "PUT_YOUR_ENDPOINT_LOCAtION_HERE"  # @param {type:"string"}
```


```python
from langchain_google_vertexai import (
    GemmaChatVertexAIModelGarden,
    GemmaVertexAIModelGarden,
)
```

    2024-02-27 17:15:10.457149: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-27 17:15:10.508925: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-02-27 17:15:10.508957: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-02-27 17:15:10.510289: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-02-27 17:15:10.518898: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    


```python
llm = GemmaVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)
```


```python
output = llm.invoke("What is the meaning of life?")
print(output)
```

    Prompt:
    What is the meaning of life?
    Output:
     Who am I? Why do I exist? These are questions I have struggled with
    

We can also use Gemma as a multi-turn chat model:


```python
from langchain_core.messages import HumanMessage

llm = GemmaChatVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

message1 = HumanMessage(content="How much is 2+2?")
answer1 = llm.invoke([message1])
print(answer1)

message2 = HumanMessage(content="How much is 3+3?")
answer2 = llm.invoke([message1, answer1, message2])

print(answer2)
```

    content='Prompt:\n<start_of_turn>user\nHow much is 2+2?<end_of_turn>\n<start_of_turn>model\nOutput:\n8-years old.<end_of_turn>\n\n<start_of'
    content='Prompt:\n<start_of_turn>user\nHow much is 2+2?<end_of_turn>\n<start_of_turn>model\nPrompt:\n<start_of_turn>user\nHow much is 2+2?<end_of_turn>\n<start_of_turn>model\nOutput:\n8-years old.<end_of_turn>\n\n<start_of<end_of_turn>\n<start_of_turn>user\nHow much is 3+3?<end_of_turn>\n<start_of_turn>model\nOutput:\nOutput:\n3-years old.<end_of_turn>\n\n<'
    

You can post-process response to avoid repetitions:


```python
answer1 = llm.invoke([message1], parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], parse_response=True)

print(answer2)
```

    content='Output:\n<<humming>>: 2+2 = 4.\n<end'
    content='Output:\nOutput:\n<<humming>>: 3+3 = 6.'
    

## Running Gemma locally from Kaggle

In order to run Gemma locally, you can download it from Kaggle first. In order to do this, you'll need to login into the Kaggle platform, create a API key and download a `kaggle.json` Read more about Kaggle auth [here](https://www.kaggle.com/docs/api).

### Installation


```python
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/kaggle.json
```

    /opt/conda/lib/python3.10/pty.py:89: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      pid, fd = os.forkpty()
    


```python
!pip install keras>=3 keras_nlp
```

    /opt/conda/lib/python3.10/pty.py:89: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      pid, fd = os.forkpty()
    

    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorstore 0.1.54 requires ml-dtypes>=0.3.1, but you have ml-dtypes 0.2.0 which is incompatible.[0m[31m
    [0m

### Usage


```python
from langchain_google_vertexai import GemmaLocalKaggle
```

    2024-02-27 16:38:40.797559: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-27 16:38:40.848444: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-02-27 16:38:40.848478: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-02-27 16:38:40.849728: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-02-27 16:38:40.857936: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    

You can specify the keras backend (by default it's `tensorflow`, but you can change it be `jax` or `torch`).


```python
# @title Basic parameters
keras_backend: str = "jax"  # @param {type:"string"}
model_name: str = "gemma_2b_en"  # @param {type:"string"}
```


```python
llm = GemmaLocalKaggle(model_name=model_name, keras_backend=keras_backend)
```

    2024-02-27 16:23:14.661164: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 20549 MB memory:  -> device: 0, name: NVIDIA L4, pci bus id: 0000:00:03.0, compute capability: 8.9
    normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
    


```python
output = llm.invoke("What is the meaning of life?", max_tokens=30)
print(output)
```

    W0000 00:00:1709051129.518076  774855 graph_launch.cc:671] Fallback to op-by-op mode because memset node breaks graph update
    

    What is the meaning of life?
    
    The question is one of the most important questions in the world.
    
    It’s the question that has
    

### ChatModel

Same as above, using Gemma locally as a multi-turn chat model. You might need to re-start the notebook and clean your GPU memory in order to avoid OOM errors:


```python
from langchain_google_vertexai import GemmaChatLocalKaggle
```

    2024-02-27 16:58:22.331067: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-27 16:58:22.382948: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-02-27 16:58:22.382978: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-02-27 16:58:22.384312: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-02-27 16:58:22.392767: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    


```python
# @title Basic parameters
keras_backend: str = "jax"  # @param {type:"string"}
model_name: str = "gemma_2b_en"  # @param {type:"string"}
```


```python
llm = GemmaChatLocalKaggle(model_name=model_name, keras_backend=keras_backend)
```

    2024-02-27 16:58:29.001922: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 20549 MB memory:  -> device: 0, name: NVIDIA L4, pci bus id: 0000:00:03.0, compute capability: 8.9
    normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
    


```python
from langchain_core.messages import HumanMessage

message1 = HumanMessage(content="Hi! Who are you?")
answer1 = llm.invoke([message1], max_tokens=30)
print(answer1)
```

    2024-02-27 16:58:49.848412: I external/local_xla/xla/service/service.cc:168] XLA service 0x55adc0cf2c10 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2024-02-27 16:58:49.848458: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA L4, Compute Capability 8.9
    2024-02-27 16:58:50.116614: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
    2024-02-27 16:58:54.389324: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8900
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1709053145.225207  784891 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
    W0000 00:00:1709053145.284227  784891 graph_launch.cc:671] Fallback to op-by-op mode because memset node breaks graph update
    

    content="<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\nI'm a model.\n Tampoco\nI'm a model."
    


```python
message2 = HumanMessage(content="What can you help me with?")
answer2 = llm.invoke([message1, answer1, message2], max_tokens=60)

print(answer2)
```

    content="<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\n<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\nI'm a model.\n Tampoco\nI'm a model.<end_of_turn>\n<start_of_turn>user\nWhat can you help me with?<end_of_turn>\n<start_of_turn>model"
    

You can post-process the response if you want to avoid multi-turn statements:


```python
answer1 = llm.invoke([message1], max_tokens=30, parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], max_tokens=60, parse_response=True)
print(answer2)
```

    content="I'm a model.\n Tampoco\nI'm a model."
    content='I can help you with your modeling.\n Tampoco\nI can'
    

## Running Gemma locally from HuggingFace


```python
from langchain_google_vertexai import GemmaChatLocalHF, GemmaLocalHF
```

    2024-02-27 17:02:21.832409: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-27 17:02:21.883625: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-02-27 17:02:21.883656: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-02-27 17:02:21.884987: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-02-27 17:02:21.893340: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    


```python
# @title Basic parameters
hf_access_token: str = "PUT_YOUR_TOKEN_HERE"  # @param {type:"string"}
model_name: str = "google/gemma-2b"  # @param {type:"string"}
```


```python
llm = GemmaLocalHF(model_name="google/gemma-2b", hf_access_token=hf_access_token)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]



```python
output = llm.invoke("What is the meaning of life?", max_tokens=50)
print(output)
```

    What is the meaning of life?
    
    The question is one of the most important questions in the world.
    
    It’s the question that has been asked by philosophers, theologians, and scientists for centuries.
    
    And it’s the question that
    

Same as above, using Gemma locally as a multi-turn chat model. You might need to re-start the notebook and clean your GPU memory in order to avoid OOM errors:


```python
llm = GemmaChatLocalHF(model_name=model_name, hf_access_token=hf_access_token)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]



```python
from langchain_core.messages import HumanMessage

message1 = HumanMessage(content="Hi! Who are you?")
answer1 = llm.invoke([message1], max_tokens=60)
print(answer1)
```

    content="<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\nI'm a model.\n<end_of_turn>\n<start_of_turn>user\nWhat do you mean"
    


```python
message2 = HumanMessage(content="What can you help me with?")
answer2 = llm.invoke([message1, answer1, message2], max_tokens=140)

print(answer2)
```

    content="<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\n<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\nI'm a model.\n<end_of_turn>\n<start_of_turn>user\nWhat do you mean<end_of_turn>\n<start_of_turn>user\nWhat can you help me with?<end_of_turn>\n<start_of_turn>model\nI can help you with anything.\n<"
    

And the same with posprocessing:


```python
answer1 = llm.invoke([message1], max_tokens=60, parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], max_tokens=120, parse_response=True)
print(answer2)
```

    content="I'm a model.\n<end_of_turn>\n"
    content='I can help you with anything.\n<end_of_turn>\n<end_of_turn>\n'
    


```python

```
