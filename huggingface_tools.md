# HuggingFace Hub Tools

>[Huggingface Tools](https://huggingface.co/docs/transformers/v4.29.0/en/custom_tools) that supporting text I/O can be
loaded directly using the `load_huggingface_tool` function.


```python
# Requires transformers>=4.29.0 and huggingface_hub>=0.14.1
%pip install --upgrade --quiet  transformers huggingface_hub > /dev/null
```


```python
%pip install --upgrade --quiet  langchain-community
```


```python
from langchain_community.agent_toolkits.load_tools import load_huggingface_tool

tool = load_huggingface_tool("lysandre/hf-model-downloads")

print(f"{tool.name}: {tool.description}")
```

    model_download_counter: This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. It takes the name of the category (such as text-classification, depth-estimation, etc), and returns the name of the checkpoint
    


```python
tool.run("text-classification")
```




    'facebook/bart-large-mnli'




```python

```
