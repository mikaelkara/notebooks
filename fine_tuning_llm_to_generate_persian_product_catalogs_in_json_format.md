# Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format

_Authored by: [Mohammadreza Esmaeiliyan](https://github.com/MrzEsma)_

In this notebook, we have attempted to fine-tune a large language model with no added complexity. The model has been optimized for use on a customer-level GPU to generate Persian product catalogs and produce structured output in JSON format. It is particularly effective for creating structured outputs from the unstructured titles and descriptions of products on Iranian platforms with user-generated content, such as [Basalam](https://basalam.com), [Divar](https://divar.ir/), [Digikala](https://www.digikala.com/), and others. 

You can see a fine-tuned LLM with this code on [our HF account](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v1). Additionally, one of the fastest open-source inference engines, [Vllm](https://github.com/vllm-project/vllm), is employed for inference. 

Let's get started!


```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
```

The `peft` library, or parameter efficient fine tuning, has been created to fine-tune LLMs more efficiently. If we were to open and fine-tune the upper layers of the network traditionally like all neural networks, it would require a lot of processing and also a significant amount of VRAM. With the methods developed in recent papers, this library has been implemented for efficient fine-tuning of LLMs. Read more about peft here: [Hugging Face PEFT](https://huggingface.co/blog/peft).

## Set hyperparameters


```python
# General parameters
model_name = "NousResearch/Llama-2-7b-chat-hf"  # The model that you want to train from the Hugging Face hub
dataset_name = "BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1"  # The instruction dataset to use
new_model = "llama-persian-catalog-generator"  # The name for fine-tuned LoRA Adaptor
```


```python
# LoRA parameters
lora_r = 64
lora_alpha = lora_r * 2
lora_dropout = 0.1
target_modules = ["q_proj", "v_proj", 'k_proj']

```

LoRA (Low-Rank Adaptation) stores changes in weights by constructing and adding a low-rank matrix to each model layer. This method opens only these layers for fine-tuning, without changing the original model weights or requiring lengthy training. The resulting weights are lightweight and can be produced multiple times, allowing for the fine-tuning of multiple tasks with an LLM loaded into RAM. 


Read about LoRA [here at Lightning AI](https://lightning.ai/pages/community/tutorial/lora-llm/). For other efficient training methods, see [Hugging Face Docs on Performance Training](https://huggingface.co/docs/transformers/perf_train_gpu_one) and [SFT Trainer Enhancement](https://huggingface.co/docs/trl/main/en/sft_trainer#enhance-models-performances-using-neftune).



```python
# QLoRA parameters
load_in_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = False

```

QLoRA (Quantized Low-Rank Adaptation) is an efficient fine-tuning approach that enables large language models to run on smaller GPUs by using 4-bit quantization. This method preserves the full performance of 16-bit fine-tuning while reducing memory usage, making it possible to fine-tune models with up to 65 billion parameters on a single 48GB GPU. QLoRA combines 4-bit NormalFloat data types, double quantization, and paged optimizers to manage memory efficiently. It allows fine-tuning of models with low-rank adapters, significantly enhancing accessibility for AI model development.

Read about QLoRA [here at Hugging Face](https://huggingface.co/blog/4bit-transformers-bitsandbytes).


```python
# TrainingArguments parameters
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
learning_rate = 0.00015
weight_decay = 0.01
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25

# SFT parameters
max_seq_length = None
packing = False
device_map = {"": 0}

# Dataset parameters
use_special_template = True
response_template = ' ### Answer:'
instruction_prompt_template = '"### Human:"'
use_llama_like_model = True

```

## Model Training


```python
# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")
percent_of_train_dataset = 0.95
other_columns = [i for i in dataset.column_names if i not in ['instruction', 'output']]
dataset = dataset.remove_columns(other_columns)
split_dataset = dataset.train_test_split(train_size=int(dataset.num_rows * percent_of_train_dataset), seed=19, shuffle=False)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(eval_dataset)}")
```


```python
# Load LoRA configuration
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules
)
```

The LoraConfig object is used to configure the LoRA (Low-Rank Adaptation) settings for the model when using the Peft library. This can help to reduce the number of parameters that need to be fine-tuned, which can lead to faster training and lower memory usage. Here's a breakdown of the parameters:
- `r`: The rank of the low-rank matrices used in LoRA. This parameter controls the dimensionality of the low-rank adaptation and directly impacts the model's capacity to adapt and the computational cost.
- `lora_alpha`: This parameter controls the scaling factor for the low-rank adaptation matrices. A higher alpha value can increase the model's capacity to learn new tasks.
- `lora_dropout`: The dropout rate for LoRA. This can help to prevent overfitting during fine-tuning. In this case, it's set to 0.1.
- `bias`: Specifies whether to add a bias term to the low-rank matrices. In this case, it's set to "none", which means that no bias term will be added.
- `task_type`: Defines the type of task for which the model is being fine-tuned. Here, "CAUSAL_LM" indicates that the task is a causal language modeling task, which predicts the next word in a sequence.
- `target_modules`: Specifies the modules in the model to which LoRA will be applied. In this case, it's set to `["q_proj", "v_proj", 'k_proj']`, which are the query, value, and key projection layers in the model's attention mechanism.


```python
# Load QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
)
```

This block configures the settings for using BitsAndBytes (bnb), a library that provides efficient memory management and compression techniques for PyTorch models. Specifically, it defines how the model weights will be loaded and quantized in 4-bit precision, which is useful for reducing memory usage and potentially speeding up inference.

- `load_in_4bit`: A boolean that determines whether to load the model in 4-bit precision.
- `bnb_4bit_quant_type`: Specifies the type of 4-bit quantization to use. Here, it's set to 4-bit NormalFloat (NF4) quantization type, which is a new data type introduced in QLoRA. This type is information-theoretically optimal for normally distributed weights, providing an efficient way to quantize the model for fine-tuning.
- `bnb_4bit_compute_dtype`: Sets the data type used for computations involving the quantized model. In QLoRA, it's set to "float16", which is commonly used for mixed-precision training to balance performance and precision.
- `bnb_4bit_use_double_quant`: This boolean parameter indicates whether to use double quantization. Setting it to False means that only single quantization will be used, which is typically faster but might be slightly less accurate.

Why we have two data type (quant_type and compute_type)? 
QLoRA employs two distinct data types: one for storing base model weights (in here 4-bit NormalFloat) and another for computational operations (16-bit). During the forward and backward passes, QLoRA dequantizes the weights from the storage format to the computational format. However, it only calculates gradients for the LoRA parameters, which utilize 16-bit bfloat. This approach ensures that weights are decompressed only when necessary, maintaining low memory usage throughout both training and inference phases.



```python
# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
```


```python
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=new_model,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    gradient_checkpointing=gradient_checkpointing,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type
)
```


```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
if not tokenizer.chat_template:
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
```

Regarding the chat template, we will briefly explain that to understand the structure of the conversation between the user and the model during model training, a series of reserved phrases are created to separate the user's message and the model's response. This ensures that the model precisely understands where each message comes from and maintains a sense of the conversational structure. Typically, adhering to a chat template helps increase accuracy in the intended task. However, when there is a distribution shift between the fine-tuning dataset and the model, using a specific chat template can be even more helpful. For further reading, visit [Hugging Face Blog on Chat Templates](https://huggingface.co/blog/chat-templates).



```python
def special_formatting_prompts(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"{instruction_prompt_template}{example['instruction'][i]}\n{response_template} {example['output'][i]}"
        output_texts.append(text)
    return output_texts


def normal_formatting_prompts(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        chat_temp = [{"role": "system", "content": example['instruction'][i]},
                     {"role": "assistant", "content": example['output'][i]}]
        text = tokenizer.apply_chat_template(chat_temp, tokenize=False)
        output_texts.append(text)
    return output_texts

```


```python
if use_special_template:
    formatting_func = special_formatting_prompts
    if use_llama_like_model:
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)
    else:
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
else:
    formatting_func = normal_formatting_prompts
```


```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    data_collator=collator,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing
)
```

The `SFTTrainer` is then instantiated to handle supervised fine-tuning (SFT) of the model. This trainer is specifically designed for SFT and includes additional parameters such as `formatting_func` and `packing` which are not typically found in standard trainers.
`formatting_func`: A custom function to format training examples by combining instruction and response templates.
`packing`: Disables packing multiple samples into one sequence, which is not a standard parameter in the typical Trainer class.



```python
# Train model
trainer.train()

# Save fine tuned Lora Adaptor 
trainer.model.save_pretrained(new_model)
```

## Inference


```python
import torch
import gc


def clear_hardwares():
    torch.clear_autocast_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()


clear_hardwares()
clear_hardwares()
```


```python
def generate(model, prompt: str, kwargs):
    tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(model.device)

    prompt_length = len(tokenized_prompt.get('input_ids')[0])

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**tokenized_prompt, **kwargs) if kwargs else model.generate(**tokenized_prompt)
        output = tokenizer.decode(output_tokens[0][prompt_length:], skip_special_tokens=True)

    return output
```


```python
base_model = AutoModelForCausalLM.from_pretrained(new_model, return_dict=True, device_map='auto', token='')
tokenizer = AutoTokenizer.from_pretrained(new_model, max_length=max_seq_length)
model = PeftModel.from_pretrained(base_model, new_model)
del base_model
```


```python
sample = eval_dataset[0]
if use_special_template:
    prompt = f"{instruction_prompt_template}{sample['instruction']}\n{response_template}"
else:
    chat_temp = [{"role": "system", "content": sample['instruction']}]
    prompt = tokenizer.apply_chat_template(chat_temp, tokenize=False, add_generation_prompt=True)
```


```python
gen_kwargs = {"max_new_tokens": 1024}
generated_texts = generate(model=model, prompt=prompt, kwargs=gen_kwargs)
print(generated_texts)
```

## Merge to base model


```python
clear_hardwares()
merged_model = model.merge_and_unload()
clear_hardwares()
del model
adapter_model_name = 'your_hf_account/your_desired_name'
merged_model.push_to_hub(adapter_model_name)
```

Here, we merged the adapter with the base model and push the merged model on the hub. You can just push the adapter in the hub and avoid pushing the heavy base model file in this way:
```
model.push_to_hub(adapter_model_name)
```
And then you load the model in this way:
```
config = PeftConfig.from_pretrained(adapter_model_name)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, adapter_model_name)
```

## Fast Inference with [Vllm](https://github.com/vllm-project/vllm)


The `vllm` library is one of the fastest inference engines for LLMs. For a comparative overview of available options, you can use this blog: [7 Frameworks for Serving LLMs](https://medium.com/@gsuresh957/7-frameworks-for-serving-llms-5044b533ee88). 
In this example, we are inferring version 1 of our fine-tuned model on this task.


```python
from vllm import LLM, SamplingParams

prompt = """### Question: here is a product title from a Iranian marketplace.  \n         give me the Product Entity and Attributes of this product in Persian language.\n         give the output in this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': '<product entity>'}.\n         Don't make assumptions about what values to plug into json. Just give Json not a single word more.\n         \nproduct title:"""
user_prompt_template = '### Question: '
response_template = ' ### Answer:'

llm = LLM(model='BaSalam/Llama2-7b-entity-attr-v1', gpu_memory_utilization=0.9, trust_remote_code=True)

product = 'مانتو اسپرت پانیذ قد جلوی کار حدودا 85 سانتی متر قد پشت کار حدودا 88 سانتی متر'
sampling_params = SamplingParams(temperature=0.0, max_tokens=75)
prompt = f'{user_prompt_template} {prompt}{product}\n {response_template}'
outputs = llm.generate(prompt, sampling_params)

print(outputs[0].outputs[0].text)
```

### Example Output

```
{
    "attributes": {
        "قد جلوی کار": "85 سانتی متر",
        "قد پشت کار": "88 سانتی متر"
    },
    "product_entity": "مانتو اسپرت"
}
```


In this blog, you can read about the best practices for fine-tuning LLMs [Sebastian Raschka's Magazine](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms?r=1h0eu9&utm_campaign=post&utm_medium=web). 

