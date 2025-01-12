# Fine-tuning a Code LLM on Custom Code on a single GPU

_Authored by: [Maria Khalusova](https://github.com/MKhalusova)_

Publicly available code LLMs such as Codex, StarCoder, and Code Llama are great at generating code that adheres to general programming principles and syntax, but they may not align with an organization's internal conventions, or be aware of proprietary libraries.

In this notebook, we'll see show how you can fine-tune a code LLM on private code bases to enhance its contextual awareness and improve a model's usefulness to your organization's needs. Since the code LLMs are quite large, fine-tuning them in a traditional manner can be resource-draining. Worry not! We will show how you can optimize fine-tuning to fit on a single GPU.


## Dataset

For this example, we picked the top 10 Hugging Face public repositories on GitHub. We have excluded non-code files from the data, such as images, audio files, presentations, and so on. For Jupyter notebooks, we've kept only cells containing code. The resulting code is stored as a dataset that you can find on the Hugging Face Hub under [`smangrul/hf-stack-v1`](https://huggingface.co/datasets/smangrul/hf-stack-v1). It contains repo id, file path, and file content.


## Model

We'll finetune [`bigcode/starcoderbase-1b`](https://huggingface.co/bigcode/starcoderbase-1b), which is a 1B parameter model trained on 80+ programming languages. This is a gated model, so if you plan to run this notebook with this exact model, you'll need to gain access to it on the model's page. Log in to your Hugging Face account to do so:


```python
from huggingface_hub import notebook_login

notebook_login()
```

To get started, let's install all the necessary libraries. As you can see, in addition to `transformers` and `datasets`, we'll be using `peft`, `bitsandbytes`, and `flash-attn` to optimize the training.

By employing parameter-efficient training techniques, we can run this notebook on a single A100 High-RAM GPU.


```python
!pip install -q transformers datasets peft bitsandbytes flash-attn
```

Let's define some variables now. Feel free to play with these.


```python
MODEL="bigcode/starcoderbase-1b" # Model checkpoint on the Hugging Face Hub
DATASET="smangrul/hf-stack-v1"   # Dataset on the Hugging Face Hub
DATA_COLUMN="content"            # Column name containing the code content

SEQ_LENGTH=2048                  # Sequence length

# Training arguments
MAX_STEPS=2000                   # max_steps
BATCH_SIZE=16                    # batch_size
GR_ACC_STEPS=1                   # gradient_accumulation_steps
LR=5e-4                          # learning_rate
LR_SCHEDULER_TYPE="cosine"       # lr_scheduler_type
WEIGHT_DECAY=0.01                # weight_decay
NUM_WARMUP_STEPS=30              # num_warmup_steps
EVAL_FREQ=100                    # eval_freq
SAVE_FREQ=100                    # save_freq
LOG_FREQ=25                      # log_freq
OUTPUT_DIR="peft-starcoder-lora-a100" # output_dir
BF16=True                        # bf16
FP16=False                       # no_fp16

# FIM trasformations arguments
FIM_RATE=0.5                     # fim_rate
FIM_SPM_RATE=0.5                 # fim_spm_rate

# LORA
LORA_R=8                         # lora_r
LORA_ALPHA=32                    # lora_alpha
LORA_DROPOUT=0.0                 # lora_dropout
LORA_TARGET_MODULES="c_proj,c_attn,q_attn,c_fc,c_proj"    # lora_target_modules

# bitsandbytes config
USE_NESTED_QUANT=True            # use_nested_quant
BNB_4BIT_COMPUTE_DTYPE="bfloat16"# bnb_4bit_compute_dtype

SEED=0
```


```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
)

set_seed(SEED)
```

## Prepare the data

Begin by loading the data. As the dataset is likely to be quite large, make sure to enable the streaming mode. Streaming allows us to load the data progressively as we iterate over the dataset instead of downloading the whole dataset at once.

We'll reserve the first 4000 examples as the validation set, and everything else will be the training data.


```python
from datasets import load_dataset
import torch
from tqdm import tqdm


dataset = load_dataset(
    DATASET,
    data_dir="data",
    split="train",
    streaming=True,
)

valid_data = dataset.take(4000)
train_data = dataset.skip(4000)
train_data = train_data.shuffle(buffer_size=5000, seed=SEED)
```

At this step, the dataset still contains raw data with code of arbitraty length. For training, we need inputs of fixed length. Let's create an Iterable dataset that would return constant-length chunks of tokens from a stream of text files.

First, let's estimate the average number of characters per token in the dataset, which will help us later estimate the number of tokens in the text buffer later. By default, we'll only take 400 examples (`nb_examples`) from the dataset. Using only a subset of the entire dataset will reduce computational cost while still providing a reasonable estimate of the overall character-to-token ratio.


```python
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """

    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


chars_per_token = chars_token_ratio(train_data, tokenizer, DATA_COLUMN)
print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
```

    100%|██████████| 400/400 [00:10<00:00, 39.87it/s] 

    The character to token ratio of the dataset is: 2.43
    

    
    

The character-to-token ratio can also be used as an indicator of the quality of text tokenization. For instance, a character-to-token ratio of 1.0 would mean that each character is represented with a token, which is not very meaningful. This would indicate poor tokenization. In standard English text, one token is typically equivalent to approximately four characters, meaning the character-to-token ratio is around 4.0. We can expect a lower ratio in the code dataset, but generally speaking, a number between 2.0 and 3.5 can be considered good enough.

**Optional FIM transformations**


Autoregressive language models typically generate sequences from left to right. By applying the FIM transformations, the model can also learn to infill text.  Check out ["Efficient Training of Language Models to Fill in the Middle" paper](https://arxiv.org/pdf/2207.14255.pdf) to learn more about the technique.
We'll define the FIM transformations here and will use them when creating the Iterable Dataset. However, if you want to omit transformations, feel free to set `fim_rate` to 0.


```python
import functools
import numpy as np


# Helper function to get token ids of the special tokens for prefix, suffix and middle for FIM transformations.
@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    try:
        FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.special_tokens_map["additional_special_tokens"][1:5]
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = None, None, None, None
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    # The if condition will trigger with the probability of fim_rate
    # This means FIM transformations will apply to samples with a probability of fim_rate
    if np_rng.binomial(1, fim_rate):

        # Split the sample into prefix, middle, and suffix, based on randomly generated indices stored in the boundaries list.
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            # calculate the new total length of the sample, taking into account tokens indicating prefix, middle, and suffix
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)

            # trancate or pad if there's a difference in length between the new length and the original
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        # With the probability of fim_spm_rateapply SPM variant of FIM transformations
        # SPM: suffix, prefix, middle
        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        # Otherwise, apply the PSM variant of FIM transformations
        # PSM: prefix, suffix, middle
        else:

            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't apply FIM transformations
        new_sample = sample

    return list(new_sample), np_rng

```

Let's define the `ConstantLengthDataset`, an Iterable dataset that will return constant-length chunks of tokens. To do so, we'll read a buffer of text from the original dataset until we hit the size limits and then apply tokenizer to convert the raw text into tokenized inputs. Optionally, we'll perform FIM transformations on some sequences (the proportion of sequences affected is controlled by `fim_rate`).

Once defined, we can create instances of the `ConstantLengthDataset` from both training and validation data.


```python
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
import random

# Create an Iterable dataset that returns constant-length chunks of tokens from a stream of text files.

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []

            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )

                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=SEQ_LENGTH,
        chars_per_token=chars_per_token,
        content_field=DATA_COLUMN,
        fim_rate=FIM_RATE,
        fim_spm_rate=FIM_SPM_RATE,
        seed=SEED,
)
eval_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=SEQ_LENGTH,
        chars_per_token=chars_per_token,
        content_field=DATA_COLUMN,
        fim_rate=FIM_RATE,
        fim_spm_rate=FIM_SPM_RATE,
        seed=SEED,
)
```

## Prepare the model

Now that the data is prepared, it's time to load the model! We're going to load the quantized version of the model.

This will allow us to reduce memory usage, as quantization represents data with fewer bits. We'll use the `bitsandbytes` library to quantize the model, as it has a nice integration with `transformers`. All we need to do is define a `bitsandbytes` config, and then use it when loading the model.

There are different variants of 4bit quantization, but generally, we recommend using NF4 quantization for better performance (`bnb_4bit_quant_type="nf4"`).

The `bnb_4bit_use_double_quant` option adds a second quantization after the first one to save an additional 0.4 bits per parameter.

To learn more about quantization, check out the ["Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA" blog post](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

Once defined, pass the config to the `from_pretrained` method to load the quantized version of the model.


```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

load_in_8bit = False

# 4-bit quantization
compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=USE_NESTED_QUANT,
)

device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=False,  # We will be using gradient checkpointing
        trust_remote_code=True,
        use_flash_attention_2=True,
)

```

When using a quantized model for training, you need to call the `prepare_model_for_kbit_training()` function to preprocess the quantized model for training.


```python
model = prepare_model_for_kbit_training(model)
```

Now that the quantized model is ready, we can set up a LoRA configuration. LoRA makes fine-tuning more efficient by drastically reducing the number of trainable parameters.

To train a model using LoRA technique, we need to wrap the base model as a `PeftModel`. This involves definign LoRA configuration with `LoraConfig`, and wrapping the original model with `get_peft_model()` using the `LoraConfig`.

To learn more about LoRA and its parameters, refer to [PEFT documentation](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora).


```python
# Set up lora
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES.split(","),
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

    trainable params: 5,554,176 || all params: 1,142,761,472 || trainable%: 0.4860310866343243
    

As you can see, by applying LoRA technique we will now need to train less than 1% of the parameters.

## Train the model

Now that we have prepared the data, and optimized the model, we are ready to bring everything together to start the training.

To instantiate a `Trainer`, you need to define the training configuration. The most important is the `TrainingArguments`, which is a class that contains all the attributes to configure the training.

These are similar to any other kind of model training you may run, so we won't go into detail here.


```python
train_data.start_iteration = 0


training_args = TrainingArguments(
    output_dir=f"Your_HF_username/{OUTPUT_DIR}",
    dataloader_drop_last=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    max_steps=MAX_STEPS,
    eval_steps=EVAL_FREQ,
    save_steps=SAVE_FREQ,
    logging_steps=LOG_FREQ,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_steps=NUM_WARMUP_STEPS,
    gradient_accumulation_steps=GR_ACC_STEPS,
    gradient_checkpointing=True,
    fp16=FP16,
    bf16=BF16,
    weight_decay=WEIGHT_DECAY,
    push_to_hub=True,
    include_tokens_per_second=True,
)

```

As a final step, instantiate the `Trainer` and call the `train` method.   


```python
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
)

print("Training...")
trainer.train()

```

    Training...
    



    <div>

      <progress value='2000' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [2000/2000 4:16:10, Epoch 1/9223372036854775807]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>5.524600</td>
      <td>7.456872</td>
    </tr>
    <tr>
      <td>200</td>
      <td>5.617800</td>
      <td>7.262190</td>
    </tr>
    <tr>
      <td>300</td>
      <td>5.129100</td>
      <td>6.410039</td>
    </tr>
    <tr>
      <td>400</td>
      <td>5.052200</td>
      <td>6.306774</td>
    </tr>
    <tr>
      <td>500</td>
      <td>5.202900</td>
      <td>6.117062</td>
    </tr>
    <tr>
      <td>600</td>
      <td>4.654100</td>
      <td>6.018349</td>
    </tr>
    <tr>
      <td>700</td>
      <td>5.100200</td>
      <td>6.000355</td>
    </tr>
    <tr>
      <td>800</td>
      <td>5.049800</td>
      <td>5.889457</td>
    </tr>
    <tr>
      <td>900</td>
      <td>4.541200</td>
      <td>5.813823</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>5.000700</td>
      <td>5.834208</td>
    </tr>
    <tr>
      <td>1100</td>
      <td>5.026500</td>
      <td>5.781939</td>
    </tr>
    <tr>
      <td>1200</td>
      <td>4.411800</td>
      <td>5.720596</td>
    </tr>
    <tr>
      <td>1300</td>
      <td>4.782500</td>
      <td>5.736376</td>
    </tr>
    <tr>
      <td>1400</td>
      <td>4.980200</td>
      <td>5.712276</td>
    </tr>
    <tr>
      <td>1500</td>
      <td>4.368700</td>
      <td>5.689637</td>
    </tr>
    <tr>
      <td>1600</td>
      <td>4.884700</td>
      <td>5.675920</td>
    </tr>
    <tr>
      <td>1700</td>
      <td>4.914400</td>
      <td>5.662421</td>
    </tr>
    <tr>
      <td>1800</td>
      <td>4.248700</td>
      <td>5.660122</td>
    </tr>
    <tr>
      <td>1900</td>
      <td>4.798400</td>
      <td>5.664026</td>
    </tr>
    <tr>
      <td>2000</td>
      <td>4.704200</td>
      <td>5.655665</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=2000, training_loss=4.885598585128784, metrics={'train_runtime': 15380.3075, 'train_samples_per_second': 2.081, 'train_steps_per_second': 0.13, 'train_tokens_per_second': 4261.033, 'total_flos': 4.0317260660736e+17, 'train_loss': 4.885598585128784, 'epoch': 1.0})



Finally, you can push the fine-tuned model to your Hub repository to share with your team.


```python
trainer.push_to_hub()
```

## Inference

Once the model is uploaded to Hub, we can use it for inference. To do so we first initialize the original base model and its tokenizer. Next, we need to merge the fine-duned weights with the base model.


```python
from peft import PeftModel
import torch

# load the original model first
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=None,
    device_map=None,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()

# merge fine-tuned weights with the base model
peft_model_id = f"Your_HF_username/{OUTPUT_DIR}"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_and_unload()
```

Now we can use the merged model for inference. For convenience, we'll define a `get_code_completion` - feel free to experiment with text generation parameters!


```python
def get_code_completion(prefix, suffix):
    text = prompt = f"""<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"""
    model.eval()
    outputs = model.generate(
        input_ids=tokenizer(text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.0,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
```

Now all we need to do to get code completion is call the `get_code_complete` function and pass the first few lines that we want to be completed as a prefix, and an empty string as a suffix.


```python
prefix = """from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
peft_config = LoraConfig(
"""
suffix =""""""

print(get_code_completion(prefix, suffix))
```

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["q_proj", "v_proj"],
        inference_mode=False,
    )
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    

As someone who has just used the PEFT library earlier in this notebook, you can see that the generated result for creating a `LoraConfig` is rather good!

If you go back to the cell where we instantiate the model for inference, and comment out the lines where we merge the fine-tuned weights, you can see what the original model would've generated for the exact same prefix:


```python
prefix = """from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
peft_config = LoraConfig(
"""
suffix =""""""

print(get_code_completion(prefix, suffix))
```

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM
    peft_config = LoraConfig(
        model_name_or_path="facebook/wav2vec2-base-960h",
        num_labels=1,
        num_features=1,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_hidden_layers_per_attention_head=1,
        num_attention_heads_per_hidden_layer=1,
        hidden_size=1024,
        hidden_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_act_dropout_prob=0.1,
        hidden
    

While it is Python syntax, you can see that the original model has no understanding of what a `LoraConfig` should be doing.

To learn how this kind of fine-tuning compares to full fine-tuning, and how to use a model like this as your copilot in VS Code via Inference Endpoints, or locally, check out the ["Personal Copilot: Train Your Own Coding Assistant" blog post](https://huggingface.co/blog/personal-copilot). This notebook complements the original blog post.

