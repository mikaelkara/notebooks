# CTranslate2

**CTranslate2** is a C++ and Python library for efficient inference with Transformer models.

The project implements a custom runtime that applies many performance optimization techniques such as weights quantization, layers fusion, batch reordering, etc., to accelerate and reduce the memory usage of Transformer models on CPU and GPU.

Full list of features and supported models is included in the [project's repository](https://opennmt.net/CTranslate2/guides/transformers.html). To start, please check out the official [quickstart guide](https://opennmt.net/CTranslate2/quickstart.html).

To use, you should have `ctranslate2` python package installed.


```python
%pip install --upgrade --quiet  ctranslate2
```

To use a Hugging Face model with CTranslate2, it has to be first converted to CTranslate2 format using the `ct2-transformers-converter` command. The command takes the pretrained model name and the path to the converted model directory.


```python
# conversation can take several minutes
!ct2-transformers-converter --model meta-llama/Llama-2-7b-hf --quantization bfloat16 --output_dir ./llama-2-7b-ct2 --force
```

    Loading checkpoint shards: 100%|██████████████████| 2/2 [00:01<00:00,  1.81it/s]
    


```python
from langchain_community.llms import CTranslate2

llm = CTranslate2(
    # output_dir from above:
    model_path="./llama-2-7b-ct2",
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    device="cuda",
    # device_index can be either single int or list or ints,
    # indicating the ids of GPUs to use for inference:
    device_index=[0, 1],
    compute_type="bfloat16",
)
```

## Single call


```python
print(
    llm.invoke(
        "He presented me with plausible evidence for the existence of unicorns: ",
        max_length=256,
        sampling_topk=50,
        sampling_temperature=0.2,
        repetition_penalty=2,
        cache_static_prompt=False,
    )
)
```

    He presented me with plausible evidence for the existence of unicorns: 1) they are mentioned in ancient texts; and, more importantly to him (and not so much as a matter that would convince most people), he had seen one.
    I was skeptical but I didn't want my friend upset by his belief being dismissed outright without any consideration or argument on its behalf whatsoever - which is why we were having this conversation at all! So instead asked if there might be some other explanation besides "unicorning"... maybe it could have been an ostrich? Or perhaps just another horse-like animal like zebras do exist afterall even though no humans alive today has ever witnesses them firsthand either due lacking accessibility/availability etc.. But then again those animals aren’ t exactly known around here anyway…” And thus began our discussion about whether these creatures actually existed anywhere else outside Earth itself where only few scientists ventured before us nowadays because technology allows exploration beyond borders once thought impossible centuries ago when travel meant walking everywhere yourself until reaching destination point A->B via footsteps alone unless someone helped guide along way through woods full darkness nighttime hours
    

## Multiple calls:


```python
print(
    llm.generate(
        ["The list of top romantic songs:\n1.", "The list of top rap songs:\n1."],
        max_length=128,
    )
)
```

    generations=[[Generation(text='The list of top romantic songs:\n1. “I Will Always Love You” by Whitney Houston\n2. “Can’t Help Falling in Love” by Elvis Presley\n3. “Unchained Melody” by The Righteous Brothers\n4. “I Will Always Love You” by Dolly Parton\n5. “I Will Always Love You” by Whitney Houston\n6. “I Will Always Love You” by Dolly Parton\n7. “I Will Always Love You” by The Beatles\n8. “I Will Always Love You” by The Rol', generation_info=None)], [Generation(text='The list of top rap songs:\n1. “God’s Plan” by Drake\n2. “Rockstar” by Post Malone\n3. “Bad and Boujee” by Migos\n4. “Humble” by Kendrick Lamar\n5. “Bodak Yellow” by Cardi B\n6. “I’m the One” by DJ Khaled\n7. “Motorsport” by Migos\n8. “No Limit” by G-Eazy\n9. “Bounce Back” by Big Sean\n10. “', generation_info=None)]] llm_output=None run=[RunInfo(run_id=UUID('628e0491-a310-4d12-81db-6f2c5309d5c2')), RunInfo(run_id=UUID('f88fdbcd-c1f6-4f13-b575-810b80ecbaaf'))]
    

## Integrate the model in an LLMChain


```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

template = """{question}

Let's think step by step. """
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who was the US president in the year the first Pokemon game was released?"

print(llm_chain.run(question))
```

    Who was the US president in the year the first Pokemon game was released?
    
    Let's think step by step. 1996 was the year the first Pokemon game was released.
    
    \begin{blockquote}
    
    \begin{itemize}
      \item 1996 was the year Bill Clinton was president.
      \item 1996 was the year the first Pokemon game was released.
      \item 1996 was the year the first Pokemon game was released.
    
    \end{itemize}
    \end{blockquote}
    
    I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: @JoeZ. I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: @JoeZ. I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: @JoeZ. I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: @JoeZ. I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: @JoeZ. I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: @JoeZ. I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: @JoeZ. I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: @JoeZ. I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: @JoeZ. I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    Comment: @JoeZ. I'm not sure if this is a valid question, but I'm sure it's a fun one.
    
    
