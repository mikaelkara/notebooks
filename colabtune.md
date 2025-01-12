# Model tuning in colab

**Note:** this notebook is intended as a demonstration that finetuning can be done in a free colab tier. If you need to train a large model on a sizable dataset, you should consider upgrading to a pro tier and getting more resources. Otherwise, the training will take impractically long time. If you don't use a large enough model or tune on insufficient number of sampes, the quality won't be good.

## Environment setup

Make sure that you are connected to a GPU host.


```python
!nvidia-smi
```

Check out the cook book code from github.


```python
!git clone https://github.com/fw-ai/cookbook.git
```

Install required dependencies.


```python
!pip install \
    accelerate \
    bitsandbytes \
    datasets \
    evals \
    fire \
    guidance \
    huggingface_hub \
    hydra-core \
    ninja \
    packaging \
    peft \
    py7zr \
    s3fs \
    sentencepiece \
    torchx \
    transformers \
    zstandard
```

Log into huggingface. For that, you will need a huggingface token. You can find it by going to https://huggingface.co/settings/tokens
If you are running a recipe that uses llama2 models, don't forget to sign the license agreement on the model card.


```python
!huggingface-cli login --token <your_hf_token>
```

## Fine tuning


```python
!PYTHONPATH="$PYTHONPATH:/content/cookbook" torchx run -s local_cwd dist.ddp -j 1x1 --script /content/cookbook/recipes/tune/instruct_lora/finetune.py -- --config-name=summarize model=llama2-7b-chat-colab
```
