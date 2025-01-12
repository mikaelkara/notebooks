Copyright (c) Meta Platforms, Inc. and affiliates.
This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

Use this notebook to pull in datasets and apply pre-processing.  Most grammar datasets unfortunately require preprocessing before being usable in training. (example - jfleg has 4 targets per input, so we have to rematch as 1:1 pairings) 


```python
import csv
from datasets import load_metric, load_dataset
from pathlib import Path
```


```python
list_replacements = [
  (" .", "."), 
  (" ,", ","),
  (" '", "'"),
  (" ?", "?"),
  (" !", "!"),
  (" :", ":"),
  (" ;", ";"),
  (" n't", "n't"),
  (" v", "v"),
  ("2 0 0 6", "2006"),
  ("5 5", "55"),
  ("4 0 0", "400"),
  ("1 7-5 0", "1750"),
  ("2 0 %", "20%"),
  ("5 0", "50"),
  ("1 2", "12"),
  ("1 0", "10"),
  ('" ballast water', '"ballast water')
  ]
```


```python
def correct_spacing(item):
    """ we iterate through the list of all replacements per each item in dataset"""
    for fix in list_replacements:
        item = item.replace(fix[0], fix[1])
    return item


```


```python
def generate_csv(csv_path, dataset):
    """ apply spacing corrections and save out matched pairs to csv file as dataset"""
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["input", "target"])
        for case in dataset:
     	    # Adding the t5 task indication prefix to input 
            input_text = case["sentence"]
            input_text = correct_spacing(input_text)

            for correction in case["corrections"]:
              correction = correct_spacing(correction)
              # a few of the cases contain blank strings. 
              if input_text and correction:
                writer.writerow([input_text, correction])
```

In Jfleg  - validation will be used as 'train', test will be 'validation'


```python
train_dataset = load_dataset("jfleg", split='validation[:]') 
eval_dataset = load_dataset("jfleg", split='test[:]')

```

    Found cached dataset jfleg (/data/home/mreso/.cache/huggingface/datasets/jfleg/default/1.0.0/ed4ab2367351fe31949f48849ae6732b164f0d5ea6bb5d4357ff4293ac89511b)
    Found cached dataset jfleg (/data/home/mreso/.cache/huggingface/datasets/jfleg/default/1.0.0/ed4ab2367351fe31949f48849ae6732b164f0d5ea6bb5d4357ff4293ac89511b)
    


```python
print(train_dataset)
print(eval_dataset)

```

    Dataset({
        features: ['sentence', 'corrections'],
        num_rows: 755
    })
    Dataset({
        features: ['sentence', 'corrections'],
        num_rows: 748
    })
    


```python
print(train_dataset['sentence'][22])
print(train_dataset['corrections'][22])
```

    Students can focus on only a few subjects they are intwerested in and they will become an experts in those areas . 
    ['Students can focus on only a few subjects they are interested in and they will become experts in those areas . ', 'Students can focus on only a few subjects they are interested in and they will become experts in those areas . ', 'Students can focus on only a few subjects they are interested in and they will become an expert in those areas . ', 'Students can focus on only a few subjects they are interested in and they will become an expert in those areas . ']
    


```python
clean22 = correct_spacing(train_dataset['sentence'][22])
clean22
```




    'Students can focus on only a few subjects they are intwerested in and they will become an experts in those areas. '




```python
jfleg_dir = Path.cwd()/'jfleg_dataset'  # if you only use 'jfleg', hf will try and use that and complain
jfleg_dir.mkdir(parents=True,exist_ok=True)
c4_dir = Path.cwd()/'c4_dataset'
c4_dir.mkdir(parents=True,exist_ok=True)
```

Process Jfleg data  


```python
j_train_file = jfleg_dir/'jtrain.csv'
j_eval_file = jfleg_dir/'jeval.csv'
```


```python
generate_csv(j_train_file, train_dataset)
```


```python
generate_csv(j_eval_file, eval_dataset)
```

Process C4_200M (!) - we'll pull 10K to start


```python
c4_dataset = load_dataset("liweili/c4_200m", streaming = True)
```


```python
iterator = iter(c4_dataset['train'])
```


```python
def c4_generate_csv(csv_path, iterator, num_examples):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["input", "target"])
        for i in range(0,num_examples):
          data = next(iterator)
          input_text = data["input"]
          input_text = correct_spacing(input_text)
          correction = correct_spacing(data["output"])
          if input_text and correction:
            writer.writerow([input_text, correction])
```


```python
c4_dir = Path.cwd()/'c4_dataset'
c4_dir.mkdir(parents=True,exist_ok=True)
```

You can modify the following to make the csv file with desired number of instances, here we go for 10k to make a quick test


```python
c4_filename = c4_dir/'c4train_10k.csv'
```


```python
c4_generate_csv(c4_filename, iterator, num_examples=10000)
```

Create a single training file by combining jtrain and c4train


```python
merge_list = [j_train_file, c4_filename, ]
```


```python
import pandas as pd
```


```python
combined_csv = pd.concat([pd.read_csv(fn) for fn in merge_list])

```


```python
merged_name = "gtrain_10k.csv"
```


```python
combined_csv.to_csv(merged_name, index=False, encoding = 'utf-8-sig', )
```


```python
eval_name = "grammar_validation.csv"
```


```python
eval_csv = pd.read_csv(j_eval_file)
eval_csv.to_csv(eval_name, index=False, encoding = 'utf-8-sig', )
```
