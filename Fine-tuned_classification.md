# Fine tuning classification example

We will fine-tune a `babbage-002` classifier (replacement for the `ada` models) to distinguish between the two sports: Baseball and Hockey.


```python
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import openai
import os

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

categories = ['rec.sport.baseball', 'rec.sport.hockey']
sports_dataset = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, categories=categories)
```

 ## Data exploration
 The newsgroup dataset can be loaded using sklearn. First we will look at the data itself:


```python
print(sports_dataset['data'][0])
```

    From: dougb@comm.mot.com (Doug Bank)
    Subject: Re: Info needed for Cleveland tickets
    Reply-To: dougb@ecs.comm.mot.com
    Organization: Motorola Land Mobile Products Sector
    Distribution: usa
    Nntp-Posting-Host: 145.1.146.35
    Lines: 17
    
    In article <1993Apr1.234031.4950@leland.Stanford.EDU>, bohnert@leland.Stanford.EDU (matthew bohnert) writes:
    
    |> I'm going to be in Cleveland Thursday, April 15 to Sunday, April 18.
    |> Does anybody know if the Tribe will be in town on those dates, and
    |> if so, who're they playing and if tickets are available?
    
    The tribe will be in town from April 16 to the 19th.
    There are ALWAYS tickets available! (Though they are playing Toronto,
    and many Toronto fans make the trip to Cleveland as it is easier to
    get tickets in Cleveland than in Toronto.  Either way, I seriously
    doubt they will sell out until the end of the season.)
    
    -- 
    Doug Bank                       Private Systems Division
    dougb@ecs.comm.mot.com          Motorola Communications Sector
    dougb@nwu.edu                   Schaumburg, Illinois
    dougb@casbah.acns.nwu.edu       708-576-8207                    
    
    


```python
sports_dataset.target_names[sports_dataset['target'][0]]

```




    'rec.sport.baseball'




```python
len_all, len_baseball, len_hockey = len(sports_dataset.data), len([e for e in sports_dataset.target if e == 0]), len([e for e in sports_dataset.target if e == 1])
print(f"Total examples: {len_all}, Baseball examples: {len_baseball}, Hockey examples: {len_hockey}")
```

    Total examples: 1197, Baseball examples: 597, Hockey examples: 600
    

One sample from the baseball category can be seen above. It is an email to a mailing list. We can observe that we have 1197 examples in total, which are evenly split between the two sports.

## Data Preparation
We transform the dataset into a pandas dataframe, with a column for prompt and completion. The prompt contains the email from the mailing list, and the completion is a name of the sport, either hockey or baseball. For demonstration purposes only and speed of fine-tuning we take only 300 examples. In a real use case the more examples the better the performance.


```python
import pandas as pd

labels = [sports_dataset.target_names[x].split('.')[-1] for x in sports_dataset['target']]
texts = [text.strip() for text in sports_dataset['data']]
df = pd.DataFrame(zip(texts, labels), columns = ['prompt','completion']) #[:300]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>completion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>From: dougb@comm.mot.com (Doug Bank)\nSubject:...</td>
      <td>baseball</td>
    </tr>
    <tr>
      <th>1</th>
      <td>From: gld@cunixb.cc.columbia.edu (Gary L Dare)...</td>
      <td>hockey</td>
    </tr>
    <tr>
      <th>2</th>
      <td>From: rudy@netcom.com (Rudy Wade)\nSubject: Re...</td>
      <td>baseball</td>
    </tr>
    <tr>
      <th>3</th>
      <td>From: monack@helium.gas.uug.arizona.edu (david...</td>
      <td>hockey</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Subject: Let it be Known\nFrom: &lt;ISSBTL@BYUVM....</td>
      <td>baseball</td>
    </tr>
  </tbody>
</table>
</div>



Both baseball and hockey are single tokens. We save the dataset as a jsonl file.


```python
df.to_json("sport2.jsonl", orient='records', lines=True)
```

### Data Preparation tool
We can now use a data preparation tool which will suggest a few improvements to our dataset before fine-tuning. Before launching the tool we update the openai library to ensure we're using the latest data preparation tool. We additionally specify `-q` which auto-accepts all suggestions.


```python
!openai tools fine_tunes.prepare_data -f sport2.jsonl -q
```

    Analyzing...
    
    - Your file contains 1197 prompt-completion pairs
    - Based on your data it seems like you're trying to fine-tune a model for classification
    - For classification, we recommend you try one of the faster and cheaper models, such as `ada`
    - For classification, you can estimate the expected model performance by keeping a held out dataset, which is not used for training
    - There are 11 examples that are very long. These are rows: [134, 200, 281, 320, 404, 595, 704, 838, 1113, 1139, 1174]
    For conditional generation, and for classification the examples shouldn't be longer than 2048 tokens.
    - Your data does not contain a common separator at the end of your prompts. Having a separator string appended to the end of the prompt makes it clearer to the fine-tuned model where the completion should begin. See https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset for more detail and examples. If you intend to do open-ended generation, then you should leave the prompts empty
    - The completion should start with a whitespace character (` `). This tends to produce better results due to the tokenization we use. See https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset for more details
    
    Based on the analysis we will perform the following actions:
    - [Recommended] Remove 11 long examples [Y/n]: Y
    - [Recommended] Add a suffix separator `\n\n###\n\n` to all prompts [Y/n]: Y
    - [Recommended] Add a whitespace character to the beginning of the completion [Y/n]: Y
    - [Recommended] Would you like to split into training and validation set? [Y/n]: Y
    
    
    Your data will be written to a new JSONL file. Proceed [Y/n]: Y
    
    Wrote modified files to `sport2_prepared_train (1).jsonl` and `sport2_prepared_valid (1).jsonl`
    Feel free to take a look!
    
    Now use that file when fine-tuning:
    > openai api fine_tunes.create -t "sport2_prepared_train (1).jsonl" -v "sport2_prepared_valid (1).jsonl" --compute_classification_metrics --classification_positive_class " baseball"
    
    After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string `\n\n###\n\n` for the model to start generating completions, rather than continuing with the prompt.
    Once your model starts training, it'll approximately take 30.8 minutes to train a `curie` model, and less for `ada` and `babbage`. Queue will approximately take half an hour per job ahead of you.
    

The tool helpfully suggests a few improvements to the dataset and splits the dataset into training and validation set.

A suffix between a prompt and a completion is necessary to tell the model that the input text has stopped, and that it now needs to predict the class. Since we use the same separator in each example, the model is able to learn that it is meant to predict either baseball or hockey following the separator.
A whitespace prefix in completions is useful, as most word tokens are tokenized with a space prefix.
The tool also recognized that this is likely a classification task, so it suggested to split the dataset into training and validation datasets. This will allow us to easily measure expected performance on new data.

## Fine-tuning
The tool suggests we run the following command to train the dataset. Since this is a classification task, we would like to know what the generalization performance on the provided validation set is for our classification use case.

We can simply copy the suggested command from the CLI tool. We specifically add `-m ada` to fine-tune a cheaper and faster ada model, which is usually comperable in performance to slower and more expensive models on classification use cases. 


```python
train_file = client.files.create(file=open("sport2_prepared_train.jsonl", "rb"), purpose="fine-tune")
valid_file = client.files.create(file=open("sport2_prepared_valid.jsonl", "rb"), purpose="fine-tune")

fine_tuning_job = client.fine_tuning.jobs.create(training_file=train_file.id, validation_file=valid_file.id, model="babbage-002")

print(fine_tuning_job)
```

    FineTuningJob(id='ftjob-REo0uLpriEAm08CBRNDlPJZC', created_at=1704413736, error=None, fine_tuned_model=None, finished_at=None, hyperparameters=Hyperparameters(n_epochs='auto', batch_size='auto', learning_rate_multiplier='auto'), model='babbage-002', object='fine_tuning.job', organization_id='org-9HXYFy8ux4r6aboFyec2OLRf', result_files=[], status='validating_files', trained_tokens=None, training_file='file-82XooA2AUDBAUbN5z2DuKRMs', validation_file='file-wTOcQF8vxQ0Z6fNY2GSm0z4P')
    

The model is successfully trained in about ten minutes. You can watch the finetune happen on [https://platform.openai.com/finetune/](https://platform.openai.com/finetune/)

You can also check on its status programatically:


```python
fine_tune_results = client.fine_tuning.jobs.retrieve(fine_tuning_job.id)
print(fine_tune_results.finished_at)
```

    1704414393
    

### [Advanced] Results and expected model performance
We can now download the results file to observe the expected performance on a held out validation set.


```python
fine_tune_results = client.fine_tuning.jobs.retrieve(fine_tuning_job.id).result_files
result_file = client.files.retrieve(fine_tune_results[0])
content = client.files.content(result_file.id)
# save content to file
with open("result.csv", "wb") as f:
    f.write(content.text.encode("utf-8"))
```


```python
results = pd.read_csv('result.csv')
results[results['train_accuracy'].notnull()].tail(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>step</th>
      <th>train_loss</th>
      <th>train_accuracy</th>
      <th>valid_loss</th>
      <th>valid_mean_token_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2843</th>
      <td>2844</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The accuracy reaches 99.6%. On the plot below we can see how accuracy on the validation set increases during the training run. 


```python
results[results['train_accuracy'].notnull()]['train_accuracy'].plot()
```

## Using the model
We can now call the model to get the predictions.


```python
test = pd.read_json('sport2_prepared_valid.jsonl', lines=True)
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>completion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>From: gld@cunixb.cc.columbia.edu (Gary L Dare)...</td>
      <td>hockey</td>
    </tr>
    <tr>
      <th>1</th>
      <td>From: smorris@venus.lerc.nasa.gov (Ron Morris ...</td>
      <td>hockey</td>
    </tr>
    <tr>
      <th>2</th>
      <td>From: golchowy@alchemy.chem.utoronto.ca (Geral...</td>
      <td>hockey</td>
    </tr>
    <tr>
      <th>3</th>
      <td>From: krattige@hpcc01.corp.hp.com (Kim Krattig...</td>
      <td>baseball</td>
    </tr>
    <tr>
      <th>4</th>
      <td>From: warped@cs.montana.edu (Doug Dolven)\nSub...</td>
      <td>baseball</td>
    </tr>
  </tbody>
</table>
</div>



We need to use the same separator following the prompt which we used during fine-tuning. In this case it is `\n\n###\n\n`. Since we're concerned with classification, we want the temperature to be as low as possible, and we only require one token completion to determine the prediction of the model.


```python
ft_model = fine_tune_results.fine_tuned_model

# note that this calls the legacy completions api - https://platform.openai.com/docs/api-reference/completions
res = client.completions.create(model=ft_model, prompt=test['prompt'][0] + '\n\n###\n\n', max_tokens=1, temperature=0)
res.choices[0].text

```




    ' hockey'



To get the log probabilities, we can specify logprobs parameter on the completion request


```python
res = client.completions.create(model=ft_model, prompt=test['prompt'][0] + '\n\n###\n\n', max_tokens=1, temperature=0, logprobs=2)
res.choices[0].logprobs.top_logprobs
```




    [{' hockey': 0.0, ' Hockey': -22.504879}]



We can see that the model predicts hockey as a lot more likely than baseball, which is the correct prediction. By requesting log_probs, we can see the prediction (log) probability for each class.

### Generalization
Interestingly, our fine-tuned classifier is quite versatile. Despite being trained on emails to different mailing lists, it also successfully predicts tweets.


```python
sample_hockey_tweet = """Thank you to the 
@Canes
 and all you amazing Caniacs that have been so supportive! You guys are some of the best fans in the NHL without a doubt! Really excited to start this new chapter in my career with the 
@DetroitRedWings
 !!"""
res = client.completions.create(model=ft_model, prompt=sample_hockey_tweet + '\n\n###\n\n', max_tokens=1, temperature=0, logprobs=2)
res.choices[0].text
```




    ' hockey'




```python
sample_baseball_tweet="""BREAKING: The Tampa Bay Rays are finalizing a deal to acquire slugger Nelson Cruz from the Minnesota Twins, sources tell ESPN."""
res = client.completions.create(model=ft_model, prompt=sample_baseball_tweet + '\n\n###\n\n', max_tokens=1, temperature=0, logprobs=2)
res.choices[0].text
```
