<span style="color:orange; font-weight:bold">Note: To answer questions based on text documents, we recommend the procedure in <a href="https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb">Question Answering using Embeddings</a>. Some of the code below may rely on <a href="https://github.com/openai/openai-cookbook/tree/main/transition_guides_for_deprecated_API_endpoints">deprecated API endpoints</a>.</span>

# 3. Train a fine-tuning model specialized for Q&A
This notebook will utilize the dataset of context, question and answer pairs to additionally create adversarial questions and context pairs, where the question was not generated on that context. In those cases the model will be prompted to answer "No sufficient context for answering the question". We will also train a discriminator model, which predicts whether the question can be answered based on the context or not.

We will add hard adversarial examples as well, which will be based either on semantically similar sections, or neighbouring sections, originating from the same article.


```python
import openai
import pandas as pd
df = pd.read_csv('olympics-data/olympics_qa.csv')
olympics_search_fileid = "file-c3shd8wqF3vSCKaukW4Jr1TT"
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
      <th>title</th>
      <th>heading</th>
      <th>content</th>
      <th>tokens</th>
      <th>context</th>
      <th>questions</th>
      <th>answers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020 Summer Olympics</td>
      <td>Summary</td>
      <td>The 2020 Summer Olympics (Japanese: 2020年夏季オリン...</td>
      <td>713</td>
      <td>2020 Summer Olympics\nSummary\n\nThe 2020 Summ...</td>
      <td>1. What is the 2020 Summer Olympics?\n2. When ...</td>
      <td>1. The 2020 Summer Olympics is an internationa...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020 Summer Olympics</td>
      <td>Host city selection</td>
      <td>The International Olympic Committee (IOC) vote...</td>
      <td>126</td>
      <td>2020 Summer Olympics\nHost city selection\n\nT...</td>
      <td>1. \n2. \n3. \n4.</td>
      <td>1. What is the International Olympic Committee...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020 Summer Olympics</td>
      <td>Impact of the COVID-19 pandemic</td>
      <td>In January 2020, concerns were raised about th...</td>
      <td>369</td>
      <td>2020 Summer Olympics\nImpact of the COVID-19 p...</td>
      <td>1. What was the COVID-19 pandemic?\n2. How did...</td>
      <td>1. The COVID-19 pandemic was a pandemic that o...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020 Summer Olympics</td>
      <td>Qualifying event cancellation and postponement</td>
      <td>Concerns about the pandemic began to affect qu...</td>
      <td>298</td>
      <td>2020 Summer Olympics\nQualifying event cancell...</td>
      <td>1. What was the original location of the Asia ...</td>
      <td>1. The original location of the Asia &amp; Oceania...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020 Summer Olympics</td>
      <td>Effect on doping tests</td>
      <td>Mandatory doping tests were being severely res...</td>
      <td>163</td>
      <td>2020 Summer Olympics\nEffect on doping tests\n...</td>
      <td>1. What was the COVID-19 pandemic?\n2. What di...</td>
      <td>1. The COVID-19 pandemic was a pandemic that o...</td>
    </tr>
  </tbody>
</table>
</div>



Split the sections into a training and testing set


```python
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
len(train_df), len(test_df)
```




    (3014, 754)



we check that the separator we intend to use isn't present within the contexts


```python
df.context.str.contains('->').sum()
```




    0



## 3.1 Create the fine-tuning datasets for Q&A and discriminator models
The fine-tuning dataset is created in the following way. For every corresponding question, answer and context pair we create:
- Positive example: correct question, answer, context pair
- Negative examples:
  - random negative example, where the random context is paired with the question 
  - two hard negative examples
    - one originating from the same wikipedia article
    - another, which is most similar to the correct context

This process is noisy, as sometimes the question might be answerable given a different context, but on average we hope this won't affect the performance too much.

We apply the same process of dataset creation for both the discriminator, and the Q&A answering model. We apply the process separately for the training and testing set, to ensure that the examples from the training set don't feature within the test set.


```python
import random

def get_random_similar_contexts(question, context, file_id=olympics_search_fileid, search_model='ada', max_rerank=10):
    """
    Find similar contexts to the given context using the search file
    """
    try:
        # TODO: openai.Engine(search_model) is deprecated
        results = openai.Engine(search_model).search(
            search_model=search_model, 
            query=question, 
            max_rerank=max_rerank,
            file=file_id
        )
        candidates = []
        for result in results['data'][:3]:
            if result['text'] == context:
                continue
            candidates.append(result['text'])
        random_candidate = random.choice(candidates)
        return random_candidate
    except Exception as e:
        print(e)
        return ""

def create_fine_tuning_dataset(df, discriminator=False, n_negative=1, add_related=False):
    """
    Create a dataset for fine tuning the OpenAI model; either for a discriminator model, 
    or a model specializing in Q&A, where it says if no relevant context is found.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the question, answer and context pairs
    discriminator: bool
        Whether to create a dataset for the discriminator
    n_negative: int
        The number of random negative samples to add (using a random context)
    add_related: bool
        Whether to add the related contexts to the correct context. These are hard negative examples

    Returns
    -------
    pd.DataFrame
        The dataframe containing the prompts and completions, ready for fine-tuning
    """
    rows = []
    for i, row in df.iterrows():
        for q, a in zip(("1." + row.questions).split('\n'), ("1." + row.answers).split('\n')):
            if len(q) >10 and len(a) >10:
                if discriminator:
                    rows.append({"prompt":f"{row.context}\nQuestion: {q[2:].strip()}\n Related:", "completion":f" yes"})
                else:
                    rows.append({"prompt":f"{row.context}\nQuestion: {q[2:].strip()}\nAnswer:", "completion":f" {a[2:].strip()}"})

    for i, row in df.iterrows():
        for q in ("1." + row.questions).split('\n'):
            if len(q) >10:
                for j in range(n_negative + (2 if add_related else 0)):
                    random_context = ""
                    if j == 0 and add_related:
                        # add the related contexts based on originating from the same wikipedia page
                        subset = df[(df.title == row.title) & (df.context != row.context)]
                        
                        if len(subset) < 1:
                            continue
                        random_context = subset.sample(1).iloc[0].context
                    if j == 1 and add_related:
                        # add the related contexts based on the most similar contexts according to the search
                        random_context = get_random_similar_contexts(q[2:].strip(), row.context, search_model='ada', max_rerank=10)
                    else:
                        while True:
                            # add random context, which isn't the correct context
                            random_context = df.sample(1).iloc[0].context
                            if random_context != row.context:
                                break
                    if discriminator:
                        rows.append({"prompt":f"{random_context}\nQuestion: {q[2:].strip()}\n Related:", "completion":f" no"})
                    else:
                        rows.append({"prompt":f"{random_context}\nQuestion: {q[2:].strip()}\nAnswer:", "completion":f" No appropriate context found to answer the question."})

    return pd.DataFrame(rows) 
```

We apply the same process of dataset creation for both the discriminator, and the Q&A answering model. We apply the process separately for the training and testing set, to ensure that the examples from the training set don't feature within the test set.


```python
for name, is_disc in [('discriminator', True), ('qa', False)]:
    for train_test, dt in [('train', train_df), ('test', test_df)]:
        ft = create_fine_tuning_dataset(dt, discriminator=is_disc, n_negative=1, add_related=True)
        ft.to_json(f'{name}_{train_test}.jsonl', orient='records', lines=True)
```




    



We formatted the data according to the recommendations from the fine-tuning tool, which is available using
> openai tools fine_tunes.prepare_data -f qa_train.jsonl

We highly recommend that you use this tool, which suggests improvements in your data formatting for fine-tuning.


## 3.2 Submit the datasets for fine-tuning


```python
!openai api fine_tunes.create -t "olympics-data/discriminator_train.jsonl" -v "olympics-data/discriminator_test.jsonl" --batch_size 16  --compute_classification_metrics --classification_positive_class " yes" --model ada
```




    




```python
!openai api fine_tunes.create -t "olympics-data/qa_train.jsonl" -v "olympics-data/qa_test.jsonl" --batch_size 16
```




    



## 3.3 Using the fine-tuned models

We will now use the fine-tuned discriminator and the fine-tuned Q&A model. By requesting logprobs, we can see how certain the discriminator is in a `yes` vs `no` answer.


```python
ft_discriminator = "curie:ft-openai-internal-2021-08-23-23-58-57"
ft_qa = "curie:ft-openai-internal-2021-08-23-17-54-10"

def apply_ft_discriminator(context, question, discriminator_model):
    """
    Apply the fine tuned discriminator to a question, to assess whether it can be answered from the context.
    """
    prompt = f"{context}\nQuestion: {question}\n Related:"
    result = openai.chat.completions.create(model=discriminator_model, prompt=prompt, max_tokens=1, temperature=0, top_p=1, n=1, logprobs=2)
    return result['choices'][0]['logprobs']['top_logprobs']

apply_ft_discriminator('The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.', 
                        'What was the first human-made object in space?', ft_discriminator)
```




    [<OpenAIObject at 0x7fe812e602b0> JSON: {
       " no": -10.819577,
       " yes": -2.045765e-05
     }]



We can see that the model can generalize well to different contexts and questions. 


```python
def apply_ft_qa_answer(context, question, answering_model):
    """
    Apply the fine tuned discriminator to a question
    """
    prompt = f"{context}\nQuestion: {question}\nAnswer:"
    result = openai.chat.completions.create(model=answering_model, prompt=prompt, max_tokens=30, temperature=0, top_p=1, n=1, stop=['.','\n'])
    return result['choices'][0]['text']

apply_ft_qa_answer('The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.', 
                    'What was the first human-made object in space?', ft_qa)

```




    ' The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957'



We can see that the model can answer the question, when the context is appropriate.


```python
apply_ft_qa_answer('The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.',
                    'What is impressive about the Soviet Union?', ft_qa)
```




    ' The Soviet Union was the first country to successfully launch a satellite into space'




```python
apply_ft_qa_answer('The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.',
                    'How many cars were produced in the Soviet Union in 1970?', ft_qa)
```




    ' No appropriate context found to answer the question'



We can see that the model knows when to answer the question, and when to say that insufficient context is present to answer the question.

We can also combine a discriminator and a base model, or a fine-tuned Q&A model. Discriminator can essentially serve as a decision whether the question can be answered given the context or not.


```python
def answer_question_conditionally(answering_model, discriminator_model, context, question, discriminator_logprob_yes_modifier=0):
    logprobs = apply_ft_discriminator(context, question, discriminator_model)
    yes_logprob = logprobs[' yes'] if ' yes' in logprobs else -100
    no_logprob = logprobs[' no'] if ' no' in logprobs else -100
    if yes_logprob + discriminator_logprob_yes_modifier < no_logprob:
        return " No appropriate context found to answer the question based on the discriminator."
    return apply_ft_qa_answer(context, question, answering_model)
answer_question_conditionally(ft_qa, ft_discriminator, 
                                "Crowdless games are a rare although not unheard-of occurrence in sports. \
                                 When they do occur, it is usually the result of events beyond the control \
                                 of the teams or fans, such as weather-related concerns, public health concerns, \
                                 or wider civil disturbances unrelated to the game. For instance, \
                                 the COVID-19 pandemic caused many sports leagues around the world \
                                 to be played behind closed doors.",
                                "Could weather cause a sport event to have no crowd?")
```




    ' Weather could cause a sport event to have no crowd'



The above function illustrates how to potentially combine a discriminator and a fine-tuned Q&A model. This gives a more fine-grained control over how certain we want the model to be before it answers the question.

We'll now take a look on how answers endpoint works - combining search to retrieve the relevant context from a knowledge base, and then using the fine-tuned Q&A model to answer the question.

## 3.4 Answering the question based on a knowledge base
Finally we can use a logic similar to the [/answers](https://beta.openai.com/docs/api-reference/answers) endpoint, where we first search for the relevant context, and then ask a Q&A model to answer the question given that context. If you'd like to see the implementation details, check out the [`answers_with_ft.py`](answers_with_ft.py) file.


```python
from answers_with_ft import answer_question
answer_question(olympics_search_fileid, ft_qa, "Which country won the Women's football tournament at the 2020 Olympic games?")
```




    " Canada won the Women's football tournament at the 2020 Olympic games"


