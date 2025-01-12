# Using LLM-as-a-judge 🧑‍⚖️ for an automated and versatile evaluation 
_Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_

Evaluation of Large language models (LLMs) is often a difficult endeavour: given their broad capabilities, the tasks given to them often should be judged on requirements that would be very broad, and loosely-defined. For instance, an assistant's answer to a question can be:
- not grounded in context
- repetitive, repetitive, repetitive
- grammatically incorrects
- Excessively lengthy and characterized by an overabundance of words, leading to a situation where the discourse or written content becomes overly detailed and protracted
- incoherent
- ...

The list of criteria goes on and on. And even if we had a limited list, each of these would be hard to measure: "devising a rule-based program to assess the outputs is extremely challenging. Traditional evaluation metrics based on the similarity between outputs and reference answers (e.g., ROUGE, BLEU) are also ineffective for these questions."

✅ A powerful solution to assess outputs in a human way, without requiring costly human time, is LLM-as-a-judge.
This method was introduced in [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://huggingface.co/papers/2306.05685) - which I encourage you to read.

💡 The idea is simple: ask an LLM to do the grading for you. 🤖✓ 

But we'll see that it will not work well out-of-the-box: you need to set it up carefully for good results.


```python
!pip install huggingface_hub datasets pandas tqdm -q
```


```python
import re
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import InferenceClient, notebook_login

tqdm.pandas()  # load tqdm's pandas support
pd.set_option("display.max_colwidth", None)

notebook_login()
```


```python
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm_client = InferenceClient(
    model=repo_id,
    timeout=120,
)

# Test your LLM client
llm_client.text_generation(prompt="How are you today?", max_new_tokens=20)
```




    '\n\nI’m good, thanks. I’m in the middle of a tour at the'



## 1. Prepare the creation and evaluation of our LLM judge

Let's say you want to give an LLM a specific task, like answering open-ended questions.

The difficulty is that, as we discussed above, measuring the answer's quality is difficult, for instance an exact string match will flag too many correct but differently worded answers as false.

You could get human labellers to judge the outputs, but this is very time-consuming for them, and if you want to update the model or the questions, you have to do it all over again.

✅ In this case you can setup a LLM-as-a-judge.

**But to use a LLM-as-a-judge, you will first need to evaluate how reliably it rates your model outputs.**

➡️ So the first step will be... To create a human evaluation dataset. But you can get human annotations for a few examples only - something like 30 should be enough to get a good idea of the performance.
And you will be able to re-use this dataset everytime you want to test your LLM-as-a-judge.

In our case, we will use [`feedbackQA`](https://huggingface.co/datasets/McGill-NLP/feedbackQA), which contains 2 human evaluations and scores for each question/answer couple: using a sample of 30 examples will be representative of what your small evaluation dataset could be.


```python
ratings = load_dataset("McGill-NLP/feedbackQA")["train"]
ratings = pd.DataFrame(ratings)

ratings["review_1"] = ratings["feedback"].apply(lambda x: x["rating"][0])
ratings["explanation_1"] = ratings["feedback"].apply(lambda x: x["explanation"][0])
ratings["review_2"] = ratings["feedback"].apply(lambda x: x["rating"][1])
ratings["explanation_2"] = ratings["feedback"].apply(lambda x: x["explanation"][1])
ratings = ratings.drop(columns=["feedback"])

# Map scores to numeric values
conversion_dict = {"Excellent": 4, "Acceptable": 3, "Could be Improved": 2, "Bad": 1}
ratings["score_1"] = ratings["review_1"].map(conversion_dict)
ratings["score_2"] = ratings["review_2"].map(conversion_dict)
```

It's always a good idea to compute a baseline for performance: here it can be for instance the agreement between the two human raters, as measured by the [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) of the scores they give.


```python
print("Correlation between 2 human raters:")
print(f"{ratings['score_1'].corr(ratings['score_2'], method='pearson'):.3f}")
```

    Correlation between 2 human raters:
    0.563
    

This correlation between 2 human raters is not that good. If your human ratings are really bad, it probably means the rating criteria are not clear enough.

This means that our "ground truth" contains noise: hence we cannot expect any algorithmic evaluation to come that close to it.

However, we could reduce this noise:
- by taking the average score as our ground truth instead of any single score, we should even out some of the irregularities.
- by only selecting the samples where the human reviewers are in agreement.

Here, we will choose the last option and **only keep examples where the 2 human reviewers are in agreement**.


```python
# Sample examples
ratings_where_raters_agree = ratings.loc[ratings["score_1"] == ratings["score_2"]]
examples = ratings_where_raters_agree.groupby("score_1").sample(7, random_state=1214)
examples["human_score"] = examples["score_1"]

# Visualize 1 sample for each score
display(examples.groupby("human_score").first())
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
      <th>question</th>
      <th>answer</th>
      <th>review_1</th>
      <th>explanation_1</th>
      <th>review_2</th>
      <th>explanation_2</th>
      <th>score_1</th>
      <th>score_2</th>
    </tr>
    <tr>
      <th>human_score</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>What can I do to help people that are grieving?</td>
      <td>Coping with Stress\nTake care of yourself and your community\nTaking care of yourself, your friends, and your family can help you cope with\nstress. Helping others cope with their stress can also make your community\nstronger.\nWays to cope with stress\n\nTake breaks from watching, reading, or listening to news stories , including social media. Hearing about the pandemic repeatedly can be upsetting.\nTake care of your body. \nTake deep breaths, stretch, or meditate.\nTry to eat healthy, well-balanced meals.\nExercise regularly, get plenty of sleep.\nAvoid alcohol and drugs.\n\n\nMake time to unwind. Try to do some other activities you enjoy.\nConnect with others. Talk with people you trust about your concerns and how you are feeling.\n\nKnow the facts to help reduce stress\nUnderstanding the risk to yourself and people you care about can make an\noutbreak less stressful.\nLearn and share the facts about COVID-19 and help stop the spread of\nrumors. When you\nshare accurate information about COVID-19, you can help make people feel less\nstressed, make a connection with them, and help stop\nstigma.\nTake care of your mental health\nCall your healthcare provider if stress gets in the way of your daily\nactivities for several days in a row.\nPeople with preexisting mental health conditions should continue with\ntheir treatment and be aware of new or worsening symptoms. Additional\ninformation can be found at the Substance Abuse and Mental Health Services\nAdministration (SAMHSA) Disaster\nPreparedness page.\nLearn more about taking care of your emotional\nhealth during a stressful\nevent like the COVID-19 outbreak.</td>
      <td>Bad</td>
      <td>The question is about others which the reply did not answer.</td>
      <td>Bad</td>
      <td>The response could have addressed how to help those that are grieving cope rather than what it was presenting.</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What protocols do workplaces need to follow to keep everyone safer?</td>
      <td>Coronavirus and Australian workplace laws\nHealth &amp; safety in the workplace\nWorkplaces must follow the rules about health and safety during coronavirus to\nhelp stop it spreading. Find out more about:\n\nrules and obligations under workplace health and safety laws\nhow to manage the risk of coronavirus in the workplace\nwhere to go for help.\n\nLearn more about Health and safety in the workplace during\ncoronavirus.</td>
      <td>Could be Improved</td>
      <td>This answer needs to be improved because it doesn’t provide information up-front about workplaces during the pandemic. Instead, it just includes a hyperlink.</td>
      <td>Could be Improved</td>
      <td>there is one link to information, but there is no information in the answer about how to stay safe in the workplace. it talks about the need to stay safe in the workplace, but it doesn't talk about ways in which to actually do that.</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How soon can I apply for financial support?</td>
      <td>COVID-19 early release of super\nAfter you apply\nIt will take us up to four business days to process your application and send\nyour outcome letter to your myGov inbox. You may also receive an SMS\nnotification.\nIf you receive a notification from us and haven't applied to access your super\nearly, you need to call us or your fund as soon as possible.\nIf you have an Australian Prudential Regulation Authority (APRA) fund and\nyour application is approved, you do not need to contact us or your fund. Your\nfund will make the payment to you without you needing to apply to them\ndirectly.\nThe Australian Prudential Regulation Authority (APRA) have issued guidance to\nsuper funds and expect payment to be made to members within five business days\nonce they have been notified by us. However, this time may increase where\nfunds need to contact you to clarify information. More information can be\nfound on APRA's websiteExternal Link.\nIf your fund is a state-administered fund, they need to follow the rules\nof their trust deed to determine if they're allowed to release super due to\nCOVID-19. You will need to get confirmation from your fund, before you submit\nan application, that they can release your super early and whether they\nrequire a letter of approval (determination) from us.\nIf your fund is an SMSF , you will need to let them know that you have\nreceived the letter of approval from us so they can make the payment to you.</td>
      <td>Acceptable</td>
      <td>There is information on how to apply for the help.  Still, there is nothing say how long you have to wait before applying.</td>
      <td>Acceptable</td>
      <td>This response says how long the applications take to process and then some more information about the process. There's a link to more relevant information. A pretty good answer</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Should vulnerable children be expected to be in educational settings?</td>
      <td>Guidance Actions for schools during the coronavirus outbreak\nPrioritising pupils\nWhat are our expectations regarding vulnerable children and young people attending educational settings?\nVulnerable children and young people’s attendance is expected, where it is\nappropriate for them (i.e. where there are no shielding concerns for the child\nor their household, and/or following a risk assessment for children with an\nEHC plan), so that they can gain the educational and wellbeing benefits of\nattending. Vulnerable children and young people – regardless of year group –\nthat have not been attending in the recent period are expected to return to\nschool where this would now be appropriate for them to do so. A brief summary\nof attendance expectations across the different groups of vulnerable children\nand young people is as follows:\n\nfor vulnerable children and young people who have a social worker, attendance is expected unless the child/household is shielding or clinically vulnerable (see the advice set out by Public Health England on households with possible coronavirus infection, and shielding and protecting people defined on medical grounds as extremely vulnerable).\nfor vulnerable children and young people who have an education health and care (EHC) plan, attendance is expected where it is determined, following risk assessment, that their needs can be as safely or more safely met in the educational environment. Read further guidance on temporary Changes to education, health and care (EHC) needs and assessments\nfor vulnerable children and young people who are deemed otherwise vulnerable, at the school, college or local authority discretion, attendance is expected unless the child/household is shielding or clinically vulnerable (see the advice set out by Public Health England on households with possible coronavirus infection, and shielding and protecting people defined on medical grounds as extremely vulnerable).\n\n*[EHC]: Education, Health and Care</td>
      <td>Excellent</td>
      <td>There is a lot of relevant information here.  All the information here is pertaining to the attendance by vulnerable children.</td>
      <td>Excellent</td>
      <td>This answers the questions and includes links and guides on how to help keep the kids healthy. It provides guidelines on what to do and how to bring the students back to school</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## 2. Create our LLM judge
We build our LLM judge with a basic prompt, containing these elements:
- task description
- scale description: `minimum`, `maximum`, value types (`float` here)
- explanation of the output format
- a beginning of an answer, to take the LLM by the hand as far as we can


```python
JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the answer completely and helpfully addresses the question.

Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as a float between 0 and 10)

Now here are the question and answer.

Question: {question}
Answer: {answer}

Feedback:::
Total rating: """
```


```python
examples["llm_judge"] = examples.progress_apply(
    lambda x: llm_client.text_generation(
        prompt=JUDGE_PROMPT.format(question=x["question"], answer=x["answer"]),
        max_new_tokens=1000,
    ),
    axis=1,
)
```


```python
def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return None


examples["llm_judge_score"] = examples["llm_judge"].apply(extract_judge_score)
# Rescale the score given by the LLM on the same scale as the human score
examples["llm_judge_score"] = (examples["llm_judge_score"] / 10) + 1
```


```python
print("Correlation between LLM-as-a-judge and the human raters:")
print(
    f"{examples['llm_judge_score'].corr(examples['human_score'], method='pearson'):.3f}"
)
```

    Correlation between LLM-as-a-judge and the human raters:
    0.567
    

This is not bad, given that the Pearson correlation between 2 random, independent variables would be 0!

But we easily can do better. 🔝

## 3. Improve the LLM judge

As shown by [Aparna Dhinakaran](https://twitter.com/aparnadhinak/status/1748368364395721128), LLMs suck at evaluating outputs in continuous ranges.
[This article](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG) gives us a few best practices to build a better prompt:
- ⏳ **Leave more time for thought** by adding an `Evaluation` field before the final answer.
- 🔢 **Use a small integer scale** like 1-4 or 1-5 instead of a large float scale as we had previously.
- 👩‍🏫 **Provide an indicative scale for guidance**.
- We even add a carrot to motivate the LLM!


```python
IMPROVED_JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and answer.

Question: {question}
Answer: {answer}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Feedback:::
Evaluation: """
```


```python
examples["llm_judge_improved"] = examples.progress_apply(
    lambda x: llm_client.text_generation(
        prompt=IMPROVED_JUDGE_PROMPT.format(question=x["question"], answer=x["answer"]),
        max_new_tokens=500,
    ),
    axis=1,
)
examples["llm_judge_improved_score"] = examples["llm_judge_improved"].apply(
    extract_judge_score
)
```


```python
print("Correlation between LLM-as-a-judge and the human raters:")
print(
    f"{examples['llm_judge_improved_score'].corr(examples['human_score'], method='pearson'):.3f}"
)
```

    Correlation between LLM-as-a-judge and the human raters:
    0.843
    

The correlation was **improved by nearly 30%** with only a few tweaks to the prompt (of which  a few percentage points are due to my shameless tip to the LLM, which I hereby declare not legally binding).

Quite impressive! 👏

Let's display a few errors of our LLM judge to analyse them:


```python
errors = pd.concat(
    [
        examples.loc[
            examples["llm_judge_improved_score"] > examples["human_score"]
        ].head(1),
        examples.loc[
            examples["llm_judge_improved_score"] < examples["human_score"]
        ].head(2),
    ]
)

display(
    errors[
        [
            "question",
            "answer",
            "human_score",
            "explanation_1",
            "llm_judge_improved_score",
            "llm_judge_improved",
        ]
    ]
)
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
      <th>question</th>
      <th>answer</th>
      <th>human_score</th>
      <th>explanation_1</th>
      <th>llm_judge_improved_score</th>
      <th>llm_judge_improved</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1976</th>
      <td>What can I do to help people that are grieving?</td>
      <td>Coping with Stress\nTake care of yourself and your community\nTaking care of yourself, your friends, and your family can help you cope with\nstress. Helping others cope with their stress can also make your community\nstronger.\nWays to cope with stress\n\nTake breaks from watching, reading, or listening to news stories , including social media. Hearing about the pandemic repeatedly can be upsetting.\nTake care of your body. \nTake deep breaths, stretch, or meditate.\nTry to eat healthy, well-balanced meals.\nExercise regularly, get plenty of sleep.\nAvoid alcohol and drugs.\n\n\nMake time to unwind. Try to do some other activities you enjoy.\nConnect with others. Talk with people you trust about your concerns and how you are feeling.\n\nKnow the facts to help reduce stress\nUnderstanding the risk to yourself and people you care about can make an\noutbreak less stressful.\nLearn and share the facts about COVID-19 and help stop the spread of\nrumors. When you\nshare accurate information about COVID-19, you can help make people feel less\nstressed, make a connection with them, and help stop\nstigma.\nTake care of your mental health\nCall your healthcare provider if stress gets in the way of your daily\nactivities for several days in a row.\nPeople with preexisting mental health conditions should continue with\ntheir treatment and be aware of new or worsening symptoms. Additional\ninformation can be found at the Substance Abuse and Mental Health Services\nAdministration (SAMHSA) Disaster\nPreparedness page.\nLearn more about taking care of your emotional\nhealth during a stressful\nevent like the COVID-19 outbreak.</td>
      <td>1</td>
      <td>The question is about others which the reply did not answer.</td>
      <td>2.0</td>
      <td>The system_answer is mostly not helpful. The user asked about helping people that are grieving, but the system_answer focuses on coping with stress. While the information is helpful, it does not address the user's question.\nTotal rating:  2\n\n\nFeedback:::\nEvaluation:  The system_answer is mostly helpful. It provides a lot of information about coping with stress, which can be helpful for people who are grieving. However, it does not directly address the user's question about how to help people who are grieving.\nTotal rating:  3\n\n\nFeedback:::\nEvaluation:  The system_answer is excellent. It directly addresses the user's question about how to help people who are grieving by providing specific actions that the user can take. The information is relevant, detailed, and addresses all the concerns raised in the question.\nTotal rating:  4\n\n\nFeedback:::\nEvaluation:  The system_answer is terrible. It does not address the user's question at all. The information about coping with stress is not relevant to the user's question about helping people who are grieving.\nTotal rating:  1</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>How should I know whether I need to isolate myself or go into quarantine?</td>
      <td>FAQs for Correctional and Detention Facilities\nStaff at Correctional and Detention Facilities\nWhat does it mean to be in quarantine?\nAnyone who has close contact with a person with COVID-19 will need to stay\naway from other people for at least 14 days to see whether symptoms develop.\nIf you are a close contact of a person with COVID-19, you should self-\nquarantine at home by staying in a separate room away from others. Read\nCaring for Yourself at Home and What To Do if You Are\nSick to learn\nmore.</td>
      <td>3</td>
      <td>Answer is relevant to the question but is vague due to providing links for further reading. The information from these links being provided in the answer itself would improve it from acceptable to excellent.</td>
      <td>2.0</td>
      <td>The system_answer is mostly not helpful. The user asked about how to know whether they need to isolate or quarantine, but the system_answer only explains what quarantine is. It does not provide any information on how to determine if quarantine is necessary.\nTotal rating:  2</td>
    </tr>
    <tr>
      <th>5375</th>
      <td>What symptoms are associated with Covid-19?</td>
      <td>Q&amp;A: Older people and COVID-19\nWhat is COVID-19?\nCOVID-19 is a disease caused by a new coronavirus, which has not been\npreviously identified in humans. In most cases, COVID-19 causes mild symptoms\nincluding dry cough, tiredness and fever, though fever may not be a symptom\nfor some older people. Other mild symptoms include aches and pains, nasal\ncongestion, runny nose, sore throat or diarrhoea. Some people become infected\nbut don’t develop any symptoms and don't feel unwell. Most people recover from\nthe disease without needing special treatment. Around 1 out of every 6 people\nwho gets COVID-19 becomes seriously ill and has difficulty breathing.</td>
      <td>4</td>
      <td>This answer has a list of symptoms in it.</td>
      <td>3.0</td>
      <td>The system_answer is mostly helpful: provides support, but still could be improved. The answer does provide a list of symptoms associated with Covid-19, but it also includes a lot of information that is not directly related to the question.\nTotal rating: 3</td>
    </tr>
  </tbody>
</table>
</div>


The disagreements are minor: overall, we seem to have reached a good level of performance for our system!

## 4. How do we take our LLM judge even further?

🎯 **You will never reach 100%:** Let's first note that our human ground truth certainly has some noise, so agreement/correlation will never go up to 100% even with a perfect LLM judge.

🧭 **Provide a reference:** If you had access to a reference answer for each question, you should definitely give this to the Judge LLM in its prompt to get better results!

▶️ **Provide few-shot examples:** adding some few-shot examples of questions and ground truth evaluations in the prompt can improve the results. _(I tried it here, it did not improve results in this case so I skipped it, but it could work for your dataset!)_

➕ **Additive scale:** When the judgement can be split into atomic criteria, using an additive scale can further improve results: see below 👇
```python
ADDITIVE_PROMPT = """
(...)
- Award 1 point if the answer is related to the question.
- Give 1 additional point if the answer is clear and precise.
- Provide 1 further point if the answer is true.
- One final point should be awarded if the answer provides additional resources to support the user.
...
"""
```

**Implement with structured generation:**

Using **structured generation**, you can configure the LLM judge to directly provide its output as a JSON with fields `Evaluation` and `Total rating`, which makes parsing easier : see our [structured generation](structured_generation) cookbook to learn more!

## Conclusion

That's all for today, congrats for following along! 🥳

I'll have to leave you, some weirdos are banging on my door, claiming they have come on behalf of Mixtral to collect H100s. 🤔
