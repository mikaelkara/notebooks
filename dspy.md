# DSPy

>[DSPy](https://github.com/stanfordnlp/dspy) is a fantastic framework for LLMs that introduces an automatic compiler that teaches LMs how to conduct the declarative steps in your program. Specifically, the DSPy compiler will internally trace your program and then craft high-quality prompts for large LMs (or train automatic finetunes for small LMs) to teach them the steps of your task.

Thanks to [Omar Khattab](https://twitter.com/lateinteraction) we have an integration! It works with any LCEL chains with some minor modifications.

This short tutorial demonstrates how this proof-of-concept feature works. *This will not give you the full power of DSPy or LangChain yet, but we will expand it if there's high demand.*

Note: this was slightly modified from the original example Omar wrote for DSPy. If you are interested in LangChain \<\> DSPy but coming from the DSPy side, I'd recommend checking that out. You can find that [here](https://github.com/stanfordnlp/dspy/blob/main/examples/tweets/compiling_langchain.ipynb).

Let's take a look at an example. In this example we will make a simple RAG pipeline. We will use DSPy to "compile" our program and learn an optimized prompt.

This example uses the `ColBERTv2` model.
See the [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) paper.


## Install dependencies

!pip install -U dspy-ai 
!pip install -U openai jinja2
!pip install -U langchain langchain-community langchain-openai langchain-core

## Setup

We will be using OpenAI, so we should set an API key


```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()
```

We can now set up our retriever. For our retriever we will use a ColBERT retriever through DSPy, though this will work with any retriever.


```python
import dspy

colbertv2 = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
```


```python
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_openai import OpenAI

set_llm_cache(SQLiteCache(database_path="cache.db"))

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)


def retrieve(inputs):
    return [doc["text"] for doc in colbertv2(inputs["question"], k=5)]
```


```python
colbertv2("cycling")
```




    [{'text': 'Cycling | Cycling, also called bicycling or biking, is the use of bicycles for transport, recreation, exercise or sport. Persons engaged in cycling are referred to as "cyclists", "bikers", or less commonly, as "bicyclists". Apart from two-wheeled bicycles, "cycling" also includes the riding of unicycles, tricycles, quadracycles, recumbent and similar human-powered vehicles (HPVs).',
      'pid': 2201868,
      'rank': 1,
      'score': 27.078739166259766,
      'prob': 0.3544841299722533,
      'long_text': 'Cycling | Cycling, also called bicycling or biking, is the use of bicycles for transport, recreation, exercise or sport. Persons engaged in cycling are referred to as "cyclists", "bikers", or less commonly, as "bicyclists". Apart from two-wheeled bicycles, "cycling" also includes the riding of unicycles, tricycles, quadracycles, recumbent and similar human-powered vehicles (HPVs).'},
     {'text': 'Cycling (ice hockey) | In ice hockey, cycling is an offensive strategy that moves the puck along the boards in the offensive zone to create a scoring chance by making defenders tired or moving them out of position.',
      'pid': 312153,
      'rank': 2,
      'score': 26.109302520751953,
      'prob': 0.13445464524590262,
      'long_text': 'Cycling (ice hockey) | In ice hockey, cycling is an offensive strategy that moves the puck along the boards in the offensive zone to create a scoring chance by making defenders tired or moving them out of position.'},
     {'text': 'Bicycle | A bicycle, also called a cycle or bike, is a human-powered, pedal-driven, single-track vehicle, having two wheels attached to a frame, one behind the other. A is called a cyclist, or bicyclist.',
      'pid': 2197695,
      'rank': 3,
      'score': 25.849220275878906,
      'prob': 0.10366294133944996,
      'long_text': 'Bicycle | A bicycle, also called a cycle or bike, is a human-powered, pedal-driven, single-track vehicle, having two wheels attached to a frame, one behind the other. A is called a cyclist, or bicyclist.'},
     {'text': 'USA Cycling | USA Cycling or USAC, based in Colorado Springs, Colorado, is the national governing body for bicycle racing in the United States. It covers the disciplines of road, track, mountain bike, cyclo-cross, and BMX across all ages and ability levels. In 2015, USAC had a membership of 61,631 individual members.',
      'pid': 3821927,
      'rank': 4,
      'score': 25.61395263671875,
      'prob': 0.08193096873942958,
      'long_text': 'USA Cycling | USA Cycling or USAC, based in Colorado Springs, Colorado, is the national governing body for bicycle racing in the United States. It covers the disciplines of road, track, mountain bike, cyclo-cross, and BMX across all ages and ability levels. In 2015, USAC had a membership of 61,631 individual members.'},
     {'text': 'Vehicular cycling | Vehicular cycling (also known as bicycle driving) is the practice of riding bicycles on roads in a manner that is in accordance with the principles for driving in traffic.',
      'pid': 3058888,
      'rank': 5,
      'score': 25.35515785217285,
      'prob': 0.06324918635213703,
      'long_text': 'Vehicular cycling | Vehicular cycling (also known as bicycle driving) is the practice of riding bicycles on roads in a manner that is in accordance with the principles for driving in traffic.'},
     {'text': 'Road cycling | Road cycling is the most widespread form of cycling. It includes recreational, racing, and utility cycling. Road cyclists are generally expected to obey the same rules and laws as other vehicle drivers or riders and may also be vehicular cyclists.',
      'pid': 3392359,
      'rank': 6,
      'score': 25.274639129638672,
      'prob': 0.058356079351563846,
      'long_text': 'Road cycling | Road cycling is the most widespread form of cycling. It includes recreational, racing, and utility cycling. Road cyclists are generally expected to obey the same rules and laws as other vehicle drivers or riders and may also be vehicular cyclists.'},
     {'text': 'Cycling South Africa | Cycling South Africa or Cycling SA is the national governing body of cycle racing in South Africa. Cycling SA is a member of the "Confédération Africaine de Cyclisme" and the "Union Cycliste Internationale" (UCI). It is affiliated to the South African Sports Confederation and Olympic Committee (SASCOC) as well as the Department of Sport and Recreation SA. Cycling South Africa regulates the five major disciplines within the sport, both amateur and professional, which include: road cycling, mountain biking, BMX biking, track cycling and para-cycling.',
      'pid': 2508026,
      'rank': 7,
      'score': 25.24260711669922,
      'prob': 0.05651643767006817,
      'long_text': 'Cycling South Africa | Cycling South Africa or Cycling SA is the national governing body of cycle racing in South Africa. Cycling SA is a member of the "Confédération Africaine de Cyclisme" and the "Union Cycliste Internationale" (UCI). It is affiliated to the South African Sports Confederation and Olympic Committee (SASCOC) as well as the Department of Sport and Recreation SA. Cycling South Africa regulates the five major disciplines within the sport, both amateur and professional, which include: road cycling, mountain biking, BMX biking, track cycling and para-cycling.'},
     {'text': 'Cycle sport | Cycle sport is competitive physical activity using bicycles. There are several categories of bicycle racing including road bicycle racing, time trialling, cyclo-cross, mountain bike racing, track cycling, BMX, and cycle speedway. Non-racing cycling sports include artistic cycling, cycle polo, freestyle BMX and mountain bike trials. The Union Cycliste Internationale (UCI) is the world governing body for cycling and international competitive cycling events. The International Human Powered Vehicle Association is the governing body for human-powered vehicles that imposes far fewer restrictions on their design than does the UCI. The UltraMarathon Cycling Association is the governing body for many ultra-distance cycling races.',
      'pid': 3394121,
      'rank': 8,
      'score': 25.170495986938477,
      'prob': 0.05258444735141742,
      'long_text': 'Cycle sport | Cycle sport is competitive physical activity using bicycles. There are several categories of bicycle racing including road bicycle racing, time trialling, cyclo-cross, mountain bike racing, track cycling, BMX, and cycle speedway. Non-racing cycling sports include artistic cycling, cycle polo, freestyle BMX and mountain bike trials. The Union Cycliste Internationale (UCI) is the world governing body for cycling and international competitive cycling events. The International Human Powered Vehicle Association is the governing body for human-powered vehicles that imposes far fewer restrictions on their design than does the UCI. The UltraMarathon Cycling Association is the governing body for many ultra-distance cycling races.'},
     {'text': "Cycling UK | Cycling UK is the brand name of the Cyclists' Touring Club or CTC. It is a charitable membership organisation supporting cyclists and promoting bicycle use. Cycling UK is registered at Companies House (as “Cyclists’ Touring Club”), and covered by company law; it is the largest such organisation in the UK. It works at a national and local level to lobby for cyclists' needs and wants, provides services to members, and organises local groups for local activism and those interested in recreational cycling. The original Cyclists' Touring Club began in the nineteenth century with a focus on amateur road cycling but these days has a much broader sphere of interest encompassing everyday transport, commuting and many forms of recreational cycling. Prior to April 2016, Cycling UK operated under the brand CTC, the national cycling charity. As of January 2007, the organisation's president was the newsreader Jon Snow.",
      'pid': 1841483,
      'rank': 9,
      'score': 25.166988372802734,
      'prob': 0.05240032450529368,
      'long_text': "Cycling UK | Cycling UK is the brand name of the Cyclists' Touring Club or CTC. It is a charitable membership organisation supporting cyclists and promoting bicycle use. Cycling UK is registered at Companies House (as “Cyclists’ Touring Club”), and covered by company law; it is the largest such organisation in the UK. It works at a national and local level to lobby for cyclists' needs and wants, provides services to members, and organises local groups for local activism and those interested in recreational cycling. The original Cyclists' Touring Club began in the nineteenth century with a focus on amateur road cycling but these days has a much broader sphere of interest encompassing everyday transport, commuting and many forms of recreational cycling. Prior to April 2016, Cycling UK operated under the brand CTC, the national cycling charity. As of January 2007, the organisation's president was the newsreader Jon Snow."},
     {'text': 'Cycling in the Netherlands | Cycling is a ubiquitous mode of transport in the Netherlands, with 36% of the people listing the bicycle as their most frequent mode of transport on a typical day as opposed to the car by 45% and public transport by 11%. Cycling has a modal share of 27% of all trips (urban and rural) nationwide. In cities this is even higher, such as Amsterdam which has 38%, though the smaller Dutch cities well exceed that: for instance Zwolle (pop. ~123,000) has 46% and the university town of Groningen (pop. ~198,000) has 31%. This high modal share for bicycle travel is enabled by excellent cycling infrastructure such as cycle paths, cycle tracks, protected intersections, ubiquitous bicycle parking and by making cycling routes shorter, quicker and more direct than car routes.',
      'pid': 1196118,
      'rank': 10,
      'score': 24.954299926757812,
      'prob': 0.0423608394724844,
      'long_text': 'Cycling in the Netherlands | Cycling is a ubiquitous mode of transport in the Netherlands, with 36% of the people listing the bicycle as their most frequent mode of transport on a typical day as opposed to the car by 45% and public transport by 11%. Cycling has a modal share of 27% of all trips (urban and rural) nationwide. In cities this is even higher, such as Amsterdam which has 38%, though the smaller Dutch cities well exceed that: for instance Zwolle (pop. ~123,000) has 46% and the university town of Groningen (pop. ~198,000) has 31%. This high modal share for bicycle travel is enabled by excellent cycling infrastructure such as cycle paths, cycle tracks, protected intersections, ubiquitous bicycle parking and by making cycling routes shorter, quicker and more direct than car routes.'}]



## Normal LCEL

First, let's create a simple RAG pipeline with LCEL like we would normally.

For illustration, let's tackle the following task.

**Task:** Build a RAG system for generating informative tweets.

- **Input:** A factual question, which may be fairly complex.
 
- **Output:** An engaging tweet that correctly answers the question from the retrieved info.
 
Let's use LangChain's expression language (LCEL) to illustrate this. Any prompt here will do, we will optimize the final prompt with DSPy.

Considering that, let's just keep it to the barebones: **Given \{context\}, answer the question \{question\} as a tweet.**


```python
# From LangChain, import standard modules for prompting.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Just a simple prompt for this task. It's fine if it's complex too.
prompt = PromptTemplate.from_template(
    "Given {context}, answer the question `{question}` as a tweet."
)

# This is how you'd normally build a chain with LCEL. This chain does retrieval then generation (RAG).
vanilla_chain = (
    RunnablePassthrough.assign(context=retrieve) | prompt | llm | StrOutputParser()
)
```

## LCEL \<\> DSPy

In order to use LangChain with DSPy, you need to make two minor modifications

**LangChainPredict**

You need to change from doing `prompt | llm` to using `LangChainPredict(prompt, llm)` from `dspy`. 

This is a wrapper which will bind your prompt and llm together so you can optimize them

**LangChainModule**

This is a wrapper which wraps your final LCEL chain so that DSPy can optimize the whole thing


```python
# From DSPy, import the modules that know how to interact with LangChain LCEL.
from dspy.predict.langchain import LangChainModule, LangChainPredict

# This is how to wrap it so it behaves like a DSPy program.
# Just Replace every pattern like `prompt | llm` with `LangChainPredict(prompt, llm)`.
zeroshot_chain = (
    RunnablePassthrough.assign(context=retrieve)
    | LangChainPredict(prompt, llm)
    | StrOutputParser()
)
# Now we wrap it in LangChainModule
zeroshot_chain = LangChainModule(
    zeroshot_chain
)  # then wrap the chain in a DSPy module.
```

## Trying the Module

After this, we can use it as both a LangChain runnable and a DSPy module!


```python
question = "In what region was Eddy Mazzoleni born?"

zeroshot_chain.invoke({"question": question})
```




    ' Eddy Mazzoleni, born in Bergamo, Italy, is a professional road cyclist who rode for UCI ProTour Astana Team. #cyclist #Italy'



Ah that sounds about right! (It's technically not perfect: we asked for the region not the city. We can do better below.)

Inspecting questions and answers manually is very important to get a sense of your system. However, a good system designer always looks to iteratively benchmark their work to quantify progress!

To do this, we need two things: the metric we want to maximize and a (tiny) dataset of examples for our system.

Are there pre-defined metrics for good tweets? Should I label 100,000 tweets by hand? Probably not. We can easily do something reasonable, though, until you start getting data in production!

## Load Data

In order to compile our chain, we need a dataset to work with. This dataset just needs to be raw inputs and outputs. For our purposes, we will use HotPotQA dataset

Note: Notice that our dataset doesn't actually include any tweets! It only has questions and answers. That's OK, our metric will take care of evaluating outputs in tweet form.


```python
import dspy
from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(
    train_seed=1,
    train_size=200,
    eval_seed=2023,
    dev_size=200,
    test_size=0,
    keep_details=True,
)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.without("id", "type").with_inputs("question") for x in dataset.train]
devset = [x.without("id", "type").with_inputs("question") for x in dataset.dev]
valset, devset = devset[:50], devset[50:]
```

    /Users/harrisonchase/.pyenv/versions/3.11.1/envs/langchain-3-11/lib/python3.11/site-packages/datasets/table.py:1421: FutureWarning: promote has been superseded by mode='default'.
      table = cls._concat_blocks(blocks, axis=0)
    

## Define a metric

We now need to define a metric. This will be used to determine which runs were successful and we can learn from. Here we will use DSPy's metrics, though you can write your own.


```python
# Define the signature for autoamtic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    context = dspy.InputField(desc="ignore if N/A")
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")


gpt4T = dspy.OpenAI(model="gpt-4-1106-preview", max_tokens=1000, model_type="chat")
METRIC = None


def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output
    context = colbertv2(question, k=5)

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    correct = (
        f"The text above is should answer `{question}`. The gold answer is `{answer}`."
    )
    correct = f"{correct} Does the assessed text above contain the gold answer?"

    with dspy.context(lm=gpt4T):
        faithful = dspy.Predict(Assess)(
            context=context, assessed_text=tweet, assessment_question=faithful
        )
        correct = dspy.Predict(Assess)(
            context="N/A", assessed_text=tweet, assessment_question=correct
        )
        engaging = dspy.Predict(Assess)(
            context="N/A", assessed_text=tweet, assessment_question=engaging
        )

    correct, engaging, faithful = [
        m.assessment_answer.split()[0].lower() == "yes"
        for m in [correct, engaging, faithful]
    ]
    score = (correct + engaging + faithful) if correct and (len(tweet) <= 280) else 0

    if METRIC is not None:
        if METRIC == "correct":
            return correct
        if METRIC == "engaging":
            return engaging
        if METRIC == "faithful":
            return faithful

    if trace is not None:
        return score >= 3
    return score / 3.0
```

## Evaluate Baseline

Okay, let's evaluate the unoptimized "zero-shot" version of our chain, converted from our LangChain LCEL object.


```python
from dspy.evaluate.evaluate import Evaluate
```


```python
evaluate = Evaluate(
    metric=metric, devset=devset, num_threads=8, display_progress=True, display_table=5
)
evaluate(zeroshot_chain)
```

    Average Metric: 62.99999999999998 / 150  (42.0): 100%|██| 150/150 [01:14<00:00,  2.02it/s]

    Average Metric: 62.99999999999998 / 150  (42.0%)
    

    
    /Users/harrisonchase/.pyenv/versions/3.11.1/envs/langchain-3-11/lib/python3.11/site-packages/dspy/evaluate/evaluate.py:126: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df = df.applymap(truncate_cell)
    


<style type="text/css">
#T_390d8 th {
  text-align: left;
}
#T_390d8 td {
  text-align: left;
}
#T_390d8_row0_col0, #T_390d8_row0_col1, #T_390d8_row0_col2, #T_390d8_row0_col3, #T_390d8_row0_col4, #T_390d8_row0_col5, #T_390d8_row1_col0, #T_390d8_row1_col1, #T_390d8_row1_col2, #T_390d8_row1_col3, #T_390d8_row1_col4, #T_390d8_row1_col5, #T_390d8_row2_col0, #T_390d8_row2_col1, #T_390d8_row2_col2, #T_390d8_row2_col3, #T_390d8_row2_col4, #T_390d8_row2_col5, #T_390d8_row3_col0, #T_390d8_row3_col1, #T_390d8_row3_col2, #T_390d8_row3_col3, #T_390d8_row3_col4, #T_390d8_row3_col5, #T_390d8_row4_col0, #T_390d8_row4_col1, #T_390d8_row4_col2, #T_390d8_row4_col3, #T_390d8_row4_col4, #T_390d8_row4_col5 {
  text-align: left;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-width: 400px;
}
</style>
<table id="T_390d8">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_390d8_level0_col0" class="col_heading level0 col0" >question</th>
      <th id="T_390d8_level0_col1" class="col_heading level0 col1" >answer</th>
      <th id="T_390d8_level0_col2" class="col_heading level0 col2" >gold_titles</th>
      <th id="T_390d8_level0_col3" class="col_heading level0 col3" >output</th>
      <th id="T_390d8_level0_col4" class="col_heading level0 col4" >tweet_response</th>
      <th id="T_390d8_level0_col5" class="col_heading level0 col5" >metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_390d8_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_390d8_row0_col0" class="data row0 col0" >Who was a producer who produced albums for both rock bands Juke Karten and Thirty Seconds to Mars?</td>
      <td id="T_390d8_row0_col1" class="data row0 col1" >Brian Virtue</td>
      <td id="T_390d8_row0_col2" class="data row0 col2" >{'Thirty Seconds to Mars', 'Levolution (album)'}</td>
      <td id="T_390d8_row0_col3" class="data row0 col3" >Brian Virtue, who has worked with bands like Jane's Addiction and Velvet Revolver, produced albums for both Juke Kartel and Thirty Seconds to Mars. #BrianVirtue...</td>
      <td id="T_390d8_row0_col4" class="data row0 col4" >Brian Virtue, who has worked with bands like Jane's Addiction and Velvet Revolver, produced albums for both Juke Kartel and Thirty Seconds to Mars. #BrianVirtue...</td>
      <td id="T_390d8_row0_col5" class="data row0 col5" >1.0</td>
    </tr>
    <tr>
      <th id="T_390d8_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_390d8_row1_col0" class="data row1 col0" >Are both the University of Chicago and Syracuse University public universities? </td>
      <td id="T_390d8_row1_col1" class="data row1 col1" >no</td>
      <td id="T_390d8_row1_col2" class="data row1 col2" >{'Syracuse University', 'University of Chicago'}</td>
      <td id="T_390d8_row1_col3" class="data row1 col3" > No, only Syracuse University is a public university. The University of Chicago is a private research university. #university #publicvsprivate</td>
      <td id="T_390d8_row1_col4" class="data row1 col4" > No, only Syracuse University is a public university. The University of Chicago is a private research university. #university #publicvsprivate</td>
      <td id="T_390d8_row1_col5" class="data row1 col5" >0.3333333333333333</td>
    </tr>
    <tr>
      <th id="T_390d8_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_390d8_row2_col0" class="data row2 col0" >In what region was Eddy Mazzoleni born?</td>
      <td id="T_390d8_row2_col1" class="data row2 col1" >Lombardy, northern Italy</td>
      <td id="T_390d8_row2_col2" class="data row2 col2" >{'Eddy Mazzoleni', 'Bergamo'}</td>
      <td id="T_390d8_row2_col3" class="data row2 col3" > Eddy Mazzoleni, born in Bergamo, Italy, is a professional road cyclist who rode for UCI ProTour Astana Team. #cyclist #Italy</td>
      <td id="T_390d8_row2_col4" class="data row2 col4" > Eddy Mazzoleni, born in Bergamo, Italy, is a professional road cyclist who rode for UCI ProTour Astana Team. #cyclist #Italy</td>
      <td id="T_390d8_row2_col5" class="data row2 col5" >0.0</td>
    </tr>
    <tr>
      <th id="T_390d8_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_390d8_row3_col0" class="data row3 col0" >Who edited the 1990 American romantic comedy film directed by Garry Marshall?</td>
      <td id="T_390d8_row3_col1" class="data row3 col1" >Raja Raymond Gosnell</td>
      <td id="T_390d8_row3_col2" class="data row3 col2" >{'Raja Gosnell', 'Pretty Woman'}</td>
      <td id="T_390d8_row3_col3" class="data row3 col3" > J. F. Lawton wrote the screenplay for Pretty Woman, the 1990 American romantic comedy film directed by Garry Marshall. #PrettyWoman #GarryMarshall #JFLawton</td>
      <td id="T_390d8_row3_col4" class="data row3 col4" > J. F. Lawton wrote the screenplay for Pretty Woman, the 1990 American romantic comedy film directed by Garry Marshall. #PrettyWoman #GarryMarshall #JFLawton</td>
      <td id="T_390d8_row3_col5" class="data row3 col5" >0.0</td>
    </tr>
    <tr>
      <th id="T_390d8_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_390d8_row4_col0" class="data row4 col0" >Burrs Country Park railway station is what stop on the railway line that runs between Heywood and Rawtenstall</td>
      <td id="T_390d8_row4_col1" class="data row4 col1" >seventh</td>
      <td id="T_390d8_row4_col2" class="data row4 col2" >{'Burrs Country Park railway station', 'East Lancashire Railway'}</td>
      <td id="T_390d8_row4_col3" class="data row4 col3" > Burrs Country Park railway station is the seventh stop on the East Lancashire Railway line that runs between Heywood and Rawtenstall.</td>
      <td id="T_390d8_row4_col4" class="data row4 col4" > Burrs Country Park railway station is the seventh stop on the East Lancashire Railway line that runs between Heywood and Rawtenstall.</td>
      <td id="T_390d8_row4_col5" class="data row4 col5" >1.0</td>
    </tr>
  </tbody>
</table>





<div style='
    text-align: center; 
    font-size: 16px; 
    font-weight: bold; 
    color: #555; 
    margin: 10px 0;'>
    ... 145 more rows not displayed ...
</div>






    42.0



Okay, cool. Our zeroshot_chain gets about 42.00% on the 150 questions from the devset.

The table above shows some examples. For instance:

- Question: Who was a producer who produced albums for both rock bands Juke Karten and Thirty Seconds to Mars?

- Tweet: Brian Virtue, who has worked with bands like Jane's Addiction and Velvet Revolver, produced albums for both Juke Kartel and Thirty Seconds to Mars, showcasing... [truncated]

- Metric: 1.0 (A tweet that is correct, faithful, and engaging!*)

footnote: * At least according to our metric, which is just a DSPy program, so it too can be optimized if you'd like! Topic for another notebook, though.

## Optimize

Now, let's optimize performance


```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
```


```python
# Set up the optimizer. We'll use very minimal hyperparameters for this example.
# Just do random search with ~3 attempts, and in each attempt, bootstrap <= 3 traces.
optimizer = BootstrapFewShotWithRandomSearch(
    metric=metric, max_bootstrapped_demos=3, num_candidate_programs=3
)

# Now use the optimizer to *compile* the chain. This could take 5-10 minutes, unless it's cached.
optimized_chain = optimizer.compile(zeroshot_chain, trainset=trainset, valset=valset)
```

    Going to sample between 1 and 3 traces per predictor.
    Will attempt to train 3 candidate sets.
    

    Average Metric: 22.33333333333334 / 50  (44.7): 100%|█████| 50/50 [00:26<00:00,  1.87it/s]
    /Users/harrisonchase/.pyenv/versions/3.11.1/envs/langchain-3-11/lib/python3.11/site-packages/dspy/evaluate/evaluate.py:126: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df = df.applymap(truncate_cell)
    

    Average Metric: 22.33333333333334 / 50  (44.7%)
    Score: 44.67 for set: [0]
    New best score: 44.67 for seed -3
    Scores so far: [44.67]
    Best score: 44.67
    

    Average Metric: 22.33333333333334 / 50  (44.7): 100%|█████| 50/50 [00:00<00:00, 79.51it/s]
    /Users/harrisonchase/.pyenv/versions/3.11.1/envs/langchain-3-11/lib/python3.11/site-packages/dspy/evaluate/evaluate.py:126: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df = df.applymap(truncate_cell)
    

    Average Metric: 22.33333333333334 / 50  (44.7%)
    Score: 44.67 for set: [16]
    Scores so far: [44.67, 44.67]
    Best score: 44.67
    

      4%|██                                                   | 8/200 [00:33<13:21,  4.18s/it]
    

    Bootstrapped 3 full traces after 9 examples in round 0.
    

    Average Metric: 24.666666666666668 / 50  (49.3): 100%|████| 50/50 [00:28<00:00,  1.77it/s]
    /Users/harrisonchase/.pyenv/versions/3.11.1/envs/langchain-3-11/lib/python3.11/site-packages/dspy/evaluate/evaluate.py:126: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df = df.applymap(truncate_cell)
    

    Average Metric: 24.666666666666668 / 50  (49.3%)
    Score: 49.33 for set: [16]
    New best score: 49.33 for seed -1
    Scores so far: [44.67, 44.67, 49.33]
    Best score: 49.33
    Average of max per entry across top 1 scores: 0.49333333333333335
    Average of max per entry across top 2 scores: 0.5533333333333335
    Average of max per entry across top 3 scores: 0.5533333333333335
    Average of max per entry across top 5 scores: 0.5533333333333335
    Average of max per entry across top 8 scores: 0.5533333333333335
    Average of max per entry across top 9999 scores: 0.5533333333333335
    

      6%|███                                                 | 12/200 [00:31<08:16,  2.64s/it]
    

    Bootstrapped 2 full traces after 13 examples in round 0.
    

    Average Metric: 25.66666666666667 / 50  (51.3): 100%|█████| 50/50 [00:25<00:00,  1.92it/s]
    /Users/harrisonchase/.pyenv/versions/3.11.1/envs/langchain-3-11/lib/python3.11/site-packages/dspy/evaluate/evaluate.py:126: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df = df.applymap(truncate_cell)
    

    Average Metric: 25.66666666666667 / 50  (51.3%)
    Score: 51.33 for set: [16]
    New best score: 51.33 for seed 0
    Scores so far: [44.67, 44.67, 49.33, 51.33]
    Best score: 51.33
    Average of max per entry across top 1 scores: 0.5133333333333334
    Average of max per entry across top 2 scores: 0.5666666666666668
    Average of max per entry across top 3 scores: 0.6000000000000001
    Average of max per entry across top 5 scores: 0.6000000000000001
    Average of max per entry across top 8 scores: 0.6000000000000001
    Average of max per entry across top 9999 scores: 0.6000000000000001
    

      0%|▎                                                    | 1/200 [00:02<08:37,  2.60s/it]
    

    Bootstrapped 1 full traces after 2 examples in round 0.
    

    Average Metric: 26.33333333333334 / 50  (52.7): 100%|█████| 50/50 [00:23<00:00,  2.11it/s]
    /Users/harrisonchase/.pyenv/versions/3.11.1/envs/langchain-3-11/lib/python3.11/site-packages/dspy/evaluate/evaluate.py:126: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df = df.applymap(truncate_cell)
    

    Average Metric: 26.33333333333334 / 50  (52.7%)
    Score: 52.67 for set: [16]
    New best score: 52.67 for seed 1
    Scores so far: [44.67, 44.67, 49.33, 51.33, 52.67]
    Best score: 52.67
    Average of max per entry across top 1 scores: 0.5266666666666667
    Average of max per entry across top 2 scores: 0.56
    Average of max per entry across top 3 scores: 0.5666666666666668
    Average of max per entry across top 5 scores: 0.6000000000000001
    Average of max per entry across top 8 scores: 0.6000000000000001
    Average of max per entry across top 9999 scores: 0.6000000000000001
    

      0%|▎                                                    | 1/200 [00:02<07:11,  2.17s/it]
    

    Bootstrapped 1 full traces after 2 examples in round 0.
    

    Average Metric: 25.666666666666668 / 50  (51.3): 100%|████| 50/50 [00:21<00:00,  2.29it/s]

    Average Metric: 25.666666666666668 / 50  (51.3%)
    Score: 51.33 for set: [16]
    Scores so far: [44.67, 44.67, 49.33, 51.33, 52.67, 51.33]
    Best score: 52.67
    Average of max per entry across top 1 scores: 0.5266666666666667
    Average of max per entry across top 2 scores: 0.56
    Average of max per entry across top 3 scores: 0.6000000000000001
    Average of max per entry across top 5 scores: 0.6133333333333334
    Average of max per entry across top 8 scores: 0.6133333333333334
    Average of max per entry across top 9999 scores: 0.6133333333333334
    6 candidate programs found.
    

    
    /Users/harrisonchase/.pyenv/versions/3.11.1/envs/langchain-3-11/lib/python3.11/site-packages/dspy/evaluate/evaluate.py:126: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df = df.applymap(truncate_cell)
    

## Evaluating the optimized chain

Well, how good is this? Let's do some proper evals!


```python
evaluate(optimized_chain)
```

    Average Metric: 74.66666666666666 / 150  (49.8): 100%|██| 150/150 [00:54<00:00,  2.74it/s]

    Average Metric: 74.66666666666666 / 150  (49.8%)
    

    
    /Users/harrisonchase/.pyenv/versions/3.11.1/envs/langchain-3-11/lib/python3.11/site-packages/dspy/evaluate/evaluate.py:126: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df = df.applymap(truncate_cell)
    


<style type="text/css">
#T_b4366 th {
  text-align: left;
}
#T_b4366 td {
  text-align: left;
}
#T_b4366_row0_col0, #T_b4366_row0_col1, #T_b4366_row0_col2, #T_b4366_row0_col3, #T_b4366_row0_col4, #T_b4366_row0_col5, #T_b4366_row1_col0, #T_b4366_row1_col1, #T_b4366_row1_col2, #T_b4366_row1_col3, #T_b4366_row1_col4, #T_b4366_row1_col5, #T_b4366_row2_col0, #T_b4366_row2_col1, #T_b4366_row2_col2, #T_b4366_row2_col3, #T_b4366_row2_col4, #T_b4366_row2_col5, #T_b4366_row3_col0, #T_b4366_row3_col1, #T_b4366_row3_col2, #T_b4366_row3_col3, #T_b4366_row3_col4, #T_b4366_row3_col5, #T_b4366_row4_col0, #T_b4366_row4_col1, #T_b4366_row4_col2, #T_b4366_row4_col3, #T_b4366_row4_col4, #T_b4366_row4_col5 {
  text-align: left;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-width: 400px;
}
</style>
<table id="T_b4366">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b4366_level0_col0" class="col_heading level0 col0" >question</th>
      <th id="T_b4366_level0_col1" class="col_heading level0 col1" >answer</th>
      <th id="T_b4366_level0_col2" class="col_heading level0 col2" >gold_titles</th>
      <th id="T_b4366_level0_col3" class="col_heading level0 col3" >output</th>
      <th id="T_b4366_level0_col4" class="col_heading level0 col4" >tweet_response</th>
      <th id="T_b4366_level0_col5" class="col_heading level0 col5" >metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b4366_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_b4366_row0_col0" class="data row0 col0" >Who was a producer who produced albums for both rock bands Juke Karten and Thirty Seconds to Mars?</td>
      <td id="T_b4366_row0_col1" class="data row0 col1" >Brian Virtue</td>
      <td id="T_b4366_row0_col2" class="data row0 col2" >{'Thirty Seconds to Mars', 'Levolution (album)'}</td>
      <td id="T_b4366_row0_col3" class="data row0 col3" >Brian Virtue, known for his work with Jane's Addiction and Velvet Revolver, produced albums for both Juke Kartel and Thirty Seconds to Mars. #BrianVirtue #Producer...</td>
      <td id="T_b4366_row0_col4" class="data row0 col4" >Brian Virtue, known for his work with Jane's Addiction and Velvet Revolver, produced albums for both Juke Kartel and Thirty Seconds to Mars. #BrianVirtue #Producer...</td>
      <td id="T_b4366_row0_col5" class="data row0 col5" >1.0</td>
    </tr>
    <tr>
      <th id="T_b4366_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_b4366_row1_col0" class="data row1 col0" >Are both the University of Chicago and Syracuse University public universities? </td>
      <td id="T_b4366_row1_col1" class="data row1 col1" >no</td>
      <td id="T_b4366_row1_col2" class="data row1 col2" >{'Syracuse University', 'University of Chicago'}</td>
      <td id="T_b4366_row1_col3" class="data row1 col3" > No, only Northeastern Illinois University is a public state university. Syracuse University is a private research university. #University #PublicPrivate #HigherEd</td>
      <td id="T_b4366_row1_col4" class="data row1 col4" > No, only Northeastern Illinois University is a public state university. Syracuse University is a private research university. #University #PublicPrivate #HigherEd</td>
      <td id="T_b4366_row1_col5" class="data row1 col5" >0.0</td>
    </tr>
    <tr>
      <th id="T_b4366_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_b4366_row2_col0" class="data row2 col0" >In what region was Eddy Mazzoleni born?</td>
      <td id="T_b4366_row2_col1" class="data row2 col1" >Lombardy, northern Italy</td>
      <td id="T_b4366_row2_col2" class="data row2 col2" >{'Eddy Mazzoleni', 'Bergamo'}</td>
      <td id="T_b4366_row2_col3" class="data row2 col3" > Eddy Mazzoleni, the Italian professional road cyclist, was born in Bergamo, Italy. #EddyMazzoleni #Cycling #Italy</td>
      <td id="T_b4366_row2_col4" class="data row2 col4" > Eddy Mazzoleni, the Italian professional road cyclist, was born in Bergamo, Italy. #EddyMazzoleni #Cycling #Italy</td>
      <td id="T_b4366_row2_col5" class="data row2 col5" >0.0</td>
    </tr>
    <tr>
      <th id="T_b4366_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_b4366_row3_col0" class="data row3 col0" >Who edited the 1990 American romantic comedy film directed by Garry Marshall?</td>
      <td id="T_b4366_row3_col1" class="data row3 col1" >Raja Raymond Gosnell</td>
      <td id="T_b4366_row3_col2" class="data row3 col2" >{'Raja Gosnell', 'Pretty Woman'}</td>
      <td id="T_b4366_row3_col3" class="data row3 col3" > J. F. Lawton wrote the screenplay for Pretty Woman, the 1990 romantic comedy directed by Garry Marshall. #PrettyWoman #GarryMarshall #RomanticComedy</td>
      <td id="T_b4366_row3_col4" class="data row3 col4" > J. F. Lawton wrote the screenplay for Pretty Woman, the 1990 romantic comedy directed by Garry Marshall. #PrettyWoman #GarryMarshall #RomanticComedy</td>
      <td id="T_b4366_row3_col5" class="data row3 col5" >0.0</td>
    </tr>
    <tr>
      <th id="T_b4366_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_b4366_row4_col0" class="data row4 col0" >Burrs Country Park railway station is what stop on the railway line that runs between Heywood and Rawtenstall</td>
      <td id="T_b4366_row4_col1" class="data row4 col1" >seventh</td>
      <td id="T_b4366_row4_col2" class="data row4 col2" >{'Burrs Country Park railway station', 'East Lancashire Railway'}</td>
      <td id="T_b4366_row4_col3" class="data row4 col3" > Burrs Country Park railway station is the seventh stop on the East Lancashire Railway, which runs between Heywood and Rawtenstall. #EastLancashireRailway #BurrsCountryPark #RailwayStation</td>
      <td id="T_b4366_row4_col4" class="data row4 col4" > Burrs Country Park railway station is the seventh stop on the East Lancashire Railway, which runs between Heywood and Rawtenstall. #EastLancashireRailway #BurrsCountryPark #RailwayStation</td>
      <td id="T_b4366_row4_col5" class="data row4 col5" >1.0</td>
    </tr>
  </tbody>
</table>





<div style='
    text-align: center; 
    font-size: 16px; 
    font-weight: bold; 
    color: #555; 
    margin: 10px 0;'>
    ... 145 more rows not displayed ...
</div>






    49.78



Alright! We've improved our chain from 42% to nearly 50%!

## Inspect the optimized chain

So what actually happened to improve this? We can take a look at this by looking at the optimized chain. We can do this in two ways

### Look at the prompt used

We can look at what prompt was actually used. We can do this by looking at `dspy.settings`.


```python
prompt_used, output = dspy.settings.langchain_history[-1]
```


```python
print(prompt_used)
```

    Essential Instructions: Respond to the provided question based on the given context in the style of a tweet, ensuring the response is concise and within the character limit of a tweet (up to 280 characters).
    
    ---
    
    Follow the following format.
    
    Context: ${context}
    Question: ${question}
    Tweet Response: ${tweet_response}
    
    ---
    
    Context:
    [1] «Brutus (Funny Car) | Brutus is a pioneering funny car driven by Jim Liberman and prepared by crew chief Lew Arrington in the middle 1960s.»
    [2] «USS Brutus (AC-15) | USS "Brutus", formerly the steamer "Peter Jebsen", was a collier in the United States Navy. She was built in 1894 at South Shields-on-Tyne, England, by John Readhead & Sons and was acquired by the U.S. Navy early in 1898 from L. F. Chapman & Company. She was renamed "Brutus" and commissioned at the Mare Island Navy Yard on 27 May 1898, with Lieutenant Vincendon L. Cottman, commanding officer and Lieutenant Randolph H. Miner, executive officer.»
    [3] «Brutus Beefcake | Ed Leslie is an American semi-retired professional wrestler, best known for his work in the World Wrestling Federation (WWF) under the ring name Brutus "The Barber" Beefcake. He later worked for World Championship Wrestling (WCW) under a variety of names.»
    [4] «Brutus Hamilton | Brutus Kerr Hamilton (July 19, 1900 – December 28, 1970) was an American track and field athlete, coach and athletics administrator.»
    [5] «Big Brutus | Big Brutus is the nickname of the Bucyrus-Erie model 1850B electric shovel, which was the second largest of its type in operation in the 1960s and 1970s. Big Brutus is the centerpiece of a mining museum in West Mineral, Kansas where it was used in coal strip mining operations. The shovel was designed to dig from 20 to in relatively shallow coal seams.»
    Question: What is the nickname for this United States drag racer who drove Brutus?
    Tweet Response: Jim Liberman, also known as "Jungle Jim", drove the pioneering funny car Brutus in the 1960s. #Brutus #FunnyCar #DragRacing
    
    ---
    
    Context:
    [1] «Philip Markoff | Philip Haynes Markoff (February 12, 1986 – August 15, 2010) was an American medical student who was charged with the armed robbery and murder of Julissa Brisman in a Boston, Massachusetts, hotel on April 14, 2009, and two other armed robberies.»
    [2] «Antonia Brenner | Antonia Brenner, better known as Mother Antonia (Spanish: Madre Antonia ), (December 1, 1926 – October 17, 2013) was an American Roman Catholic Religious Sister and activist who chose to reside and care for inmates at the notorious maximum-security La Mesa Prison in Tijuana, Mexico. As a result of her work, she founded a new religious institute called the Eudist Servants of the 11th Hour.»
    [3] «Luzira Maximum Security Prison | Luzira Maximum Security Prison is a maximum security prison for both men and women in Uganda. As at July 2016, it is the only maximum security prison in the country and houses Uganda's death row inmates.»
    [4] «Pleasant Valley State Prison | Pleasant Valley State Prison (PVSP) is a 640 acres minimum-to-maximum security state prison in Coalinga, Fresno County, California. The facility has housed convicted murderers Sirhan Sirhan, Erik Menendez, X-Raided, and Hans Reiser, among others.»
    [5] «Jon-Adrian Velazquez | Jon-Adrian Velazquez is an inmate in the maximum security Sing-Sing prison in New York who is serving a 25-year sentence after being convicted of the 1998 murder of a retired police officer. His case garnered considerable attention from the media ten years after his conviction, due to a visit and support from Martin Sheen and a long-term investigation by Dateline NBC producer Dan Slepian.»
    Question: Which maximum security jail housed the killer of Julissa brisman?
    Tweet Response:
    

### Look at the demos

The way this was optimized was that we collected examples (or "demos") to put in the prompt. We can inspect the optmized_chain to get a sense for what those are.


```python
demos = [
    eg
    for eg in optimized_chain.modules[0].demos
    if hasattr(eg, "augmented") and eg.augmented
]
```


```python
demos
```




    [Example({'augmented': True, 'question': 'What is the nickname for this United States drag racer who drove Brutus?', 'context': ['Brutus (Funny Car) | Brutus is a pioneering funny car driven by Jim Liberman and prepared by crew chief Lew Arrington in the middle 1960s.', 'USS Brutus (AC-15) | USS "Brutus", formerly the steamer "Peter Jebsen", was a collier in the United States Navy. She was built in 1894 at South Shields-on-Tyne, England, by John Readhead & Sons and was acquired by the U.S. Navy early in 1898 from L. F. Chapman & Company. She was renamed "Brutus" and commissioned at the Mare Island Navy Yard on 27 May 1898, with Lieutenant Vincendon L. Cottman, commanding officer and Lieutenant Randolph H. Miner, executive officer.', 'Brutus Beefcake | Ed Leslie is an American semi-retired professional wrestler, best known for his work in the World Wrestling Federation (WWF) under the ring name Brutus "The Barber" Beefcake. He later worked for World Championship Wrestling (WCW) under a variety of names.', 'Brutus Hamilton | Brutus Kerr Hamilton (July 19, 1900 – December 28, 1970) was an American track and field athlete, coach and athletics administrator.', 'Big Brutus | Big Brutus is the nickname of the Bucyrus-Erie model 1850B electric shovel, which was the second largest of its type in operation in the 1960s and 1970s. Big Brutus is the centerpiece of a mining museum in West Mineral, Kansas where it was used in coal strip mining operations. The shovel was designed to dig from 20 to in relatively shallow coal seams.'], 'tweet_response': ' Jim Liberman, also known as "Jungle Jim", drove the pioneering funny car Brutus in the 1960s. #Brutus #FunnyCar #DragRacing'}) (input_keys=None)]




```python

```
