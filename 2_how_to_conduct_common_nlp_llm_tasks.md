## Natural language processing (NLP) LLM Tasks
This notebook goes a bit futher with diverse examples to demonstrate various tasks, emphasizing effective usage of prompt engineering through practical instances and using [CO-STAR prompt framework](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41) for 
eliciting the best response from the LLM

The tasks explored in this notebook, using sophiscated prompting techniques, show *how-to* code examples for common natural language understanfing capabilites of a generalized LLM, such as ChatGPT and Llama 2 series:

 * Text generation or completion
 * Text summarization
 * Text extraction
 * Text classification or sentiment analysis
 * Text categorization
 * Text transformation and translation
 * Simple and complex reasoning

<img src="./images/llm_prompt_req_resp.png" height="35%" width="%65">

**Note**: 
To run any of these relevant notebooks you will need an account on Anyscale Endpoints, Anthropic, or OpenAI, depending on what model you elect, along with the respective environment file. Use the template environment files to create respective `.env` file for either Anyscale Endpoints, Anthropic, or OpenAI.



```python
import warnings
import os

import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
```


```python
BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"
```


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```

    Using MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1; base=https://api.endpoints.anyscale.com/v1
    


```python
from openai import OpenAI

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base
)
```


```python
BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"
```


```python
def get_commpletion(clnt: object, model: str, system_content: str, user_content:str) -> str:
    chat_completion = clnt.chat.completions.create(
        model=model,
    messages=[{"role": "system", "content": system_content},
              {"role": "user", "content": user_content}],
    temperature = 0.8)

    response = chat_completion.choices[0].message.content
    return response
```

## Text generation or completion
In this simple task, we use an LLM to generate text by finishing an incomplete user content provided in the prompt. For example,
by providing an incomplete prompt such as "On a cold winter night, the stray dog ...". 

Let's try a few text generation or completion tasks by providing partial prompts in the user content. You will surprised at its fluency and coherency in the generated text.

For prompting, we use the C0-STAR framework.

<img src="./images/co-star-framework.png" height="35%" width="%65">


**(C): Context: Provide background and information on the task**

**(O): Objective: Define the task that you want the LLM to perform**

**(S): Style: Specify the writing style you want the LLM to use**

**(T): Set the attidue of the response**

**(A): Audience: Idenfiy who the response is for**

**(R): Provide the response format**



```python
system_content = """You are master of all knowledge, and a helpful sage.
                    You must complete any incomplete sentence by drawing from your vast
                    knowledge about history, literature, science, social science, philosophy, religion, economics, 
                    sports, etc. Do not make up any responses.
                  """

user_prompts =  ["On cold winter nights, the wolves in Siberia ...",
                 "On the day Franklin Benjamin realized his passion for printer, ...",
                 "During the final World Cup 1998 when France beat Brazil in Paris, ...",
                 "Issac Newton set under a tree when an apple fell..."
                ]
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for user_prompt in user_prompts:
    prompt = f"""
    # CONTEXT #
    I want to write a complete and cohesive paragpraph for 
    magazine: Things to know.

    #############
    
    # OBJECTIVE #
    Compete the text ```{user_prompt}``` between three backticks to the best 
    of your acquired knowledge.

    #############

    # STYLE #
    You will use simple, compound, and compound-complex sentences for all your responses, 
    and no more than one paragraph and no more than five sentences.

    Adhere to a litrary magazine writing style. Keep your sentences succinct and cohesive.

    #############

    # TONE #
    Maintain an editorial tone.

    #############

    # AUDIENCE #
    Our audience are generally curious first to second year college
    students.

    #############

    # RESPONSE #
    Finally, keep the response concise and succinct.
    """
    response = get_commpletion(client, MODEL, system_content, prompt)
    response = response.replace("```", "")
    print(f"\n{BOLD_BEGIN}Prompt:{BOLD_END} {user_prompt}")
    print(f"\n{BOLD_BEGIN}Answer:{BOLD_END} {response}")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mPrompt:[0m On cold winter nights, the wolves in Siberia ...
    
    [1mAnswer:[0m  On cold winter nights, the wolves in Siberia engage in a range of behaviors that allow them to survive and thrive in this harsh environment. As highly social animals, they often form packs for hunting and protection. These groups, usually consisting of an alpha pair and their offspring, work together to bring down prey such as elk, deer, and boar. In addition to hunting, wolves also spend a significant amount of time on communal activities like playing, grooming, and resting. This helps to strengthen social bonds and ensure the survival of the group.
    
    Interestingly, wolves in Siberia are well adapted to cold temperatures, with thick fur coats that provide insulation and a layer of fat that helps to conserve heat. They are also able to regulate their body temperature through a process called vasodilation, which allows them to constrict blood vessels in their extremities to reduce heat loss. This, combined with their highly efficient hunting techniques, makes them well suited to life in the Siberian wilderness.
    
    Despite their important role in the ecosystem, wolves in Siberia have faced significant threats from human activities, including habitat loss, hunting, and persecution. As a result, their populations have declined dramatically in recent decades, with some estimates suggesting that there are now as few as 30,000 wolves left in the wild. Conservation efforts are underway to protect these remarkable animals and ensure their survival for future generations.
    
    [1mPrompt:[0m On the day Franklin Benjamin realized his passion for printer, ...
    
    [1mAnswer:[0m  On the day Benjamin Franklin realized his passion for printing, he was just 12 years old. Working as an apprentice to his older brother James, he honed his skills in typography, composition, and presswork. This experience ignited a lifelong love for the art and science of printing, which eventually led him to establish the Pennsylvania Gazette and become one of the most influential printers and publishers of his time. In addition to his printing career, Franklin is also remembered for his groundbreaking scientific discoveries, his contributions to the creation of the United States, and his role as a leading statesman and diplomat. His insatiable curiosity and relentless pursuit of knowledge continue to inspire generations of learners and innovators.
    
    [1mPrompt:[0m During the final World Cup 1998 when France beat Brazil in Paris, ...
    
    [1mAnswer:[0m  During the final World Cup in 1998, France beat Brazil in Paris, marking a significant moment in sports history. This victory was especially sweet for the French, as it was their first time winning the World Cup. The match took place at the Stade de France, where over 80,000 spectators watched the thrilling game. The winning goal was scored by Zinedine Zidane in the 27th minute of the first half, resulting in an exciting 3-0 victory for France. This game not only solidified France's status as a soccer powerhouse but also brought the country together in a wave of national pride and celebration.
    
    [1mPrompt:[0m Issac Newton set under a tree when an apple fell...
    
    [1mAnswer:[0m  Issac Newton, one of the most influential scientists in history, was sitting under an apple tree when an apple fell, an event that sparked his interest in the laws of motion and gravity. This story, while perhaps apocryphal, highlights the serendipitous nature of many scientific discoveries. Newton, who was already fascinated by the mechanics of the universe, began to ponder why the apple fell straight down instead of going off to the side or upwards. This line of thinking led him to formulate the laws of motion and universal gravitation, which are still used today to describe the behavior of objects on Earth and in space.
    
    Newton's work set the stage for the scientific revolution, and his discoveries have had a profound impact on our understanding of the physical world. He developed the three laws of motion, which describe the relationship between a body and the forces acting upon it, and the law of universal gravitation, which states that every particle of matter in the universe attracts every other particle with a force that is directly proportional to the product of their masses and inversely proportional to the square of the distance between their centers. These laws not only helped to explain the motion of objects on Earth, but also allowed scientists to understand and predict the motion of celestial bodies, such as planets and stars.
    
    Newton's work also helped to establish the field of physics, and his methods and theories continue to be studied and built upon today. He was a key figure in the scientific revolution, and his discoveries helped to shape our understanding of the natural world. Newton's work has had a profound impact on a wide range of fields, from engineering and astronomy to mathematics and philosophy. He was a true polymath, and his contributions to science and knowledge continue to be felt to this day.
    

## Text summarization

A common task in natural langauge processing is text summiarization. A common use case
is summarizing large articles or documents, for a quick and easy-to-absorb summaries.

You can instruct LLM to generate the response in a preferable style, and comprehensibility. For example, use simple language aimed for a certain grade level, keep the orginal style of the article, use different sentence sytles (as we have done in few of examples in this notebook and previous one).

Let's try a few examples.


```python
system_content = """You are master of all knowledge about history, literature, science, philosophy, religion, 
                    economics, sports, etc. Respond to only answers
                    you know of. Do not make up answers""" 
                
user_prompts = [
    """ The emergence of large language models (LLMs) has marked a significant 
         breakthrough in natural language processing (NLP), leading to remarkable 
         advancements in text understanding and generation. 
         
         Nevertheless, alongside these strides, LLMs exhibit a critical tendency 
         to produce hallucinations, resulting in content that is inconsistent with 
         real-world facts or user inputs. This phenomenon poses substantial challenges 
         to their practical deployment and raises concerns over the reliability of LLMs 
         in real-world scenarios, which attracts increasing attention to detect and 
         mitigate these hallucinations. In this survey, we aim to provide a thorough and 
         in-depth  overview of recent advances in the field of LLM hallucinations. 
         
         We begin with an innovative taxonomy of LLM hallucinations, then delve into the 
         factors contributing to hallucinations. Subsequently, we present a comprehensive
         overview of hallucination detection methods and benchmarks. 
         Additionally, representative approaches designed to mitigate hallucinations 
         are introduced accordingly. 
         
         Finally, we analyze the challenges that highlight the current limitations and 
         formulate open questions, aiming to delineate pathways for future  research on 
         hallucinations in LLMs.""",
    """  Can a Large Language Model (LLM) solve simple abstract reasoning problems?
         We explore this broad question through a systematic analysis of GPT on the 
         Abstraction and Reasoning Corpus (ARC), a representative benchmark of abstract 
         reasoning ability from limited examples in which solutions require some 
         "core knowledge" of concepts such as objects, goal states, counting, and 
         basic geometry. GPT-4 solves only 13/50 of the most straightforward ARC 
         tasks when using textual encodings for their two-dimensional input-output grids. 
         Our failure analysis reveals that GPT-4's capacity to identify objects and 
         reason about them is significantly influenced by the sequential nature of 
         the text that represents an object within a text encoding of a task. 
         To test this hypothesis, we design a new benchmark, the 1D-ARC, which 
         consists of one-dimensional (array-like) tasks that are more conducive 
         to GPT-based reasoning, and where it indeed performs better than on 
         the (2D) ARC. To alleviate this issue, we propose an object-based 
         representation that is obtained through an external tool, resulting in 
         nearly doubling the performance on solved ARC tasks and near-perfect scores 
         on the easier 1D-ARC. Although the state-of-the-art GPT-4 is unable to 
         "reason" perfectly within non-language domains such as the 1D-ARC or a 
         simple ARC subset, our study reveals that the use of object-based representations 
         can significantly improve its reasoning ability. Visualizations, GPT logs, and 
         data are available at this https URL."""
]
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for user_prompt in user_prompts:
    prompt = f"""

    # CONTEXT #
    I want to write a summarize cohesively into at most two paragpraphs for 
    my magazine: Things to know quickly

    #############
    
    # OBJECTIVE #
    Summarize the text ```{user_prompt}``` between three backticks to the best 
    of your acquired knowledge.

     #############

    # STYLE #
    You will use simple, compound, and compound-complex sentences for all your responses, 
    and no more than one paragraph and no more than five sentences.

    Adhere to a litrary magazine writing style. Keep your sentences succinct and cohesive.

    # TONE #
    Maintain the same tone as the text supplied.

    #############

    # AUDIENCE #
    Our audience are generally curious first to second year college
    students.

    #############

     # RESPONSE #
    Finally, keep the response concise and succinct
    """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original content:{BOLD_END} {user_prompt}")
    print(f"\n{BOLD_BEGIN}Summary  content:{BOLD_END} {response}")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mOriginal content:[0m  The emergence of large language models (LLMs) has marked a significant 
             breakthrough in natural language processing (NLP), leading to remarkable 
             advancements in text understanding and generation. 
             
             Nevertheless, alongside these strides, LLMs exhibit a critical tendency 
             to produce hallucinations, resulting in content that is inconsistent with 
             real-world facts or user inputs. This phenomenon poses substantial challenges 
             to their practical deployment and raises concerns over the reliability of LLMs 
             in real-world scenarios, which attracts increasing attention to detect and 
             mitigate these hallucinations. In this survey, we aim to provide a thorough and 
             in-depth  overview of recent advances in the field of LLM hallucinations. 
             
             We begin with an innovative taxonomy of LLM hallucinations, then delve into the 
             factors contributing to hallucinations. Subsequently, we present a comprehensive
             overview of hallucination detection methods and benchmarks. 
             Additionally, representative approaches designed to mitigate hallucinations 
             are introduced accordingly. 
             
             Finally, we analyze the challenges that highlight the current limitations and 
             formulate open questions, aiming to delineate pathways for future  research on 
             hallucinations in LLMs.
    
    [1mSummary  content:[0m  The rapid development of Large Language Models (LLMs) has significantly advanced natural language processing, enabling remarkable abilities in understanding and generating human-like text. However, these models also have a tendency to produce hallucinations, where they create content inconsistent with real-world facts or user inputs. This issue raises concerns regarding their practical use and reliability in real-world scenarios, pushing researchers to focus on detecting and mitigating LLM hallucinations. In this survey, experts present a thorough overview of recent advancements in understanding and addressing LLM hallucinations.
    
    The survey introduces a new taxonomy of LLM hallucinations and examines the factors contributing to these errors. It also provides a comprehensive review of detection methods and benchmarks for hallucinations, along with showcasing representative approaches designed to minimize such errors. Recognizing the current challenges and limitations, the survey concludes by outlining open questions and potential directions for future research in managing hallucinations in LLMs. This overview serves as a valuable resource for college students interested in the fascinating and rapidly evolving field of AI and language models.
    
    [1mOriginal content:[0m   Can a Large Language Model (LLM) solve simple abstract reasoning problems?
             We explore this broad question through a systematic analysis of GPT on the 
             Abstraction and Reasoning Corpus (ARC), a representative benchmark of abstract 
             reasoning ability from limited examples in which solutions require some 
             "core knowledge" of concepts such as objects, goal states, counting, and 
             basic geometry. GPT-4 solves only 13/50 of the most straightforward ARC 
             tasks when using textual encodings for their two-dimensional input-output grids. 
             Our failure analysis reveals that GPT-4's capacity to identify objects and 
             reason about them is significantly influenced by the sequential nature of 
             the text that represents an object within a text encoding of a task. 
             To test this hypothesis, we design a new benchmark, the 1D-ARC, which 
             consists of one-dimensional (array-like) tasks that are more conducive 
             to GPT-based reasoning, and where it indeed performs better than on 
             the (2D) ARC. To alleviate this issue, we propose an object-based 
             representation that is obtained through an external tool, resulting in 
             nearly doubling the performance on solved ARC tasks and near-perfect scores 
             on the easier 1D-ARC. Although the state-of-the-art GPT-4 is unable to 
             "reason" perfectly within non-language domains such as the 1D-ARC or a 
             simple ARC subset, our study reveals that the use of object-based representations 
             can significantly improve its reasoning ability. Visualizations, GPT logs, and 
             data are available at this https URL.
    
    [1mSummary  content:[0m  A study examining the ability of a large language model, GPT-4, to solve simple abstract reasoning problems has revealed some interesting findings. The researchers tested GPT-4 on the Abstraction and Reasoning Corpus (ARC), a benchmark that requires core knowledge of concepts such as objects, goal states, counting, and basic geometry. However, GPT-4 was only able to solve 13 out of 50 of the most straightforward ARC tasks when using textual encodings for their two-dimensional input-output grids.
    
    The researchers found that GPT-4's capacity to identify objects and reason about them is influenced by the sequential nature of the text that represents an object within a text encoding of a task. To test this hypothesis, they designed a new benchmark, the 1D-ARC, which consists of one-dimensional (array-like) tasks that are more conducive to GPT-based reasoning. The study found that the use of object-based representations, obtained through an external tool, significantly improved GPT-4's reasoning ability, nearly doubling its performance on solved ARC tasks and achieving near-perfect scores on the easier 1D-ARC. While GPT-4 is not yet able to "reason" perfectly within non-language domains, this study suggests that object-based representations can enhance its reasoning ability.
    

## Text or information extraction

Another natural langauge capability, similar to summarization or text completion, is extracting key idea or infromation from an article, blog, or a paragraph. For example,
given a set of text, you can ask LLM to extract key ideas or topics or subjects. Or even
better enumerate key takeways for you, saving time if you are in a hurry.

Let's see *how-to* do it by first looking at a simple example, and then progressing into a more complex one, all along keepin 
the [CO-STAR prompting framework](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41) in mind.

### Task 1: 
 * summarize the product review
 * extract any information about shipping and packaging for shipping department
 * classify the sentiment of the review: positive or negative.
 * use precise, specific prompt to acheive the task


```python
system_content = """You are master of all knowledge about history, literature, science, social science, 
philosophy, religion, economics, sports, etc.
"""

product_review = """I got this Australian Bush Baby with soft fur for my niece's birthday,
and she absolutely loves it, carrying it around everywhere. The fur is exceptionally soft,
and its adorable face gives off a friendly vibe. While I find it a bit smaller than 
anticipated for the price, my niece's joy makes it worthwhile. What pleasantly surprised 
me was the early arrival; it came a day earlier than expected. 
I appreciated the prompt delivery, and the packaging was secure, ensuring the 
Bush Baby with soft fur arrived in perfect condition. This allowed me to play with 
it myself before presenting it to my niece.
"""     
```


```python
prompt = f"""
# CONTEXT #
In our customer service department, we value customer feedback and want to
analyze their reviews by summarizing their review assessing, labeling or categorizing
and its idenfitying its sentiment. 

#############

# OBJECTIVE #
Your task is to generate a short summary of a product 
review from an Australian e-commerce site to offer feedback to the 
shipping deparmtment. Follow the steps below:

First, generate a short summary of review below, delimited by triple 
backticks, in two sentences: a simple and compound sentence. 

Second, focus on any aspects of packaging or shipping of the product, and label it as 
"Shipping Department:".  

Third, indicate if the review is positive or negative, and label it as "Sentiment:".
Do not provide any preamble, only a simple two words: Positive or negative
Review: ```{product_review}``` 

#############

# STYLE #
You will use simple, compound, and compound-complex sentences for all your responses, 
and no more than one paragraph and no more than five sentences.

#############

# TONE #
Maintain a professional tone for internal communications

#############

 # AUDIENCE #
Our audience are internal departments in customer success to ensure we can
improve our customer service

#############

# RESPONSE #
Finally, keep the response concise and succinct
"""
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
response = get_commpletion(client, MODEL, system_content, prompt)
print(f"""\n{BOLD_BEGIN}Summary:{BOLD_END} {response.replace("```", "")}""")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mSummary:[0m  Summary: A customer purchased an Australian Bush Baby toy with soft fur for their niece's birthday, who carries it everywhere and is delighted with it. The product arrived earlier than expected, and the packaging ensured secure delivery.
    
    Shipping Department: The shipping department delivered the product a day earlier than expected, and the packaging was secure, preserving the product's quality.
    
    Sentiment: Positive. The customer is satisfied with the shipping and delivery of the product, and their niece's joy reinforces the positive sentiment towards the purchase.
    

### Task 2
 * Given a passage from an article, extract the main theme of the passage and label it as the `Subjects`, if more than one, separated by comma.
 * Identify three key takeways and enumerate them in simple sentences


```python
system_content = "You are master of all knowledge about history, literature, science, social science, philosophy, religion, economics, sports, etc."
            
user_prompts = ["""Isaac Newton sat under a tree when an apple fell, an event that, 
                according to popular legend, led to his contemplation of the forces
                of gravity. Although this story is often regarded as apocryphal or at 
                least exaggerated, it serves as a powerful symbol of Newton's insight 
                into the universal law that governs celestial and earthly bodies alike. 
                His formulation of the law of universal gravitation was revolutionary, 
                as it provided a mathematical explanation for both the motion of planets 
                and the phenomena observed on Earth. Newton's work in physics, captured 
                in his seminal work Philosophiæ Naturalis Principia Mathematica, laid the 
                groundwork for classical mechanics. His influence extended beyond his own 
                time, shaping the course of scientific inquiry for centuries to come.
                """
               ]

```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for text in user_prompts:
    prompt = f""" Given ```{text}``` delimited with triple backticks, identify a single key idea being discussed, 
    and label its 'Subject'. Next, enumerate at most three takeways. 
    Use short, simple sentences. """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original content:{BOLD_END} {text}")
    print(f"\n {BOLD_BEGIN}Extracted answers: {BOLD_END} {response}")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mOriginal content:[0m Isaac Newton sat under a tree when an apple fell, an event that, 
                    according to popular legend, led to his contemplation of the forces
                    of gravity. Although this story is often regarded as apocryphal or at 
                    least exaggerated, it serves as a powerful symbol of Newton's insight 
                    into the universal law that governs celestial and earthly bodies alike. 
                    His formulation of the law of universal gravitation was revolutionary, 
                    as it provided a mathematical explanation for both the motion of planets 
                    and the phenomena observed on Earth. Newton's work in physics, captured 
                    in his seminal work Philosophiæ Naturalis Principia Mathematica, laid the 
                    groundwork for classical mechanics. His influence extended beyond his own 
                    time, shaping the course of scientific inquiry for centuries to come.
                    
    
     [1mExtracted answers: [0m  Subject: Isaac Newton's formulation of the law of universal gravitation
    
    Takeaways:
    1. Newton's insight about gravity was inspired by an apple falling from a tree.
    2. His law of universal gravitation provides a mathematical explanation for both celestial and earthly bodies' motion.
    3. Newton's work, outlined in Philosophiæ Naturalis Principia Mathematica, established classical mechanics and significantly influenced future scientific inquiry.
    

Let's try another example to extract more than one subject or topic being
discussed in the text, and enumerate three takeways.

(Incidentally, I'm reading biography of Benjamin Franklin by Issac Stevenson, and all this seems to align with his career path and passion.)


```python
user_stories = [""""
'The Printer'
    He that has a Trade has an Office of Profit and Honour’ Poor Richard’s Almanack
Benjamin Franklin had an affinity with print and books throughout his life. 
Apprenticed as a child to his brother James, a printer, he mastered all aspects of
the trade, from typesetting to engraving, learning the latest techniques during his
first visit to London.  An avid reader, Franklin saved money to buy books by 
temporarily turning vegetarian and, once settled in Philadelphia, founded the 
Library Company, the first subscription library in the colonies.  As an elder
statesman, he even bought type and kept a press during his stay in France. 
After working as a printer’s journeyman, he set up his own Philadelphian printing 
office in 1728.  His success with the Pennslyannia Gazette and Poor Richard’s
Almanack helped to provide Franklin with the financial means to retire from
business, retaining a stake in his print shop and founding others throughout the 
colonies.  Print also gave him a public voice: Franklin preferred the printed word, 
rather than public rhetoric, influencing political and public opinion as a brilliant
journalist and pamphleteer.

'Silence Dogood and the New­England Courant'
    When James Franklin lost the contract to print the Boston Gazette, he determined
to begin his own newspaper, launching the New­England Courant in 1721.
Benjamin, who had been indentured secretly to James, helped to print the weekly 
paper.  One night he slipped a composition under the door, beginning the series
of ‘Silence Dogood’ letters, the purported epistles of a vocal widower, with strong 
opinions on drunks, clergymen, foolish fashions and Boston nightlife. Owing no
little debt to the satire of the London Spectator, the letters represented a 
remarkable literary achievement for the 16­year old.  The British Library’s copy has 
been uniquely annotated in what appears to be Franklin’s hand. The first 
‘Dogood’ letter appears on the bottom right.

‘The Main Design of the Weekly Paper will be to Entertain the Town’
    Benjamin’s brother, James, began the New­England Courant in the face of
opposition from the Boston Establishment.  He soon irritated them with his squibs
and satires on the great and the good, attacking the influential clergyman Cotton
Mather’s pet project of small pox inoculation and the authorities’ weak response 
to piracy. Twice arrested, James temporally left the paper in Benjamin’s hands, and 
then continued to publish it under Benjamin’s name to escape a ban on
publication.  This issue is the first printed item to carry the imprint ‘B. Franklin’ (on
the rear).  Franklin announces his intention to ‘Entertain the Town’ on this page.
"""]
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for story in user_stories:
    prompt = f""" Extract five subjects that are being discussed in the 
                  following text, which is delimited by triple backticks.
                  Format your response as a list of subjects 
                  "Subjects:" separate each subject by a comma.
                  Make each subject at most two words long, not longer. 
                  Next, enumerate  as a list three takeways, and label them as "Takeways:" 
                  Use short, simple sentences for your takeways.
                  Text sample: '''{story}'''
                  """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original Story:{BOLD_END} {story}")
    print(f"\n {BOLD_BEGIN}Extracted entities:{BOLD_END} {response}")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mOriginal Story:[0m "
    'The Printer'
        He that has a Trade has an Office of Profit and Honour’ Poor Richard’s Almanack
    Benjamin Franklin had an affinity with print and books throughout his life. 
    Apprenticed as a child to his brother James, a printer, he mastered all aspects of
    the trade, from typesetting to engraving, learning the latest techniques during his
    first visit to London.  An avid reader, Franklin saved money to buy books by 
    temporarily turning vegetarian and, once settled in Philadelphia, founded the 
    Library Company, the first subscription library in the colonies.  As an elder
    statesman, he even bought type and kept a press during his stay in France. 
    After working as a printer’s journeyman, he set up his own Philadelphian printing 
    office in 1728.  His success with the Pennslyannia Gazette and Poor Richard’s
    Almanack helped to provide Franklin with the financial means to retire from
    business, retaining a stake in his print shop and founding others throughout the 
    colonies.  Print also gave him a public voice: Franklin preferred the printed word, 
    rather than public rhetoric, influencing political and public opinion as a brilliant
    journalist and pamphleteer.
    
    'Silence Dogood and the New­England Courant'
        When James Franklin lost the contract to print the Boston Gazette, he determined
    to begin his own newspaper, launching the New­England Courant in 1721.
    Benjamin, who had been indentured secretly to James, helped to print the weekly 
    paper.  One night he slipped a composition under the door, beginning the series
    of ‘Silence Dogood’ letters, the purported epistles of a vocal widower, with strong 
    opinions on drunks, clergymen, foolish fashions and Boston nightlife. Owing no
    little debt to the satire of the London Spectator, the letters represented a 
    remarkable literary achievement for the 16­year old.  The British Library’s copy has 
    been uniquely annotated in what appears to be Franklin’s hand. The first 
    ‘Dogood’ letter appears on the bottom right.
    
    ‘The Main Design of the Weekly Paper will be to Entertain the Town’
        Benjamin’s brother, James, began the New­England Courant in the face of
    opposition from the Boston Establishment.  He soon irritated them with his squibs
    and satires on the great and the good, attacking the influential clergyman Cotton
    Mather’s pet project of small pox inoculation and the authorities’ weak response 
    to piracy. Twice arrested, James temporally left the paper in Benjamin’s hands, and 
    then continued to publish it under Benjamin’s name to escape a ban on
    publication.  This issue is the first printed item to carry the imprint ‘B. Franklin’ (on
    the rear).  Franklin announces his intention to ‘Entertain the Town’ on this page.
    
    
     [1mExtracted entities:[0m  Subjects:
    * Printing trade
    * Book collection
    * Journalism
    * Satire
    * Censorship
    
    Takeaways:
    * Benjamin Franklin had a lifelong affinity with printing and books.
    * Franklin's "Silence Dogood" letters in the New-England Courant were a remarkable literary achievement for a 16-year-old.
    * Franklin's brother James faced opposition and censorship for his satirical content in the New-England Courant.
    

## Text classification or sentiment analysis

Unlike classical or traditional machine learning, where you'll have to do supervised learning to collect data, label it, and train for hours, depending on how much data,classifying text using LLM is simple.

In short, you'll have to build an ML model to understand text and classify its sentiments as positive, negative or neutral. 

This onus task is easily done with LLM via clever prompting. 

Let's see what I mean in this *how-to* idenfity sentiments in text. But first let's 
generatre some sentiments as our ground truth, and supply them to LLM to observe if
LLM identifies them correctly. This bit is not needed, for I'm just curious.

*Positive*: "This movie is a true cinematic gem, blending an engaging plot with superb performances and stunning visuals. A masterpiece that leaves a lasting impression."

*Negative*: "Regrettably, the film failed to live up to expectations, with a convoluted storyline, lackluster acting, and uninspiring cinematography. A disappointment overall."

*Neutral*: "The movie had its moments, offering a decent storyline and average performances. While not groundbreaking, it provided an enjoyable viewing experience."

*Positive*: "This city is a vibrant tapestry of culture, with friendly locals, historic landmarks, and a lively atmosphere. An ideal destination for cultural exploration."

*Negative*: "The city's charm is overshadowed by traffic congestion, high pollution levels, and a lack of cleanliness. Not recommended for a peaceful retreat."

*Neutral*: "The city offers a mix of experiences, from bustling markets to serene parks. An interesting but not extraordinary destination for exploration."

*Positive*: "This song is a musical masterpiece, enchanting listeners with its soulful lyrics, mesmerizing melody, and exceptional vocals. A timeless classic."

*Negative*: "The song fails to impress, featuring uninspiring lyrics, a forgettable melody, and lackluster vocals. It lacks the creativity to leave a lasting impact."

*Neutral*: "The song is decent, with a catchy tune and average lyrics. While enjoyable, it doesn't stand out in the vast landscape of music."

*Positive*: "A delightful cinematic experience that seamlessly weaves together a compelling narrative, strong character development, and breathtaking visuals."

*Negative*: "This film, unfortunately, falls short with a disjointed plot, subpar performances, and a lack of coherence. A disappointing viewing experience."

*Neutral*: "While not groundbreaking, the movie offers a decent storyline and competent performances, providing an overall satisfactory viewing experience."

*Positive*: "This city is a haven for culture enthusiasts, boasting historical landmarks, a rich culinary scene, and a welcoming community. A must-visit destination."

*Negative*: "The city's appeal is tarnished by overcrowded streets, noise pollution, and a lack of urban planning. Not recommended for a tranquil getaway."

*Neutral*: "The city offers a diverse range of experiences, from bustling markets to serene parks. An intriguing destination for those seeking a mix of urban and natural landscapes."


```python
system_content = """You are a prominent critic of landscapes, architecture, cities, movies, songs, 
                    entertainment, and a cultural ombudsman. """

user_sentiments = [ "This movie is a true cinematic gem, blending an engaging plot with superb performances and stunning visuals. A masterpiece that leaves a lasting impression.",
                    "Regrettably, the film failed to live up to expectations, with a convoluted storyline, lackluster acting, and uninspiring cinematography. A disappointment overall.",
                    "The movie had its moments, offering a decent storyline and average performances. While not groundbreaking, it provided an enjoyable viewing experience.",
                    "This city is a vibrant tapestry of culture, with friendly locals, historic landmarks, and a lively atmosphere. An ideal destination for cultural exploration.",
                    "The city's charm is overshadowed by traffic congestion, high pollution levels, and a lack of cleanliness. Not recommended for a peaceful retreat.",
                    "The city offers a mix of experiences, from bustling markets to serene parks. An interesting but not extraordinary destination for exploration.",
                    "This song is a musical masterpiece, enchanting listeners with its soulful lyrics, mesmerizing melody, and exceptional vocals. A timeless classic.",
                    "The song fails to impress, featuring uninspiring lyrics, a forgettable melody, and lackluster vocals. It lacks the creativity to leave a lasting impact.",
                    "The song is decent, with a catchy tune and average lyrics. While enjoyable, it doesn't stand out in the vast landscape of music.",
                    "A delightful cinematic experience that seamlessly weaves together a compelling narrative, strong character development, and breathtaking visuals.",
                    "This film, unfortunately, falls short with a disjointed plot, subpar performances, and a lack of coherence. A disappointing viewing experience.",
                    "While not groundbreaking, the movie offers a decent storyline and competent performances, providing an overall satisfactory viewing experience.",
                    "This city is a haven for culture enthusiasts, boasting historical landmarks, a rich culinary scene, and a welcoming community. A must-visit destination.",
                    "The city's appeal is tarnished by overcrowded streets, noise pollution, and a lack of urban planning. Not recommended for a tranquil getaway.",
                    "The city offers a diverse range of experiences, from bustling markets to serene parks. An intriguing destination for those seeking a mix of urban and natural landscapes.",
                    "xxxyyyzzz was curious and dubious"
]
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for user_sentiment in user_sentiments:
    prompt = f"""Classify the sentiment in the ```{user_sentiment}`` which is delimited 
        with triple backticks? Classify the given text into single label as 
        neutral, negative or positive. Do not expand on your response. 
        Use single words: positive, negative, neutral
        If you cannot classify do not guess, do not ask for more info,
        just classify as NA.
    """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Sentiment:{BOLD_END} {user_sentiment}")
    print(f"\n{BOLD_BEGIN}Label    :{BOLD_END} {response}")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mSentiment:[0m This movie is a true cinematic gem, blending an engaging plot with superb performances and stunning visuals. A masterpiece that leaves a lasting impression.
    
    [1mLabel    :[0m  Positive
    
    [1mSentiment:[0m Regrettably, the film failed to live up to expectations, with a convoluted storyline, lackluster acting, and uninspiring cinematography. A disappointment overall.
    
    [1mLabel    :[0m  Negative
    
    [1mSentiment:[0m The movie had its moments, offering a decent storyline and average performances. While not groundbreaking, it provided an enjoyable viewing experience.
    
    [1mLabel    :[0m  Neutral
    
    [1mSentiment:[0m This city is a vibrant tapestry of culture, with friendly locals, historic landmarks, and a lively atmosphere. An ideal destination for cultural exploration.
    
    [1mLabel    :[0m  Positive
    
    [1mSentiment:[0m The city's charm is overshadowed by traffic congestion, high pollution levels, and a lack of cleanliness. Not recommended for a peaceful retreat.
    
    [1mLabel    :[0m  Negative
    
    [1mSentiment:[0m The city offers a mix of experiences, from bustling markets to serene parks. An interesting but not extraordinary destination for exploration.
    
    [1mLabel    :[0m  Neural. The sentiment in the given text is neutral, as it describes the city's features without expressing a positive or negative opinion.
    
    [1mSentiment:[0m This song is a musical masterpiece, enchanting listeners with its soulful lyrics, mesmerizing melody, and exceptional vocals. A timeless classic.
    
    [1mLabel    :[0m  Positive
    
    [1mSentiment:[0m The song fails to impress, featuring uninspiring lyrics, a forgettable melody, and lackluster vocals. It lacks the creativity to leave a lasting impact.
    
    [1mLabel    :[0m  Negative
    
    [1mSentiment:[0m The song is decent, with a catchy tune and average lyrics. While enjoyable, it doesn't stand out in the vast landscape of music.
    
    [1mLabel    :[0m  Neutral
    
    [1mSentiment:[0m A delightful cinematic experience that seamlessly weaves together a compelling narrative, strong character development, and breathtaking visuals.
    
    [1mLabel    :[0m  Positive
    
    [1mSentiment:[0m This film, unfortunately, falls short with a disjointed plot, subpar performances, and a lack of coherence. A disappointing viewing experience.
    
    [1mLabel    :[0m  Negative
    
    [1mSentiment:[0m While not groundbreaking, the movie offers a decent storyline and competent performances, providing an overall satisfactory viewing experience.
    
    [1mLabel    :[0m  Neutral
    
    [1mSentiment:[0m This city is a haven for culture enthusiasts, boasting historical landmarks, a rich culinary scene, and a welcoming community. A must-visit destination.
    
    [1mLabel    :[0m  Positive
    
    [1mSentiment:[0m The city's appeal is tarnished by overcrowded streets, noise pollution, and a lack of urban planning. Not recommended for a tranquil getaway.
    
    [1mLabel    :[0m  Negative
    
    [1mSentiment:[0m The city offers a diverse range of experiences, from bustling markets to serene parks. An intriguing destination for those seeking a mix of urban and natural landscapes.
    
    [1mLabel    :[0m  Positive
    
    [1mSentiment:[0m xxxyyyzzz was curious and dubious
    
    [1mLabel    :[0m  Neutral
    

## Text categorization
Like sentiment analysis, given a query, an LLM can identify from its context how to classify and route customer queries to respective departments. Also, note that LLM can detect foul language and respond politely. Text categorization can be employed to automate customer on-line queries.

Let's look at how we can achieve that with smart and deliberate prompting.

<img src="./images/category_resp.png" height="35%" width="%65">




```python
system_content = """You are a smart and helful Assistant who can route customer queries to 
                    respective customer service departments.
                    """

customer_queries = ["""My modem has stop working. I tried to restart but the orange light keep flashing. It never turns green.""",
                    """I just moved into town, and I need Internet service""",
                    """Why does my bill include an extra $20 a month for cable TV when I don't use a television?""",
                    """I need to change my user name and password since someone is using my credentials. I cannot access my account.""",
                    """What days this week are we having a general upgrades to the cable models?""",
                    """What day is the best day to call customer service so that I can avoid talking to a bot!""",
                    """Your company is full of incompetent morons and fools!""",
                    """I hate your worthless services. Cancel my stupid account or else I'll sue you!"""
                   ]
                    
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for query in customer_queries:
    prompt = f""" 
    We are an Internet Service provider in a big metropolitan city. We want to 
    improve our customer service support by building a routing system so
    that customer inquiries are routed responsively, respectfully, and actively to
    appropriate company departments. Your task is to classify each customer's {query} 
    into one of the the following five categories:
    1. Technical support
    2. Billing 
    3. Account Management
    4. New Customer  
    5. General inquiry
    
    Do not expand or explain your response. Do not include backticks or quotes 
    in your response. Do not choose more than one category in your response from these categories: 
    Technical support, Billing, Account Management, New Customer, General inquiry
    Do not include the {query} in your response.
    If you can't classify the {query}, default to "General inquiry."
    If customer {query} uses a foul language, then respond with 
    "No need for foul language. Please be respectful."
    """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Query:{BOLD_END} {query}")
    print(f"\n{BOLD_BEGIN}Route to:{BOLD_END} {response}\n")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mQuery:[0m My modem has stop working. I tried to restart but the orange light keep flashing. It never turns green.
    
    [1mRoute to:[0m  Technical support
    
    
    [1mQuery:[0m I just moved into town, and I need Internet service
    
    [1mRoute to:[0m  New Customer
    
    
    [1mQuery:[0m Why does my bill include an extra $20 a month for cable TV when I don't use a television?
    
    [1mRoute to:[0m  Billing
    
    
    [1mQuery:[0m I need to change my user name and password since someone is using my credentials. I cannot access my account.
    
    [1mRoute to:[0m  Account Management
    
    
    [1mQuery:[0m What days this week are we having a general upgrades to the cable models?
    
    [1mRoute to:[0m  General inquiry
    
    
    [1mQuery:[0m What day is the best day to call customer service so that I can avoid talking to a bot!
    
    [1mRoute to:[0m  General inquiry
    
    
    [1mQuery:[0m Your company is full of incompetent morons and fools!
    
    [1mRoute to:[0m  General inquiry.
    
    
    [1mQuery:[0m I hate your worthless services. Cancel my stupid account or else I'll sue you!
    
    [1mRoute to:[0m  Based on your comment, I would classify it as "General inquiry" and also request that you "No need for foul language. Please be respectful." as our company values respectful communication.
    
    

## Text transation and transformation

Language translation by far is the most common use case for natural language processing. 
We have seen its early uses in Google translation, but with the emergence of multi-lingual LLMs, this task is simply achieved by exact prompting. 

In this section, we'll explore tasks in how to use LLMs for text translations, langugage identication, text transformation, spelling and grammar checking, tone adjustment, and format conversion.

### Task 1:
 * Given an English text, translate into French, Spanish, and German.
 * Given a foreign language text, idenfify the language, and translate to English.


```python
system_content= """You are a world reknowned supreme lingiust and a universal translator. You are a polglot, and fluently speak many global languages"""

english_texts = [""" Welcome to New York for the United Nations General Council Meeting. Today
is a special day for us to celeberate all our achievments since this global institute's formation.
But more importantly, we want to address how we can mitigate global conflict with conversation
and promote deterence, detente, and discussion."""
]
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for english_text in english_texts:
    prompt = f""""Given an English text in triple ticks '''{english_text}'''. Translate into
three languases: Spanish, French, German, and Mandarin. 
Label each translation with the langauge Name: followed by translation on a seperate line."""
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}English Text:{BOLD_END} {english_text}")
    print(f"\n{BOLD_BEGIN}Translation: {BOLD_END}{response}\n")
                
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mEnglish Text:[0m  Welcome to New York for the United Nations General Council Meeting. Today
    is a special day for us to celeberate all our achievments since this global institute's formation.
    But more importantly, we want to address how we can mitigate global conflict with conversation
    and promote deterence, detente, and discussion.
    
    [1mTranslation: [0m Spanish:
    Nombre del idioma: Español
    '''Bienvenido a Nueva York para la reunión del Consejo General de las Naciones Unidas. Hoy
    es un día especial para nosotros para celebrar todos nuestros logros desde la formación de
    esta institución global.
    Más importante aún, queremos abordar cómo podemos mitigar los conflictos globales a través
    de la conversación y promover la disuasión, la distensión y la discusión.'''
    
    French:
    Nom de la langue: Français
    '''Bienvenue à New York pour la réunion du Conseil général des Nations Unies. Aujourd'hui
    est un jour spécial pour nous de célébrer toutes nos réalisations depuis la formation de
    cette institution globale.
    Mais ce qui est encore plus important, nous voulons aborder la manière dont nous pouvons
    atténuer les conflits mondiaux grâce à la conversation et promouvoir la dissuasion, la détente
    et la discussion.'''
    
    German:
    Name der Sprache: Deutsch
    '''Willkommen in New York für die Generalversammlung der Vereinten Nationen. Heute
    ist ein besonderer Tag für uns, um alle unsere Errungenschaften seit der Gründung
    dieser globalen Einrichtung zu feiern.
    Aber noch wichtiger ist, wir möchten behandeln, wie wir globale Konflikte durch Unterhaltung
    mit Abschreckung, Entspannung und Diskussion mindern können.'''
    
    Mandarin:
    名称：中文
    '''欢迎来到新 York 参加联合国大会。今天
    是我们庆祝这个全球机构成立以来所 achievements 的特别日子。
    但更重要的是，我们想讨论如何通过对话来减轻全球冲突，并推广吓阻、缓和和讨论。'''
    
    

Given a foreing language, identify the language and translate into English.

This is the reverse of the above.


```python
languages_texts = ["""Bienvenidos a Nueva York para la Reunión del Consejo General de las Naciones Unidas. Hoy
es un día especial para celebrar todos nuestros logros desde la formación de este instituto global.
Pero más importante aún, queremos abordar cómo podemos mitigar el conflicto global con conversaciones
y promover la disuasión, la distensión y el diálogo.""",
            """Willkommen in New York zur Sitzung des Allgemeinen Rates der Vereinten Nationen. Heute
ist ein besonderer Tag für uns, um all unsere Errungenschaften seit der Gründung dieses globalen Instituts zu feiern.
Aber wichtiger ist, dass wir ansprechen möchten, wie wir globale Konflikte durch Gespräche mildern können
und Abschreckung, Entspannung und Diskussion fördern.""",
                  """Bienvenue à New York pour la réunion du Conseil Général des Nations Unies. Aujourd'hui,
c'est un jour spécial pour nous pour célébrer toutes nos réalisations depuis la formation de cette institution mondiale.
Mais plus important encore, nous voulons aborder comment nous pouvons atténuer les conflits mondiaux grâce à la conversation
et promouvoir la dissuasion, la détente et la discussion.""",
                  """欢迎来到纽约参加联合国大会议。今天对我们来说是一个特别的日子，我们将庆祝自该全球机构成立以来取得的所有成就。但更重要的是，我们想要讨论如何通过对话来缓解全球冲突，并促进遏制、缓和和讨论。
"""]

```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for language_text in languages_texts:
    prompt = f""""Given a language text in triple ticks '''{language_text}'''. Idenfity
    the language with the langauge Name: followed by an English translation on a seperate line, labeled as English translation:"""
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN} Language Text: {BOLD_END} {language_text}")
    print(f"\n{BOLD_BEGIN}Translation: {BOLD_END} {response}\n")
                
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1m Language Text: [0m Bienvenidos a Nueva York para la Reunión del Consejo General de las Naciones Unidas. Hoy
    es un día especial para celebrar todos nuestros logros desde la formación de este instituto global.
    Pero más importante aún, queremos abordar cómo podemos mitigar el conflicto global con conversaciones
    y promover la disuasión, la distensión y el diálogo.
    
    [1mTranslation: [0m  Language: Spanish
    English Translation:
    'Welcome to New York for the General Council Meeting of the United Nations. Today
    is a special day to celebrate all our achievements since the formation of this global institute.
    But even more importantly, we want to address how we can mitigate global conflict through conversation
    and promote deterrence, relaxation and dialogue.'
    
    
    [1m Language Text: [0m Willkommen in New York zur Sitzung des Allgemeinen Rates der Vereinten Nationen. Heute
    ist ein besonderer Tag für uns, um all unsere Errungenschaften seit der Gründung dieses globalen Instituts zu feiern.
    Aber wichtiger ist, dass wir ansprechen möchten, wie wir globale Konflikte durch Gespräche mildern können
    und Abschreckung, Entspannung und Diskussion fördern.
    
    [1mTranslation: [0m  Language: German
    
    English Translation:
    'Welcome to New York for the General Assembly session of the United Nations. Today is a special day for us to celebrate all our achievements since the establishment of this global institution. But more importantly, we would like to address how we can mitigate global conflicts through dialogue, deterrence, relaxation, and discussion.'
    
    
    [1m Language Text: [0m Bienvenue à New York pour la réunion du Conseil Général des Nations Unies. Aujourd'hui,
    c'est un jour spécial pour nous pour célébrer toutes nos réalisations depuis la formation de cette institution mondiale.
    Mais plus important encore, nous voulons aborder comment nous pouvons atténuer les conflits mondiaux grâce à la conversation
    et promouvoir la dissuasion, la détente et la discussion.
    
    [1mTranslation: [0m  Language: French
    
    English Translation:
    Welcome to New York for the General Council meeting of the United Nations. Today,
    it's a special day for us to celebrate all our achievements since the formation of this world institution.
    But more importantly, we want to address how we can mitigate global conflicts through conversation
    and promote deterrence, relaxation and discussion.
    
    
    [1m Language Text: [0m 欢迎来到纽约参加联合国大会议。今天对我们来说是一个特别的日子，我们将庆祝自该全球机构成立以来取得的所有成就。但更重要的是，我们想要讨论如何通过对话来缓解全球冲突，并促进遏制、缓和和讨论。
    
    
    [1mTranslation: [0m  Language: Chinese
    English Translation:
    'Welcome to New York to attend the United Nations General Assembly. Today is a special day for us, as we will celebrate all the achievements of this global organization since its establishment. But more importantly, we want to discuss how to alleviate global conflicts through dialogue, and promote containment, mitigation and discussion.'
    
    

### Task 2

 * Given an English text, proof read it and correct any grammatical and usage errors.
 * Given a Pirate text, correct its tone to standard English.



```python
system_content = """You are a fastidious grammarian. You can proofread any English text and convert to 
its grammtical correct and usage form."""

bad_english_texts = ["""I don't know nothing about them big words and grammar rules. Me and my friend, we was talking, and he don't agree with me. We ain't never gonna figure it out, I reckon. His dog don't listen good, always running around and don't come when you call.""",
                     """Yesterday, we was at the park, and them kids was playing. She don't like the way how they acted, but I don't got no problem with it. We seen a movie last night, and it was good, but my sister, she don't seen it yet. Them books on the shelf, they ain't interesting to me."""
                    ]
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for bad_english_text in bad_english_texts:
    prompt = f""""Proofread and correct the text provided in triple ticks '''{bad_english_text}'''.
    Use standard usage and remedy any incorect grammar usage.
    """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original Text:{BOLD_END} {bad_english_text}")
    print(f"\n{BOLD_BEGIN}Corrected  Text:{BOLD_END} {response}\n")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mOriginal Text:[0m I don't know nothing about them big words and grammar rules. Me and my friend, we was talking, and he don't agree with me. We ain't never gonna figure it out, I reckon. His dog don't listen good, always running around and don't come when you call.
    
    [1mCorrected  Text:[0m  "'I don't know anything about those big words and grammar rules. My friend and I were speaking, and he did not agree with me. We are not likely to figure it out, I suppose. His dog does not listen well, always running around and not coming when called.'"
    
    
    [1mOriginal Text:[0m Yesterday, we was at the park, and them kids was playing. She don't like the way how they acted, but I don't got no problem with it. We seen a movie last night, and it was good, but my sister, she don't seen it yet. Them books on the shelf, they ain't interesting to me.
    
    [1mCorrected  Text:[0m  Surely, I can help you with that! Here's the corrected version of the text:
    
    ''Yesterday, we were at the park, and the children were playing. She does not like the way they acted, but I do not have a problem with it. We saw a movie last night, and it was good, but my sister has not seen it yet. Those books on the shelf are not interesting to me.''
    
    I have corrected the incorrect use of verbs and pronouns to conform to standard English usage. Additionally, I have added punctuation to improve the clarity and readability of the text.
    
    


```python
pirate_texts = ["""Arrr matey! I be knowin' nuthin' 'bout them fancy words and grammatical rules. Me and me heartie, we be chattin', and he don't be agreein' with me. We ain't never gonna figure it out, I reckon. His scallywag of a dog don't be listenin' well, always runnin' around and not comin' when ye call."""
                       ]
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for pirate_text in pirate_texts:
    prompt = f""""Convert the Pirate text provided in triple ticks '''{pirate_text}'''.
    Use standard usage and remedy any incorect grammar usage, dropping all Pirate greetings.
    """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original Text:{BOLD_END} {pirate_text}")
    print(f"\n{BOLD_BEGIN}Corrected  Text:{BOLD_END} {response}\n")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mOriginal Text:[0m Arrr matey! I be knowin' nuthin' 'bout them fancy words and grammatical rules. Me and me heartie, we be chattin', and he don't be agreein' with me. We ain't never gonna figure it out, I reckon. His scallywag of a dog don't be listenin' well, always runnin' around and not comin' when ye call.
    
    [1mCorrected  Text:[0m  I know nothing about those fancy words and grammatical rules. My friend and I were chatting, and he didn't agree with me. We will never figure it out, I suppose. His dog, a scoundrel, never listens well and is always running around and never comes when called.
    
    

### Task 3
* Given some text in a particular format, convert it into JSON format.
* For example, we LLM to producce names of three top shoes, but we want them it product and its items in JSON format. This JSON format can be fed downstream into another application that may process it.

Let's have go at it.



```python
system_content = """You have knowledge of all sporting goods and will provide knowledge answers
to queries about sporting goods."""
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
prompt = f"""Generate five distinct products on training shoes. Generate products and format them all as a 
            in a single JSON object. For each product, the JSON object should contain items: Brand, Description, Size, Gender: Male 
            or Female or Unisex, Price, and at least three customer reviews as Review 
            item"""
response = get_commpletion(client, MODEL, system_content, prompt)
print(f"\n {BOLD_BEGIN}JSON response:{BOLD_END} {response}\n")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
     [1mJSON response:[0m  {
    "Training Shoes": [
    {
    "Brand": "Nike",
    "Description": "These Nike training shoes are designed for maximum comfort and support during high-intensity workouts. They feature a breathable mesh upper and a cushioned midsole for superior shock absorption.",
    "Size": [8, 9, 10, 11, 12],
    "Gender": "Unisex",
    "Price": 110,
    "Reviews": [
    {
    "Reviewer": "John Doe",
    "Rating": 5,
    "Comment": "I love these shoes! They are so comfortable and have helped me improve my performance during workouts."
    },
    {
    "Reviewer": "Jane Smith",
    "Rating": 4,
    "Comment": "I've been using these shoes for a few weeks now and they are great. They are a bit tight at first but they loosen up over time."
    },
    {
    "Reviewer": "Mike Johnson",
    "Rating": 5,
    "Comment": "These are the best training shoes I've ever owned. They provide great support and cushioning for my heavy workouts."
    }
    ]
    },
    {
    "Brand": "Adidas",
    "Description": "These Adidas training shoes are perfect for CrossFit and other functional fitness workouts. They have a stable and supportive design, with a grippy outsole for excellent traction on various surfaces.",
    "Size": [7, 8, 9, 10, 11],
    "Gender": "Unisex",
    "Price": 120,
    "Reviews": [
    {
    "Reviewer": "Sarah Lee",
    "Rating": 4,
    "Comment": "I like these shoes for their stability and traction. They are a bit snug but I got used to it after a while."
    },
    {
    "Reviewer": "David Park",
    "Rating": 5,
    "Comment": "These are my go-to shoes for CrossFit. They are comfortable, durable, and provide excellent support and grip during my workouts."
    },
    {
    "Reviewer": "Jessica Kim",
    "Rating": 3,
    "Comment": "I was disappointed with the sizing. I had to return them and order a half size up. But once I got the right fit, they were great."
    }
    ]
    },
    {
    "Brand": "Reebok",
    "Description": "These Reebok training shoes are designed for versatile workouts, including weightlifting, running, and jumping. They have a low-cut design for freedom of movement and a solid rubber outsole for durability.",
    "Size": [6, 7, 8, 9, 10],
    "Gender": "Female",
    "Price": 90,
    "Reviews": [
    {
    "Reviewer": "Emily Wong",
    "Rating": 5,
    "Comment": "I love these shoes. They are perfect for my varied workouts and they are so comfortable. I highly recommend them."
    },
    {
    "Reviewer": "Grace Lee",
    "Rating": 4,
    "Comment": "I like the lightweight design and the anatomical fit of these shoes. However, I wish they had more cushioning in the midsole."
    },
    {
    "Reviewer": "Ava Chen",
    "Rating": 5,
    "Comment": "I've had these shoes for a few months now and they are still in great condition. I appreciate the solid construction and the great value for the price."
    }
    ]
    },
    {
    "Brand": "Under Armour",
    "Description": "These Under Armour training shoes are built for high-performance workouts, with a charged cushioning midsole for explosive energy return and a breathable upper for maximum ventilation.",
    "Size": [11, 12, 13, 14, 15],
    "Gender": "Male",
    "Price": 100,
    "Reviews": [
    {
    "Reviewer": "James Lee",
    "Rating": 5,
    "Comment": "These shoes are amazing. They provide great support and cushioning for my heavy weightlifting sessions. I highly recommend them."
    },
    {
    "Reviewer": "Andrew Kim",
    "Rating": 3,
    "Comment": "These shoes fit me well but I found the cushioning a bit hard for my liking. I prefer softer shoes for my running sessions."
    },
    {
    "Reviewer": "Jason Park",
    "Rating": 5,
    "Comment": "These are my second pair of Under Armour training shoes and I'm not disappointed. They are durable, comfortable, and great for my workouts."
    }
    ]
    },
    {
    "Brand": "New Balance",
    "Description": "These New Balance training shoes are designed for all-day comfort and support, with a REVlite midsole for responsive cushioning and a no-sew upper for a smooth and seamless fit.",
    "Size": [8, 9, 10, 11, 12],
    "Gender": "Male",
    "Price": 80,
    "Reviews": [
    {
    "Reviewer": "Kevin Lee",
    "Rating": 4,
    "Comment": "I like these shoes for their comfort and support. However, I wish they had more grip on the outsole for lateral movements."
    },
    {
    "Reviewer": "Daniel Kim",
    "Rating": 5,
    "Comment": "These are my everyday shoes for working out and running errands. They are stylish, comfortable, and have a great price point."
    },
    {
    "Reviewer": "Ryan Park",
    "Rating": 4,
    "Comment": "I'm happy with these shoes. They are a good value for the price. I would recommend them to people who are looking for a comfortable and supportive training shoe."
    }
    ]
    }
    ]
    }
    
    

## Simple and complex reasoning 

An import characteristic of LLM is that it's not only general respository of compressed
knowledge garned from large corpus of text, but can be employed as a simple and complex reasoning engine. With use of precise prompt, you can instruct LLM to think trough a problem in a step by step fashion.

Let's look at some tasks as examples.
 * **Task 1**: given a list of numbers identify the prime numbers, add the prime numbers and check if the sum is even or odd.
 * **Task 2**: given an hourly rate of wages, compute your yearly income if you work 30 hours a week


```python
system_content = """You are a reasoning engine. Given a problem think through the problem logically
in a step by step manner."""
```


```python
# Task 1 prompt
prime_number_prompt = f"""given a list of numbers 1,2,3,4,5,7,8, 11,13,17,19,23,24,29 identify the prime numbers, add the prime numbers, 
and check if the sum is even or odd. Explain each step how you solved the problem"""
```


```python
#Task 2 prompt
hourly_wages_prompt = f"""If my hourly rate is $117.79 per hour and 30 hours a week, what
is my yearly income? Break the problem into simple steps and explain in each step how you arrive 
to the answer. If you don't know, simple say I don't know. Do not make up answers"""

```

#### Task 1


```python
response = get_commpletion(client, MODEL, system_content, prime_number_prompt)
print(f"\n{BOLD_BEGIN}Answer: {BOLD_END}{response}\n")
```

    
    [1mAnswer: [0m Sure, I'd be happy to help you with that!
    
    Step 1: Identify the prime numbers
    A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. The given list of numbers is: 1, 2, 3, 4, 5, 7, 8, 11, 13, 17, 19, 23, 24, 29. We can identify the prime numbers by checking each number to see if it meets the criteria of a prime number.
    
    Starting with 1, we see that it is not a prime number because it has only one positive divisor (1). Moving on to 2, we see that it is a prime number because its only positive divisors are 1 and 2. Continuing this process for the remaining numbers, we find that the prime numbers in the list are: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29.
    
    Step 2: Add the prime numbers
    Now that we have identified the prime numbers, we can add them together to find their sum. The sum of the prime numbers is: 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 = 128.
    
    Step 3: Check if the sum is even or odd
    An even number is any integer that can be divided by 2 with no remainder, and an odd number is any integer that cannot be divided by 2 with no remainder. To check if a number is even or odd, we can look at its last digit. If the last digit is even (0, 2, 4, 6, or 8), then the number is even. If the last digit is odd (1, 3, 5, 7, or 9), then the number is odd.
    
    In this case, the sum of the prime numbers is 128, which has a last digit of 8. Therefore, the sum is an even number.
    
    In summary, to solve this problem, we first identified the prime numbers in the given list, which were: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29. We then added these prime numbers together to find their sum, which was 128. Finally, we checked if the sum was even or odd by looking at its last digit, and we found that it was an even number.
    
    

#### Task 2


```python
response = get_commpletion(client, MODEL, system_content, hourly_wages_prompt)
print(f"\n{BOLD_BEGIN}Answer: {BOLD_END}{response}\n")
```

    
    [1mAnswer: [0m Sure, I'd be happy to help you calculate your yearly income based on your hourly rate and weekly hours. Here are the steps we can follow:
    
    Step 1: Calculate your weekly income by multiplying your hourly rate by the number of hours you work each week.
    
    Weekly income = Hourly rate x Hours per week
    Weekly income = $117.79 x 30
    Weekly income = $3533.70
    
    Step 2: To calculate your annual income, we need to multiply your weekly income by the number of weeks in a year. In general, there are 52 weeks in a year, but if we assume that you take two weeks of vacation, then you would work for 50 weeks in a year.
    
    Annual income = Weekly income x Weeks per year
    Annual income = $3533.70 x 50
    Annual income = $176,685
    
    Therefore, based on your hourly rate of $117.79 and working 30 hours per week for 50 weeks in a year, your yearly income would be approximately $176,685.
    
    

## Code generation

Language models like ChatGPT and Llama 2 are really good at generating code. Copilot on GitHub is a cool example of this. You can do lots of different code tasks just by asking in a smart way. Let's check out a few examples to see how it's helpful.

#### Task 1
 * Generate Python code to compute the value of PI using Ray distributed framework




```python
system_content = """You are a supreme CoPilot for a developer. Given a task you can
generate code for that task."""
```


```python
python_code_prompt="""Generate Python code to compute the value of PI using Ray 
distributed framework API. Use the Monte Carlo method to compute the value of PI.
Include in-line comments explaining the code"""
```


```python
response = get_commpletion(client, MODEL, system_content, python_code_prompt)

print(f"\n{BOLD_BEGIN}Generated Python code:{BOLD_END}{response}\n")
```

    
    [1mGenerated Python code:[0m Sure, I can help you with that! The Monte Carlo method is a technique that allows you to approximate the value of PI using random sampling. Here's how you can use the Ray distributed framework API to compute the value of PI using the Monte Carlo method in Python:
    ```python
    import ray
    import random
    
    # Initialize Ray with the number of worker processes to use
    ray.init(num_workers=4)
    
    # Define the Monte Carlo function to compute the value of PI
    @ray.remote
    def monte_carlo_pi(n):
        """
        Compute the value of PI using the Monte Carlo method.
    
        :param n: The number of random points to generate.
        :return: The estimated value of PI.
        """
        inside_circle = 0
    
        # Generate n random points in the unit square
        for _ in range(n):
            x = random.random()
            y = random.random()
    
            # Check if the point is inside the unit circle
            if x**2 + y**2 <= 1.0:
                inside_circle += 1
    
        # Compute the estimated value of PI
        pi_est = 4.0 * inside_circle / n
        return pi_est
    
    # Define the number of random points to generate per worker process
    num_points = 1000000
    
    # Compute the estimated value of PI in parallel using Ray
    pi_est_list = ray.get([monte_carlo_pi.remote(num_points) for _ in range(4)])
    
    # Combine the results from each worker process to get the final estimate
    pi_est = sum(pi_est_list) / len(pi_est_list)
    
    # Print the final estimate of PI
    print("Estimated value of PI: {:.6f}".format(pi_est))
    
    # Shutdown Ray
    ray.shutdown()
    ```
    In this code, we first initialize Ray with the number of worker processes to use. We then define a remote function `monte_carlo_pi` that computes the value of PI using the Monte Carlo method. This function generates `n` random points in the unit square, checks how many of them are inside the unit circle, and computes the estimated value of PI based on that.
    
    We then define the number of random points to generate per worker process (`num_points`) and compute the estimated value of PI in parallel using Ray by calling `monte_carlo_pi.remote(num_points)` for each worker process. We use `ray.get` to wait for the results from all the worker processes and then combine them to get the final estimate.
    
    Finally, we print the final estimate of PI and shut down Ray.
    
    Note that this code assumes that you have already installed Ray and that you have enough resources (CPU cores, memory, etc.) to run the worker processes in parallel.
    
    

#### Task 2
 * Given SQL schema tables, generate an SQL query 




```python
sql_code_prompt="""Given the following SQL schema for tables
Table clicks, columns = [target_url, orig_url, user_id, clicks]
Table users, columns = [user_id, f_name, l_name, e_mail, company, title], generate
an SQL query that computes in the descening order of all the clicks. Also, for
each user_id, list the f_name, l_name, company, and title
"""
```


```python
response = get_commpletion(client, MODEL, system_content, sql_code_prompt)
print(f"\n{BOLD_BEGIN}Generated SQL code: {BOLD_END}{response}\n")
```

    
    [1mGenerated SQL code: [0m Sure, here is the SQL query that will give you the desired result:
    ```vbnet
    SELECT 
      c.user_id, 
      u.f_name, 
      u.l_name, 
      u.company, 
      u.title, 
      SUM(c.clicks) as total_clicks
    FROM 
      clicks c
    JOIN 
      users u ON c.user_id = u.user_id
    GROUP BY 
      c.user_id, 
      u.f_name, 
      u.l_name, 
      u.company, 
      u.title
    ORDER BY 
      total_clicks DESC;
    ```
    This query joins the `clicks` table with the `users` table on the `user_id` column, then groups the results by `user_id`, `f_name`, `l_name`, `company`, and `title`. It then calculates the sum of the `clicks` column for each group, and orders the final result set by the `total_clicks` column in descending order.
    
    

## All this is amazing! 😜 Feel the wizardy prompt power 🧙‍♀️
