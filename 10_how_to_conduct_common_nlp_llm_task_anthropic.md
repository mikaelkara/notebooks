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

from anthropic import Anthropic
from llm_clnt_factory_api import ClientFactory, get_commpletion

from dotenv import load_dotenv, find_dotenv
```


```python
BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"
```


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
api_key = os.getenv("ANTHROPIC_API_KEY", None)
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={'Anthropic'}")
```

    Using MODEL=claude-3-opus-20240229; base=Anthropic
    

#### Creat an anthropic client using our factory class


```python
client_factory = ClientFactory()
client_type = "anthropic"
client_factory.register_client(client_type, Anthropic)
client_kwargs = {"api_key": api_key}
# create the client
client = client_factory.create_client(client_type, **client_kwargs)
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
print(f"Using Endpoints: {'Anthropic'} ...\n")
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

    Using Endpoints: Anthropic ...
    
    
    [1mPrompt:[0m On cold winter nights, the wolves in Siberia ...
    
    [1mAnswer:[0m On cold winter nights, the wolves in Siberia howl in unison, their eerie chorus echoing across the frozen tundra. These resilient creatures, adapted to survive in one of the harshest environments on Earth, form tight-knit packs to hunt prey and protect their territory. Their thick, insulating fur allows them to withstand temperatures that plummet to -50°C (-58°F), while their keen senses help them navigate the vast, snow-covered landscape. Siberian wolves, primarily a subspecies of grey wolf, play a crucial role in maintaining the balance of the ecosystem by controlling the populations of large herbivores such as elk and reindeer. Despite their fearsome reputation, these majestic animals are an integral part of the Siberian wilderness, embodying the untamed spirit of the region.
    
    [1mPrompt:[0m On the day Franklin Benjamin realized his passion for printer, ...
    
    [1mAnswer:[0m Here is my attempt at completing the paragraph:
    
    
    On the day Franklin Benjamin realized his passion for printing, his life changed forever. As a young apprentice in his brother's print shop, Franklin discovered the power of the printed word to spread ideas and shape opinions. He threw himself into mastering the craft, spending long hours setting type and operating the printing press. Franklin's sharp mind and tireless work ethic allowed him to quickly surpass his peers and establish his own successful printing business. This early passion laid the foundation for Franklin's later accomplishments as an inventor, scientist, statesman and Founding Father of the United States.
    
    
    The paragraph provides relevant details about Franklin's early life as a printer's apprentice, how this experience ignited his passion, and the later impact this had on his storied career. It uses a mix of simple, compound and complex sentences to convey the information in a concise, engaging style appropriate for a literary magazine targeting curious college students. Please let me know if you would like me to modify anything in the paragraph.
    
    [1mPrompt:[0m During the final World Cup 1998 when France beat Brazil in Paris, ...
    
    [1mAnswer:[0m During the final World Cup 1998 when France beat Brazil in Paris, the host nation celebrated its first-ever World Cup victory. France's 3-0 triumph over the defending champions was a momentous occasion, as the French team, led by captain Didier Deschamps and star player Zinedine Zidane, put on a dominant performance. The match, played at the Stade de France in front of a capacity crowd of 80,000 spectators, saw Zidane score two goals in the first half, while Emmanuel Petit added a third in the closing minutes. The victory marked a significant milestone for French football and sparked nationwide celebrations, with an estimated one million people gathering on the Champs-Élysées to revel in the historic achievement.
    
    [1mPrompt:[0m Issac Newton set under a tree when an apple fell...
    
    [1mAnswer:[0m Isaac Newton sat under a tree when an apple fell on his head, inspiring his groundbreaking insight into the fundamental force of gravity. This serendipitous event led Newton to formulate the universal law of gravitation, which states that every particle in the universe attracts every other particle with a force proportional to the product of their masses and inversely proportional to the square of the distance between them. Newton's discovery revolutionized our understanding of the physical world, explaining the motion of objects on Earth and in the heavens, from falling apples to orbiting planets. His work laid the foundation for classical mechanics and remains a cornerstone of modern physics.
    

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
print(f"Using Endpoints: {'Anthropic'} ...\n")
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

    Using Endpoints: Anthropic ...
    
    
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
    
    [1mSummary  content:[0m Large language models (LLMs) have revolutionized natural language processing, enabling remarkable advancements in text understanding and generation. However, LLMs are prone to producing hallucinations, generating content inconsistent with real-world facts or user inputs, which poses significant challenges to their practical deployment and raises concerns about their reliability. This survey provides an in-depth overview of recent advances in LLM hallucinations, introducing a taxonomy, exploring contributing factors, presenting detection methods and benchmarks, and discussing mitigation approaches, while also highlighting current limitations and outlining future research directions.
    
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
    
    [1mSummary  content:[0m GPT-4, a state-of-the-art large language model, struggles to solve abstract reasoning problems from the Abstraction and Reasoning Corpus (ARC) benchmark when using textual encodings for two-dimensional input-output grids. The model's ability to identify and reason about objects is significantly influenced by the sequential nature of the text representation. However, when tested on a new benchmark called 1D-ARC, which consists of one-dimensional tasks more conducive to GPT-based reasoning, the model performs better than on the original 2D-ARC. Furthermore, using an object-based representation obtained through an external tool nearly doubles GPT-4's performance on solved ARC tasks and leads to near-perfect scores on the easier 1D-ARC, revealing that object-based representations can significantly improve the model's reasoning ability.
    

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
print(f"Using Endpoints: {'Anthropic'} ...\n")
response = get_commpletion(client, MODEL, system_content, prompt)
print(f"""\n{BOLD_BEGIN}Summary:{BOLD_END} {response.replace("```", "")}""")
```

    Using Endpoints: Anthropic ...
    
    
    [1mSummary:[0m GPT-4, a state-of-the-art large language model, struggles to solve abstract reasoning problems from the Abstraction and Reasoning Corpus (ARC) when using textual encodings for two-dimensional input-output grids. The model's ability to identify and reason about objects is significantly influenced by the sequential nature of the text representation, as evidenced by its improved performance on a newly designed one-dimensional benchmark called 1D-ARC. However, using object-based representations obtained through an external tool nearly doubles GPT-4's performance on solved ARC tasks and leads to near-perfect scores on the easier 1D-ARC, revealing that such representations can significantly enhance the model's reasoning ability within non-language domains.
    

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
print(f"Using Endpoints: {'Anthropic'} ...\n")
for text in user_prompts:
    prompt = f""" Given ```{text}``` delimited with triple backticks, identify a single key idea being discussed, 
    and label its 'Subject'. Next, enumerate at most three takeways. 
    Use short, simple sentences. """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original content:{BOLD_END} {text}")
    print(f"\n {BOLD_BEGIN}Extracted answers: {BOLD_END} {response}")
```

    Using Endpoints: Anthropic ...
    
    
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
    
     [1mExtracted answers: [0m Subject: Hallucinations in Large Language Models (LLMs)
    
    Takeaways:
    1. LLMs have made significant progress in natural language processing but are prone to generating hallucinations.
    2. Hallucinations in LLMs pose challenges to their practical use and reliability.
    3. The survey provides an overview of hallucination taxonomy, contributing factors, detection methods, and mitigation approaches.
    
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
    
     [1mExtracted answers: [0m Subject: GPT-4's abstract reasoning ability on the Abstraction and Reasoning Corpus (ARC)
    
    Takeaways:
    1. GPT-4 struggles to solve even the simplest tasks in the ARC benchmark when using textual encodings of the input-output grids.
    2. The sequential nature of text representations hinders GPT-4's ability to identify and reason about objects within the tasks.
    3. Using object-based representations obtained through an external tool significantly improves GPT-4's performance on both ARC and the newly designed 1D-ARC benchmark.
    

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
print(f"Using Endpoints: {'Anthropic'} ...\n")
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

    Using Endpoints: Anthropic ...
    
    
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
    
    
     [1mExtracted entities:[0m Subjects: printing, journalism, satire, apprenticeship, family
    
    Takeways:
    1. Benjamin Franklin learned the printing trade as an apprentice to his brother James.
    2. Franklin wrote satirical letters under the pseudonym "Silence Dogood" for his brother's newspaper.
    3. James Franklin faced opposition and legal troubles for the content of his newspaper, the New-England Courant.
    

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
]
```


```python
print(f"Using Endpoints: {'Anthropic'} ...\n")
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

    Using Endpoints: Anthropic ...
    
    
    [1mSentiment:[0m This movie is a true cinematic gem, blending an engaging plot with superb performances and stunning visuals. A masterpiece that leaves a lasting impression.
    
    [1mLabel    :[0m positive
    
    [1mSentiment:[0m Regrettably, the film failed to live up to expectations, with a convoluted storyline, lackluster acting, and uninspiring cinematography. A disappointment overall.
    
    [1mLabel    :[0m negative
    
    [1mSentiment:[0m The movie had its moments, offering a decent storyline and average performances. While not groundbreaking, it provided an enjoyable viewing experience.
    
    [1mLabel    :[0m neutral
    
    [1mSentiment:[0m This city is a vibrant tapestry of culture, with friendly locals, historic landmarks, and a lively atmosphere. An ideal destination for cultural exploration.
    
    [1mLabel    :[0m positive
    
    [1mSentiment:[0m The city's charm is overshadowed by traffic congestion, high pollution levels, and a lack of cleanliness. Not recommended for a peaceful retreat.
    
    [1mLabel    :[0m negative
    

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
                    # """What day is the best day to call customer service so that I can avoid talking to a bot!""",
                    # """Your company is full of incompetent morons and fools!""",
                    # """I hate your worthless services. Cancel my stupid account or else I'll sue you!"""
                   ]
                    
```


```python
print(f"Using Endpoints: {'Anthropic'} ...\n")
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

    Using Endpoints: Anthropic ...
    
    
    [1mQuery:[0m My modem has stop working. I tried to restart but the orange light keep flashing. It never turns green.
    
    [1mRoute to:[0m Technical support
    
    
    [1mQuery:[0m I just moved into town, and I need Internet service
    
    [1mRoute to:[0m New Customer
    
    
    [1mQuery:[0m Why does my bill include an extra $20 a month for cable TV when I don't use a television?
    
    [1mRoute to:[0m Billing
    
    
    [1mQuery:[0m I need to change my user name and password since someone is using my credentials. I cannot access my account.
    
    [1mRoute to:[0m Technical support
    
    
    [1mQuery:[0m What days this week are we having a general upgrades to the cable models?
    
    [1mRoute to:[0m General inquiry
    
    

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
print(f"Using Endpoints: {'Anthropic'} ...\n")
for english_text in english_texts:
    prompt = f""""Given an English text in triple ticks '''{english_text}'''. Translate into
three languases: Spanish, French, German, and Mandarin. 
Label each translation with the langauge Name: followed by translation on a seperate line."""
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}English Text:{BOLD_END} {english_text}")
    print(f"\n{BOLD_BEGIN}Translation: {BOLD_END}{response}\n")
                
```

    Using Endpoints: Anthropic ...
    
    
    [1mEnglish Text:[0m  Welcome to New York for the United Nations General Council Meeting. Today
    is a special day for us to celeberate all our achievments since this global institute's formation.
    But more importantly, we want to address how we can mitigate global conflict with conversation
    and promote deterence, detente, and discussion.
    
    [1mTranslation: [0mHere are the translations of the given English text into Spanish, French, German, and Mandarin:
    
    Spanish:
    ''' Bienvenidos a Nueva York para la Reunión del Consejo General de las Naciones Unidas. Hoy
    es un día especial para celebrar todos nuestros logros desde la formación de este instituto global.
    Pero lo que es más importante, queremos abordar cómo podemos mitigar los conflictos globales con la conversación
    y promover la disuasión, la distensión y el diálogo.'''
    
    French:
    ''' Bienvenue à New York pour la réunion du Conseil général des Nations Unies. Aujourd'hui
    est un jour spécial pour célébrer toutes nos réalisations depuis la formation de cet institut mondial.
    Mais plus important encore, nous voulons aborder comment nous pouvons atténuer les conflits mondiaux par la conversation
    et promouvoir la dissuasion, la détente et la discussion.'''
    
    German:
    ''' Willkommen in New York zur Tagung des Generalrats der Vereinten Nationen. Heute
    ist ein besonderer Tag, an dem wir alle unsere Errungenschaften seit der Gründung dieses globalen Instituts feiern.
    Aber was noch wichtiger ist, wir wollen ansprechen, wie wir globale Konflikte durch Gespräche entschärfen können
    und Abschreckung, Entspannung und Diskussion fördern.'''
    
    Mandarin:
    ''' 欢迎来到纽约参加联合国大会。今天
    是我们庆祝这个全球机构成立以来所有成就的特殊日子。
    但更重要的是，我们要讨论如何通过对话缓解全球冲突
    并促进威慑、缓和与讨论。'''
    
    

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
print(f"Using Endpoints: {'Anthropic'} ...\n")
for language_text in languages_texts:
    prompt = f""""Given a language text in triple ticks '''{language_text}'''. Idenfity
    the language with the langauge Name: followed by an English translation on a seperate line, labeled as English translation:"""
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN} Language Text: {BOLD_END} {language_text}")
    print(f"\n{BOLD_BEGIN}Translation: {BOLD_END} {response}\n")
                
```

    Using Endpoints: Anthropic ...
    
    
    [1m Language Text: [0m Bienvenidos a Nueva York para la Reunión del Consejo General de las Naciones Unidas. Hoy
    es un día especial para celebrar todos nuestros logros desde la formación de este instituto global.
    Pero más importante aún, queremos abordar cómo podemos mitigar el conflicto global con conversaciones
    y promover la disuasión, la distensión y el diálogo.
    
    [1mTranslation: [0m Language Name: Spanish
    
    English translation:
    Welcome to New York for the United Nations General Council Meeting. Today
    is a special day to celebrate all our achievements since the formation of this global institute.
    But more importantly, we want to address how we can mitigate global conflict with conversations
    and promote deterrence, détente, and dialogue.
    
    
    [1m Language Text: [0m Willkommen in New York zur Sitzung des Allgemeinen Rates der Vereinten Nationen. Heute
    ist ein besonderer Tag für uns, um all unsere Errungenschaften seit der Gründung dieses globalen Instituts zu feiern.
    Aber wichtiger ist, dass wir ansprechen möchten, wie wir globale Konflikte durch Gespräche mildern können
    und Abschreckung, Entspannung und Diskussion fördern.
    
    [1mTranslation: [0m Language Name: German
    
    English translation:
    Welcome to New York for the meeting of the United Nations General Assembly. Today
    is a special day for us to celebrate all our achievements since the founding of this global institution.
    But more importantly, we want to address how we can mitigate global conflicts through talks
    and promote deterrence, de-escalation, and discussion.
    
    
    [1m Language Text: [0m Bienvenue à New York pour la réunion du Conseil Général des Nations Unies. Aujourd'hui,
    c'est un jour spécial pour nous pour célébrer toutes nos réalisations depuis la formation de cette institution mondiale.
    Mais plus important encore, nous voulons aborder comment nous pouvons atténuer les conflits mondiaux grâce à la conversation
    et promouvoir la dissuasion, la détente et la discussion.
    
    [1mTranslation: [0m Language Name: French
    
    English translation:
    Welcome to New York for the United Nations General Council meeting. Today is a special day for us to celebrate all our achievements since the formation of this global institution. But more importantly, we want to address how we can mitigate global conflicts through conversation and promote deterrence, détente, and discussion.
    
    
    [1m Language Text: [0m 欢迎来到纽约参加联合国大会议。今天对我们来说是一个特别的日子，我们将庆祝自该全球机构成立以来取得的所有成就。但更重要的是，我们想要讨论如何通过对话来缓解全球冲突，并促进遏制、缓和和讨论。
    
    
    [1mTranslation: [0m Language Name: Chinese (Simplified)
    
    English translation:
    Welcome to New York to attend the United Nations General Assembly. Today is a special day for us as we celebrate all the achievements made since the establishment of this global organization. But more importantly, we want to discuss how to alleviate global conflicts through dialogue and promote containment, mitigation, and discussion.
    
    

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
print(f"Using Endpoints: {'Anthropic'} ...\n")
for bad_english_text in bad_english_texts:
    prompt = f""""Proofread and correct the text provided in triple ticks '''{bad_english_text}'''.
    Use standard usage and remedy any incorect grammar usage.
    """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original Text:{BOLD_END} {bad_english_text}")
    print(f"\n{BOLD_BEGIN}Corrected  Text:{BOLD_END} {response}\n")
```

    Using Endpoints: Anthropic ...
    
    
    [1mOriginal Text:[0m I don't know nothing about them big words and grammar rules. Me and my friend, we was talking, and he don't agree with me. We ain't never gonna figure it out, I reckon. His dog don't listen good, always running around and don't come when you call.
    
    [1mCorrected  Text:[0m Here is the corrected version of the text with standard usage and grammar:
    
    '''I don't know anything about those big words and grammar rules. My friend and I were talking, and he doesn't agree with me. We are never going to figure it out, I reckon. His dog doesn't listen well, always running around and not coming when you call.'''
    
    The main corrections made were:
    
    1. "I don't know nothing" changed to "I don't know anything" (double negative corrected)
    2. "them big words" changed to "those big words" (pronoun agreement)
    3. "Me and my friend" changed to "My friend and I" (subject pronoun)
    4. "we was talking" changed to "were talking" (subject-verb agreement)
    5. "he don't agree" changed to "he doesn't agree" (subject-verb agreement) 
    6. "We ain't never gonna" changed to "We are never going to" (non-standard usage of "ain't" and "gonna" corrected)
    7. "don't listen good" changed to "doesn't listen well" (subject-verb agreement and adverb form)
    8. "don't come when you call" changed to "not coming when you call" (parallel structure)
    
    
    [1mOriginal Text:[0m Yesterday, we was at the park, and them kids was playing. She don't like the way how they acted, but I don't got no problem with it. We seen a movie last night, and it was good, but my sister, she don't seen it yet. Them books on the shelf, they ain't interesting to me.
    
    [1mCorrected  Text:[0m Here is the corrected version of the text with standard usage and grammar:
    
    '''
    Yesterday, we were at the park, and those kids were playing. She doesn't like the way they acted, but I don't have a problem with it. We saw a movie last night, and it was good, but my sister hasn't seen it yet. The books on the shelf aren't interesting to me.
    '''
    
    The main corrections made:
    
    1. "we was" changed to "we were" (past tense of "to be")
    2. "them kids" changed to "those kids" (demonstrative pronoun)
    3. "kids was" changed to "kids were" (past tense of "to be") 
    4. "She don't" changed to "She doesn't" (third-person singular present tense)
    5. "the way how they acted" changed to "the way they acted" (redundant use of "how")
    6. "I don't got no" changed to "I don't have a" (double negative and non-standard verb usage)
    7. "We seen" changed to "We saw" (past tense of "to see")
    8. "she don't seen it" changed to "she hasn't seen it" (present perfect tense)
    9. "Them books" changed to "The books" (demonstrative pronoun)
    10. "they ain't" changed to "they aren't" (non-standard contraction of "are not")
    
    


```python
pirate_texts = ["""Arrr matey! I be knowin' nuthin' 'bout them fancy words and grammatical rules. Me and me heartie, we be chattin', and he don't be agreein' with me. We ain't never gonna figure it out, I reckon. His scallywag of a dog don't be listenin' well, always runnin' around and not comin' when ye call."""
                       ]
```


```python
print(f"Using Endpoints: {'Anthropic'} ...\n")
for pirate_text in pirate_texts:
    prompt = f""""Convert the Pirate text provided in triple ticks '''{pirate_text}'''.
    Use standard usage and remedy any incorect grammar usage, dropping all Pirate greetings.
    """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original Text:{BOLD_END} {pirate_text}")
    print(f"\n{BOLD_BEGIN}Corrected  Text:{BOLD_END} {response}\n")
```

    Using Endpoints: Anthropic ...
    
    
    [1mOriginal Text:[0m Arrr matey! I be knowin' nuthin' 'bout them fancy words and grammatical rules. Me and me heartie, we be chattin', and he don't be agreein' with me. We ain't never gonna figure it out, I reckon. His scallywag of a dog don't be listenin' well, always runnin' around and not comin' when ye call.
    
    [1mCorrected  Text:[0m Here is the text converted to standard English with proper grammar and usage, dropping the Pirate greetings and expressions:
    
    I don't know anything about those fancy words and grammatical rules. My friend and I were chatting, and he doesn't agree with me. We're never going to figure it out, I suppose. His mischievous dog doesn't listen well, always running around and not coming when you call.
    
    

### Task 3
* Given some text in a particular format, convert it into JSON format.
* For example, we LLM to producce names of three top shoes, but we want them it product and its items in JSON format. This JSON format can be fed downstream into another application that may process it.

Let's have go at it.



```python
system_content = """You have knowledge of all sporting goods and will provide knowledge answers
to queries about sporting goods."""
```


```python
print(f"Using Endpoints: {'Anthropic'} ...\n")
prompt = f"""Generate five distinct products on training shoes. Generate products and format them all as a 
            in a single JSON object. For each product, the JSON object should contain items: Brand, Description, Size, Gender: Male 
            or Female or Unisex, Price, and at least three customer reviews as Review 
            item"""
response = get_commpletion(client, MODEL, system_content, prompt)
print(f"\n {BOLD_BEGIN}JSON response:{BOLD_END} {response}\n")
```

    Using Endpoints: Anthropic ...
    
    
     [1mJSON response:[0m Here is a JSON object containing five distinct training shoe products with the requested information:
    
    {
      "products": [
        {
          "Brand": "Nike",
          "Description": "Nike Air Zoom Pegasus 38 Running Shoes - Lightweight, breathable mesh upper with responsive cushioning for long runs and training sessions",
          "Size": "7-15",
          "Gender": "Male",
          "Price": 129.99,
          "Reviews": [
            "Great shoes for daily training and long runs. The Zoom Air unit provides excellent cushioning.",
            "These shoes are incredibly comfortable right out of the box. No break-in period needed!",
            "The Pegasus 38 is a reliable, durable shoe that performs well for various running distances."
          ]
        },
        {
          "Brand": "Adidas",
          "Description": "Adidas Ultraboost 21 Running Shoes - Primeknit upper with Boost midsole for energy return and adaptive fit",
          "Size": "5-12",
          "Gender": "Female",
          "Price": 179.99,
          "Reviews": [
            "The Ultraboost 21 is my go-to shoe for long runs. The Boost midsole is incredibly responsive.",
            "I love the snug, adaptive fit of the Primeknit upper. It feels like a second skin.",
            "These shoes are worth the investment. They provide excellent support and comfort."
          ]
        },
        {
          "Brand": "ASICS",
          "Description": "ASICS Gel-Kayano 28 Running Shoes - Engineered mesh upper with Gel cushioning and Dynamic DuoMax support system",
          "Size": "6-14",
          "Gender": "Unisex",
          "Price": 159.99,
          "Reviews": [
            "The Gel-Kayano 28 is a fantastic stability shoe. It provides great support for overpronators.",
            "I've been wearing Kayanos for years, and the 28 does not disappoint. It's comfortable and supportive.",
            "If you need a reliable stability shoe, the Kayano 28 is an excellent choice."
          ]
        },
        {
          "Brand": "New Balance",
          "Description": "New Balance Fresh Foam 880v11 Running Shoes - Hypoknit upper with Fresh Foam midsole for soft, responsive cushioning",
          "Size": "7-15",
          "Gender": "Male",
          "Price": 129.99,
          "Reviews": [
            "The Fresh Foam 880v11 is a great daily trainer. It's comfortable and provides a smooth ride.",
            "I appreciate the roomy toe box and the soft, responsive cushioning of the Fresh Foam midsole.",
            "New Balance has done it again with the 880v11. It's a reliable, well-cushioned shoe for daily training."
          ]
        },
        {
          "Brand": "Saucony",
          "Description": "Saucony Ride 14 Running Shoes - FORMFIT upper with PWRRUN cushioning for a responsive, comfortable ride",
          "Size": "5-12",
          "Gender": "Female",
          "Price": 129.99,
          "Reviews": [
            "The Ride 14 is a versatile shoe that performs well for both short and long runs.",
            "I love the comfortable, secure fit of the FORMFIT upper. It adapts to my foot's shape.",
            "Saucony has created a winner with the Ride 14. It's a great all-around shoe for various workouts."
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

    
    [1mAnswer: [0mGreat! Let's solve this problem step by step. We need to identify the prime numbers from the given list, add them together, and determine if the sum is even or odd.
    
    Given list: 1, 2, 3, 4, 5, 7, 8, 11, 13, 17, 19, 23, 24, 29
    
    Step 1: Identify the prime numbers in the list.
    A prime number is a number greater than 1 that has no positive divisors other than 1 and itself.
    
    1 is not a prime number.
    2 is a prime number.
    3 is a prime number.
    4 is not a prime number (divisible by 2).
    5 is a prime number.
    7 is a prime number.
    8 is not a prime number (divisible by 2 and 4).
    11 is a prime number.
    13 is a prime number.
    17 is a prime number.
    19 is a prime number.
    23 is a prime number.
    24 is not a prime number (divisible by 2, 3, 4, 6, 8, and 12).
    29 is a prime number.
    
    The prime numbers in the list are: 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29.
    
    Step 2: Add the prime numbers.
    2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 = 129
    
    Step 3: Check if the sum is even or odd.
    A number is even if it is divisible by 2 without a remainder. If there is a remainder, the number is odd.
    
    129 ÷ 2 = 64 remainder 1
    Since there is a remainder of 1, the sum (129) is an odd number.
    
    Therefore, the sum of the prime numbers in the given list is 129, which is an odd number.
    
    

#### Task 2


```python
response = get_commpletion(client, MODEL, system_content, hourly_wages_prompt)
print(f"\n{BOLD_BEGIN}Answer: {BOLD_END}{response}\n")
```

    
    [1mAnswer: [0mTo calculate your yearly income, let's break it down into simple steps:
    
    Step 1: Calculate your weekly income.
    Weekly income = Hourly rate × Hours worked per week
    Weekly income = $117.79 × 30 = $3,533.70
    
    Step 2: Calculate the number of weeks in a year.
    There are approximately 52 weeks in a year.
    
    Step 3: Calculate your yearly income.
    Yearly income = Weekly income × Number of weeks in a year
    Yearly income = $3,533.70 × 52 = $183,752.40
    
    Therefore, if your hourly rate is $117.79 per hour and you work 30 hours a week, your yearly income would be approximately $183,752.40, assuming you work all 52 weeks in a year without any unpaid time off.
    
    

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

    
    [1mGenerated Python code:[0mHere's the Python code to compute the value of PI using the Monte Carlo method with the Ray distributed framework API, along with in-line comments explaining the code:
    
    ```python
    import ray
    import random
    import math
    
    # Initialize Ray
    ray.init()
    
    # Define the number of total points to generate
    NUM_POINTS = 10000000
    
    # Define a remote function to generate points and count the ones inside the unit circle
    @ray.remote
    def monte_carlo_pi(num_points):
        inside_count = 0
        for _ in range(num_points):
            # Generate random x and y coordinates between -1 and 1
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            
            # Check if the point lies inside the unit circle
            if x**2 + y**2 <= 1:
                inside_count += 1
        
        return inside_count
    
    # Define the number of parallel tasks to run
    NUM_TASKS = 10
    
    # Split the total number of points among the tasks
    points_per_task = NUM_POINTS // NUM_TASKS
    
    # Launch parallel tasks to compute the count of points inside the unit circle
    results = [monte_carlo_pi.remote(points_per_task) for _ in range(NUM_TASKS)]
    
    # Retrieve the results from the parallel tasks
    inside_counts = ray.get(results)
    
    # Sum up the counts from all the tasks
    total_inside_count = sum(inside_counts)
    
    # Calculate the approximation of PI
    pi_approx = 4 * total_inside_count / NUM_POINTS
    
    # Print the approximated value of PI
    print(f"Approximated value of PI: {pi_approx}")
    print(f"Actual value of PI: {math.pi}")
    ```
    
    Explanation of the code:
    
    1. We import the necessary libraries: `ray` for distributed computing, `random` for generating random numbers, and `math` for comparing the approximated value of PI with the actual value.
    
    2. We initialize Ray using `ray.init()` to start the Ray runtime.
    
    3. We define the total number of points to generate (`NUM_POINTS`) for the Monte Carlo simulation.
    
    4. We define a remote function `monte_carlo_pi` using the `@ray.remote` decorator. This function generates a specified number of random points and counts the number of points that lie inside the unit circle.
    
    5. Inside the `monte_carlo_pi` function, we generate random x and y coordinates between -1 and 1 using `random.uniform()`. We then check if the generated point lies inside the unit circle by calculating the distance from the origin (0, 0) using the equation `x^2 + y^2 <= 1`. If the point is inside the circle, we increment the `inside_count`.
    
    6. We define the number of parallel tasks to run (`NUM_TASKS`) and calculate the number of points to be generated by each task (`points_per_task`).
    
    7. We launch parallel tasks using a list comprehension and the `monte_carlo_pi.remote()` function, passing `points_per_task` as an argument to each task.
    
    8. We retrieve the results from the parallel tasks using `ray.get(results)`, which returns a list of `inside_counts` from each task.
    
    9. We sum up the `inside_counts` from all the tasks to get the total count of points inside the unit circle.
    
    10. We calculate the approximation of PI using the formula: `4 * total_inside_count / NUM_POINTS`.
    
    11. Finally, we print the approximated value of PI and compare it with the actual value of PI using `math.pi`.
    
    This code demonstrates how to use the Ray distributed framework to parallelize the Monte Carlo simulation for approximating the value of PI. By distributing the workload across multiple tasks, we can leverage the power of parallel computing to speed up the calculation.
    
    

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

    
    [1mGenerated SQL code: [0mHere's the SQL query that computes the total clicks in descending order and lists the first name, last name, company, and title for each user:
    
    ```sql
    SELECT 
        u.user_id,
        u.f_name,
        u.l_name,
        u.company,
        u.title,
        COALESCE(SUM(c.clicks), 0) AS total_clicks
    FROM 
        users u
        LEFT JOIN clicks c ON u.user_id = c.user_id
    GROUP BY 
        u.user_id,
        u.f_name,
        u.l_name,
        u.company,
        u.title
    ORDER BY 
        total_clicks DESC;
    ```
    
    Let's break down the query:
    
    1. The `SELECT` clause specifies the columns we want to retrieve:
       - `u.user_id`: The user ID from the `users` table.
       - `u.f_name`: The first name from the `users` table.
       - `u.l_name`: The last name from the `users` table.
       - `u.company`: The company from the `users` table.
       - `u.title`: The title from the `users` table.
       - `COALESCE(SUM(c.clicks), 0) AS total_clicks`: Calculates the total clicks for each user. The `COALESCE` function is used to handle cases where a user has no clicks, returning 0 instead of NULL.
    
    2. The `FROM` clause specifies the main table, which is the `users` table aliased as `u`.
    
    3. The `LEFT JOIN` clause joins the `clicks` table (aliased as `c`) with the `users` table based on the `user_id` column. It ensures that all users are included in the result, even if they have no corresponding clicks.
    
    4. The `GROUP BY` clause groups the result by the specified columns:
       - `u.user_id`
       - `u.f_name`
       - `u.l_name`
       - `u.company`
       - `u.title`
       This allows us to calculate the total clicks for each unique combination of user ID, first name, last name, company, and title.
    
    5. The `ORDER BY` clause sorts the result in descending order based on the `total_clicks` column, so users with the highest number of clicks appear first.
    
    This query will return a result set with columns: `user_id`, `f_name`, `l_name`, `company`, `title`, and `total_clicks`, sorted in descending order of `total_clicks`.
    
    

## All this is amazing! 😜 Feel the wizardy prompt power 🧙‍♀️
