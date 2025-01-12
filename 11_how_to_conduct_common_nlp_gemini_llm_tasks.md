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
To run any of these relevant notebooks you will need an account on Anyscale Endpoints, Anthropic, Gemini, or OpenAI, depending on what model you elect, along with the respective environment file. Use the template environment files to create respective `.env` file for either Anyscale Endpoints, Anthropic, or OpenAI.



```python
import warnings
import os

from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
```


```python
BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"
```


```python
import google.generativeai as genai

_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
api_key = os.getenv("GOOLE_API_KEY")
MODEL = os.getenv("MODEL")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL)
print(f"Using MODEL={MODEL}")
```

    Using MODEL=gemini-1.5-flash
    


```python
from llm_clnt_factory_api import ClientFactory, get_commpletion

client_factory = ClientFactory()
client_factory.register_client('google', genai.GenerativeModel)
client_type = 'google'
client_kwargs = {"model_name": "gemini-1.5-flash",
                     "generation_config": {"temperature": 0.8,}
                }

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
print(f"Using Endpoints: Google Gemini ...\n")
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

    Using Endpoints: Google Gemini ...
    
    
    [1mPrompt:[0m On cold winter nights, the wolves in Siberia ...
    
    [1mAnswer:[0m On cold winter nights, the wolves in Siberia gather in packs, their thick fur offering protection from the biting wind. They howl at the moon, their mournful cries echoing across the vast, snow-covered landscape. These ancient creatures, adapted to survive in one of the world's harshest environments, are a testament to the resilience of nature. Their presence in Siberia, a land of extreme cold and isolation, reminds us of the interconnectedness of all living things and the importance of respecting the delicate balance of our planet. 
    
    
    [1mPrompt:[0m On the day Franklin Benjamin realized his passion for printer, ...
    
    [1mAnswer:[0m On the day Franklin Benjamin realized his passion for printing, he was a young apprentice working in a Boston print shop. The clatter of the printing press and the smell of ink captivated him, sparking a desire to learn the craft. He devoured every scrap of knowledge he could find about printing, from the mechanics of the press to the art of typography.  Benjamin's insatiable curiosity and dedication led him to master the printing trade, a skill that would prove invaluable in his later life as a writer, publisher, and politician. 
    
    
    [1mPrompt:[0m During the final World Cup 1998 when France beat Brazil in Paris, ...
    
    [1mAnswer:[0m During the final World Cup 1998 when France beat Brazil in Paris, the world witnessed a dramatic and historic moment in football. This victory solidified France's dominance in the sport and marked a defining moment in their national identity.  The iconic image of Zinedine Zidane, the French captain, hoisting the World Cup trophy is etched in the minds of football fans worldwide, a symbol of French triumph and a testament to the team's brilliance.  France's victory also signaled a changing of the guard in world football, ushering in a new era for the French national team and leaving a lasting legacy on the sport. 
    
    
    [1mPrompt:[0m Issac Newton set under a tree when an apple fell...
    
    [1mAnswer:[0m Issac Newton set under a tree when an apple fell from its branch, landing squarely on his head. This seemingly mundane event sparked a profound insight in the great physicist: gravity. This pivotal moment led Newton to develop his famous law of universal gravitation, which explained the force that attracts all objects with mass towards one another. His groundbreaking work revolutionized our understanding of the universe, paving the way for future scientific discoveries. 
     
    
    

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
print(f"Using Endpoints: Google Gemini ...\n")
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

    Using Endpoints: Google Gemini ...
    
    
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
    
    [1mSummary  content:[0m Large language models (LLMs) have revolutionized natural language processing (NLP), but they often generate inaccurate or fabricated information, known as hallucinations. These hallucinations pose significant challenges for deploying LLMs in real-world applications, leading researchers to focus on detecting and mitigating them. This article delves into the intricacies of LLM hallucinations, providing a comprehensive overview of their causes, detection methods, mitigation strategies, and current limitations, ultimately aiming to guide future research in this crucial area. 
    
    
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
    
    [1mSummary  content:[0m Can a large language model (LLM) truly reason? While GPT-4, a cutting-edge LLM, excels in language tasks, its abstract reasoning abilities are limited.  Researchers tested GPT-4 on ARC, a benchmark for abstract reasoning, and found it struggled with even the simplest tasks, particularly when visual input was presented in text format. This study reveals that LLMs like GPT-4, despite their impressive capabilities, struggle with visual reasoning and benefit from object-based representations.  The findings suggest that improving LLMs' ability to process and reason about visual information is a crucial step in advancing their cognitive abilities. 
    
    

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
print(f"Using Endpoints: Google Gemini ...\n")
response = get_commpletion(client, MODEL, system_content, prompt)
print(f"""\n{BOLD_BEGIN}Summary:{BOLD_END} {response.replace("```", "")}""")
```

    Using Endpoints: Google Gemini ...
    
    
    [1mSummary:[0m 
    The customer purchased a Bush Baby plush toy for their niece's birthday, and they were very pleased with the product's softness and cuteness.  The customer was pleasantly surprised by the early arrival and appreciated the prompt delivery and secure packaging, which ensured the toy arrived in perfect condition.
    
    
    Shipping Department: The customer was pleased with the prompt delivery and secure packaging, which ensured the toy arrived in perfect condition.
    
    Sentiment: Positive 
    
    

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
print(f"Using Endpoints: Google Gemini ...\n")
for text in user_prompts:
    prompt = f""" Given ```{text}``` delimited with triple backticks, identify a single key idea being discussed, 
    and label its 'Subject'. Next, enumerate at most three takeways. 
    Use short, simple sentences. """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original content:{BOLD_END} {text}")
    print(f"\n {BOLD_BEGIN}Extracted answers: {BOLD_END} {response}")
```

    Using Endpoints: Google Gemini ...
    
    
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
                    
    
     [1mExtracted answers: [0m ```
    Isaac Newton sat under a tree when an apple fell, an event that, 
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
                    ```
    
    **Subject:** Isaac Newton's contributions to physics
    
    **Takeaways:**
    
    * Newton's law of universal gravitation explained both planetary motion and Earthly phenomena.
    * His work laid the foundation for classical mechanics.
    * Newton's influence has continued for centuries. 
    
    

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
print(f"Using Endpoints: Google Gemini ...\n")
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

    Using Endpoints: Google Gemini ...
    
    
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
    
    
     [1mExtracted entities:[0m Subjects: 
    Benjamin Franklin, Printing, Journalism, Literature,  Public Opinion
    
    Takeways:
    - Benjamin Franklin was a printer and journalist who used his skills to shape public opinion.
    - He was an avid reader and founded the first subscription library in the colonies.
    - Franklin used satire to criticize the establishment, using the pseudonym "Silence Dogood". 
    
    

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
print(f"Using Endpoints: Google Gemini ...\n")
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

    Using Endpoints: Google Gemini ...
    
    
    [1mSentiment:[0m This movie is a true cinematic gem, blending an engaging plot with superb performances and stunning visuals. A masterpiece that leaves a lasting impression.
    
    [1mLabel    :[0m positive 
    
    
    [1mSentiment:[0m Regrettably, the film failed to live up to expectations, with a convoluted storyline, lackluster acting, and uninspiring cinematography. A disappointment overall.
    
    [1mLabel    :[0m negative 
    
    
    [1mSentiment:[0m The movie had its moments, offering a decent storyline and average performances. While not groundbreaking, it provided an enjoyable viewing experience.
    
    [1mLabel    :[0m positive 
    
    
    [1mSentiment:[0m This city is a vibrant tapestry of culture, with friendly locals, historic landmarks, and a lively atmosphere. An ideal destination for cultural exploration.
    
    [1mLabel    :[0m positive 
    
    
    [1mSentiment:[0m The city's charm is overshadowed by traffic congestion, high pollution levels, and a lack of cleanliness. Not recommended for a peaceful retreat.
    
    [1mLabel    :[0m negative 
    
    
    [1mSentiment:[0m The city offers a mix of experiences, from bustling markets to serene parks. An interesting but not extraordinary destination for exploration.
    
    [1mLabel    :[0m neutral 
    
    
    [1mSentiment:[0m This song is a musical masterpiece, enchanting listeners with its soulful lyrics, mesmerizing melody, and exceptional vocals. A timeless classic.
    
    [1mLabel    :[0m positive 
    
    
    [1mSentiment:[0m The song fails to impress, featuring uninspiring lyrics, a forgettable melody, and lackluster vocals. It lacks the creativity to leave a lasting impact.
    
    [1mLabel    :[0m negative 
    
    
    [1mSentiment:[0m The song is decent, with a catchy tune and average lyrics. While enjoyable, it doesn't stand out in the vast landscape of music.
    
    [1mLabel    :[0m neutral 
    
    
    [1mSentiment:[0m A delightful cinematic experience that seamlessly weaves together a compelling narrative, strong character development, and breathtaking visuals.
    
    [1mLabel    :[0m positive 
    
    
    [1mSentiment:[0m This film, unfortunately, falls short with a disjointed plot, subpar performances, and a lack of coherence. A disappointing viewing experience.
    
    [1mLabel    :[0m negative 
    
    
    [1mSentiment:[0m While not groundbreaking, the movie offers a decent storyline and competent performances, providing an overall satisfactory viewing experience.
    
    [1mLabel    :[0m positive 
    
    
    [1mSentiment:[0m This city is a haven for culture enthusiasts, boasting historical landmarks, a rich culinary scene, and a welcoming community. A must-visit destination.
    
    [1mLabel    :[0m positive 
    
    
    [1mSentiment:[0m The city's appeal is tarnished by overcrowded streets, noise pollution, and a lack of urban planning. Not recommended for a tranquil getaway.
    
    [1mLabel    :[0m negative 
    
    
    [1mSentiment:[0m The city offers a diverse range of experiences, from bustling markets to serene parks. An intriguing destination for those seeking a mix of urban and natural landscapes.
    
    [1mLabel    :[0m positive 
    
    
    [1mSentiment:[0m xxxyyyzzz was curious and dubious
    
    [1mLabel    :[0m neutral 
    
    

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
                    """Your company did a bad job for internet service!""",
                    """I hate your worthless services. Cancel my stupid account or else I'll sue you!"""
                   ]
                    
```


```python
print(f"Using Endpoints: Google Gemini ...\n")
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

    Using Endpoints: Google Gemini ...
    
    
    [1mQuery:[0m My modem has stop working. I tried to restart but the orange light keep flashing. It never turns green.
    
    [1mRoute to:[0m Technical support 
    
    
    
    [1mQuery:[0m I just moved into town, and I need Internet service
    
    [1mRoute to:[0m New Customer 
    
    
    
    [1mQuery:[0m Why does my bill include an extra $20 a month for cable TV when I don't use a television?
    
    [1mRoute to:[0m Billing 
    
    
    
    [1mQuery:[0m I need to change my user name and password since someone is using my credentials. I cannot access my account.
    
    [1mRoute to:[0m Account Management 
    
    
    
    [1mQuery:[0m What days this week are we having a general upgrades to the cable models?
    
    [1mRoute to:[0m General inquiry 
    
    
    
    [1mQuery:[0m What day is the best day to call customer service so that I can avoid talking to a bot!
    
    [1mRoute to:[0m General inquiry 
    
    
    
    [1mQuery:[0m Your company did a bad job for internet service!
    
    [1mRoute to:[0m General inquiry 
    
    
    
    [1mQuery:[0m I hate your worthless services. Cancel my stupid account or else I'll sue you!
    
    [1mRoute to:[0m Account Management 
    
    
    

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
print(f"Using Endpoints: Google Gemini ...\n")
for english_text in english_texts:
    prompt = f""""Given an English text in triple ticks '''{english_text}'''. Translate into
three languases: Spanish, French, German, and Mandarin. 
Label each translation with the langauge Name: followed by translation on a seperate line."""
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}English Text:{BOLD_END} {english_text}")
    print(f"\n{BOLD_BEGIN}Translation: {BOLD_END}{response}\n")
                
```

    Using Endpoints: Google Gemini ...
    
    
    [1mEnglish Text:[0m  Welcome to New York for the United Nations General Council Meeting. Today
    is a special day for us to celeberate all our achievments since this global institute's formation.
    But more importantly, we want to address how we can mitigate global conflict with conversation
    and promote deterence, detente, and discussion.
    
    [1mTranslation: [0mHere are the translations of the text into Spanish, French, German, and Mandarin:
    
    **Spanish:**
    ```
    Bienvenidos a Nueva York para la reunión del Consejo General de las Naciones Unidas. Hoy
    es un día especial para celebrar todos nuestros logros desde la formación de este instituto global.
    Pero lo que es más importante, queremos abordar cómo podemos mitigar el conflicto global con la conversación
    y promover la disuasión, la distensión y el diálogo.
    ```
    
    **French:**
    ```
    Bienvenue à New York pour la réunion du Conseil général des Nations unies. Aujourd'hui
    est un jour spécial pour célébrer toutes nos réalisations depuis la formation de cet institut mondial.
    Mais plus important encore, nous voulons aborder la manière dont nous pouvons atténuer les conflits mondiaux par la conversation
    et promouvoir la dissuasion, la détente et le dialogue.
    ```
    
    **German:**
    ```
    Willkommen in New York zur Sitzung des Generalkonsels der Vereinten Nationen. Heute
    ist ein besonderer Tag für uns, um all unsere Errungenschaften seit der Gründung dieses globalen Instituts zu feiern.
    Aber noch wichtiger ist es, dass wir uns damit auseinandersetzen, wie wir globale Konflikte durch Gespräche
    mindern und Abschreckung, Entspannung und Diskussion fördern können.
    ```
    
    **Mandarin:**
    ```
    欢迎来到纽约参加联合国大会理事会会议。今天
    是我们庆祝这个全球机构成立以来所有成就的特殊日子。
    但更重要的是，我们希望探讨如何通过对话缓解全球冲突
    并促进威慑、缓和和讨论。
    ``` 
    
    
    

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
print(f"Using Endpoints: Google Gemini ...\n")
for language_text in languages_texts:
    prompt = f""""Given a language text in triple ticks '''{language_text}'''. Idenfity
    the language with the langauge Name: followed by an English translation on a seperate line, labeled as English translation:"""
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN} Language Text: {BOLD_END} {language_text}")
    print(f"\n{BOLD_BEGIN}Translation: {BOLD_END} {response}\n")
                
```

    Using Endpoints: Google Gemini ...
    
    
    [1m Language Text: [0m Bienvenidos a Nueva York para la Reunión del Consejo General de las Naciones Unidas. Hoy
    es un día especial para celebrar todos nuestros logros desde la formación de este instituto global.
    Pero más importante aún, queremos abordar cómo podemos mitigar el conflicto global con conversaciones
    y promover la disuasión, la distensión y el diálogo.
    
    [1mTranslation: [0m ```
    Language Name: Spanish
    English translation: Welcome to New York for the General Assembly of the United Nations. Today is a special day to celebrate all our achievements since the formation of this global institution. But more importantly, we want to address how we can mitigate global conflict through dialogue and promote deterrence, détente, and dialogue.
    ``` 
    
    
    
    [1m Language Text: [0m Willkommen in New York zur Sitzung des Allgemeinen Rates der Vereinten Nationen. Heute
    ist ein besonderer Tag für uns, um all unsere Errungenschaften seit der Gründung dieses globalen Instituts zu feiern.
    Aber wichtiger ist, dass wir ansprechen möchten, wie wir globale Konflikte durch Gespräche mildern können
    und Abschreckung, Entspannung und Diskussion fördern.
    
    [1mTranslation: [0m **Language Name:** German
    
    **English Translation:** 
    Welcome to New York for the session of the United Nations General Assembly. Today is a special day for us to celebrate all our achievements since the founding of this global institution. But more importantly, we want to address how we can mitigate global conflicts through dialogue and promote deterrence, détente, and discussion. 
    
    
    
    [1m Language Text: [0m Bienvenue à New York pour la réunion du Conseil Général des Nations Unies. Aujourd'hui,
    c'est un jour spécial pour nous pour célébrer toutes nos réalisations depuis la formation de cette institution mondiale.
    Mais plus important encore, nous voulons aborder comment nous pouvons atténuer les conflits mondiaux grâce à la conversation
    et promouvoir la dissuasion, la détente et la discussion.
    
    [1mTranslation: [0m Language Name: French
    
    English translation: "Welcome to New York for the meeting of the United Nations General Council. Today,
    it's a special day for us to celebrate all our achievements since the formation of this global institution.
    But more importantly, we want to address how we can mitigate global conflicts through conversation
    and promote deterrence, détente, and discussion." 
    
    
    
    [1m Language Text: [0m 欢迎来到纽约参加联合国大会议。今天对我们来说是一个特别的日子，我们将庆祝自该全球机构成立以来取得的所有成就。但更重要的是，我们想要讨论如何通过对话来缓解全球冲突，并促进遏制、缓和和讨论。
    
    
    [1mTranslation: [0m ```
    Language Name: Chinese (Simplified)
    English Translation: Welcome to New York for the United Nations General Assembly. Today is a special day for us, as we celebrate all the achievements made since the founding of this global institution. But more importantly, we want to discuss how we can alleviate global conflicts through dialogue and promote restraint, de-escalation, and discussion.
    ``` 
    
    
    

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
print(f"Using Endpoints: Google Gemini ...\n")
for bad_english_text in bad_english_texts:
    prompt = f""""Proofread and correct the text provided in triple ticks '''{bad_english_text}'''.
    Use standard usage and remedy any incorect grammar usage.
    """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original Text:{BOLD_END} {bad_english_text}")
    print(f"\n{BOLD_BEGIN}Corrected  Text:{BOLD_END} {response}\n")
```

    Using Endpoints: Google Gemini ...
    
    
    [1mOriginal Text:[0m I don't know nothing about them big words and grammar rules. Me and my friend, we was talking, and he don't agree with me. We ain't never gonna figure it out, I reckon. His dog don't listen good, always running around and don't come when you call.
    
    [1mCorrected  Text:[0m "I don't know anything about those big words and grammar rules. My friend and I were talking, and he disagreed with me. We'll never figure it out, I guess. His dog doesn't listen well, always running around and not coming when you call." 
    
    
    
    [1mOriginal Text:[0m Yesterday, we was at the park, and them kids was playing. She don't like the way how they acted, but I don't got no problem with it. We seen a movie last night, and it was good, but my sister, she don't seen it yet. Them books on the shelf, they ain't interesting to me.
    
    [1mCorrected  Text:[0m Here is the corrected text:
    
    "Yesterday, we were at the park, and the kids were playing. She doesn't like the way they acted, but I don't have any problem with it. We saw a movie last night, and it was good, but my sister hasn't seen it yet. Those books on the shelf aren't interesting to me." 
    
    Here's a breakdown of the changes:
    
    * **"was" to "were":**  "We" is a plural subject, so it needs the plural verb "were."
    * **"them" to "the":**  "Them" is a pronoun used for objects, not subjects. "The" is the correct article here. 
    * **"was" to "were":**  Again, "kids" is plural, so it needs the plural verb "were."
    * **"don't like" to "doesn't like":**  "She" is singular, so the verb needs to be singular as well.
    * **"got no" to "have any":**  This phrase is more standard in formal writing.
    * **"seen" to "saw":**  "Saw" is the past tense of "see." 
    * **"don't seen" to "hasn't seen":**  "She" is singular, so the verb needs to be singular.
    * **"them" to "those":** "Those" is the correct demonstrative pronoun to refer to specific books.
    * **"ain't" to "aren't":**  "Aren't" is the standard contraction for "are not." 
    
    
    


```python
pirate_texts = ["""Arrr matey! I be knowin' nuthin' 'bout them fancy words and grammatical rules. Me and me heartie, we be chattin', and he don't be agreein' with me. We ain't never gonna figure it out, I reckon. His scallywag of a dog don't be listenin' well, always runnin' around and not comin' when ye call."""
                       ]
```


```python
print(f"Using Endpoints: Google Gemini ...\n")
for pirate_text in pirate_texts:
    prompt = f""""Convert the Pirate text provided in triple ticks '''{pirate_text}'''.
    Use standard usage and remedy any incorect grammar usage, dropping all Pirate greetings.
    """
    response = get_commpletion(client, MODEL, system_content, prompt)
    print(f"\n{BOLD_BEGIN}Original Text:{BOLD_END} {pirate_text}")
    print(f"\n{BOLD_BEGIN}Corrected  Text:{BOLD_END} {response}\n")
```

    Using Endpoints: Google Gemini ...
    
    
    [1mOriginal Text:[0m Arrr matey! I be knowin' nuthin' 'bout them fancy words and grammatical rules. Me and me heartie, we be chattin', and he don't be agreein' with me. We ain't never gonna figure it out, I reckon. His scallywag of a dog don't be listenin' well, always runnin' around and not comin' when ye call.
    
    [1mCorrected  Text:[0m I don't know anything about those fancy words and grammatical rules. My friend and I are talking, but he doesn't agree with me. We're never going to figure it out, I guess. His dog doesn't listen well, always running around and not coming when you call. 
    
    
    

### Task 3
* Given some text in a particular format, convert it into JSON format.
* For example, we LLM to producce names of three top shoes, but we want them it product and its items in JSON format. This JSON format can be fed downstream into another application that may process it.

Let's have go at it.



```python
system_content = """You have knowledge of all sporting goods and will provide knowledge answers
to queries about sporting goods."""
```


```python
print(f"Using Endpoints: Google Gemini ...\n")
prompt = f"""Generate five distinct products on training shoes. Generate products and format them all as a 
            in a single JSON object. For each product, the JSON object should contain items: Brand, Description, Size, Gender: Male 
            or Female or Unisex, Price, and at least three customer reviews as Review 
            item"""
response = get_commpletion(client, MODEL, system_content, prompt)
print(f"\n {BOLD_BEGIN}JSON response:{BOLD_END} {response}\n")
```

    Using Endpoints: Google Gemini ...
    
    
     [1mJSON response:[0m ```json
    {
      "products": [
        {
          "Brand": "Nike",
          "Description": "Nike Air Max 270 - Sleek and stylish running shoes with a large air unit for maximum cushioning and comfort.",
          "Size": "8-13",
          "Gender": "Unisex",
          "Price": "$160",
          "Reviews": [
            {
              "Rating": 5,
              "Comment": "These shoes are amazing! They're so comfortable and stylish. I get compliments on them all the time."
            },
            {
              "Rating": 4,
              "Comment": "Great for running and everyday wear. The cushioning is incredible.  Only downside is the price."
            },
            {
              "Rating": 3,
              "Comment": "They look great, but I've found they're not the best for high-impact activities.  Good for walking though."
            }
          ]
        },
        {
          "Brand": "Adidas",
          "Description": "Adidas Ultraboost 22 - Lightweight and responsive running shoes with a Boost midsole for energy return and a breathable Primeknit upper.",
          "Size": "7-12",
          "Gender": "Unisex",
          "Price": "$180",
          "Reviews": [
            {
              "Rating": 5,
              "Comment": "These are my go-to running shoes!  Love the fit and feel."
            },
            {
              "Rating": 4,
              "Comment": "Really comfortable and supportive.  Slightly pricey, but worth the investment."
            },
            {
              "Rating": 3,
              "Comment": "They run a little small, so order a half size up."
            }
          ]
        },
        {
          "Brand": "New Balance",
          "Description": "New Balance Fresh Foam X 880 v12 -  Comfortable and versatile shoes with a plush Fresh Foam X midsole for a smooth ride and a breathable upper.",
          "Size": "6-13",
          "Gender": "Male",
          "Price": "$140",
          "Reviews": [
            {
              "Rating": 5,
              "Comment": "Amazing for walking and long runs.  I can wear these for hours without any pain."
            },
            {
              "Rating": 4,
              "Comment": "Great value for the price.  Super comfortable and well-made."
            },
            {
              "Rating": 3,
              "Comment": "The design is a little boring, but they're functional."
            }
          ]
        },
        {
          "Brand": "Brooks",
          "Description": "Brooks Ghost 14 - Durable and responsive running shoes with a soft and plush DNA LOFT midsole and a breathable engineered mesh upper.",
          "Size": "5-12",
          "Gender": "Female",
          "Price": "$130",
          "Reviews": [
            {
              "Rating": 5,
              "Comment": "These are the best running shoes I've ever owned!  They're so comfortable and supportive."
            },
            {
              "Rating": 4,
              "Comment": "Really good for long runs.  They're well-cushioned and don't feel too bulky."
            },
            {
              "Rating": 3,
              "Comment": "They're a bit pricey, but they seem to be holding up well."
            }
          ]
        },
        {
          "Brand": "ASICS",
          "Description": "ASICS GEL-Nimbus 24 -  Lightweight and cushioned running shoes with a GEL technology midsole for shock absorption and a breathable engineered mesh upper.",
          "Size": "6-13",
          "Gender": "Unisex",
          "Price": "$160",
          "Reviews": [
            {
              "Rating": 5,
              "Comment": "These shoes are perfect for running on pavement.  They're super comfortable and provide great support."
            },
            {
              "Rating": 4,
              "Comment": "They run a little wide, but I've found they're great for my high arches."
            },
            {
              "Rating": 3,
              "Comment": "The color options are a bit limited."
            }
          ]
        }
      ]
    }
    ```
    
    

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

    
    [1mAnswer: [0mHere's how we can identify the prime numbers, add them, and determine if the sum is even or odd:
    
    **1. Identifying Prime Numbers:**
    
    * **Definition:** A prime number is a whole number greater than 1 that has only two divisors: 1 and itself. 
    * **Checking the list:**
        * **1:**  Not a prime number (it only has one divisor: 1).
        * **2:** Prime number (divisors: 1 and 2).
        * **3:** Prime number (divisors: 1 and 3).
        * **4:** Not a prime number (divisors: 1, 2, and 4).
        * **5:** Prime number (divisors: 1 and 5).
        * **7:** Prime number (divisors: 1 and 7).
        * **8:** Not a prime number (divisors: 1, 2, 4, and 8).
        * **11:** Prime number (divisors: 1 and 11).
        * **13:** Prime number (divisors: 1 and 13).
        * **17:** Prime number (divisors: 1 and 17).
        * **19:** Prime number (divisors: 1 and 19).
        * **23:** Prime number (divisors: 1 and 23).
        * **24:** Not a prime number (divisors: 1, 2, 3, 4, 6, 8, 12, and 24).
        * **29:** Prime number (divisors: 1 and 29).
    
    **2. Adding the Prime Numbers:**
    
    * Prime numbers from the list: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
    * Sum = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 = 139
    
    **3. Determining Even or Odd:**
    
    * A number is even if it is divisible by 2. 
    * 139 is not divisible by 2. 
    * Therefore, the sum of the prime numbers is **odd**. 
    
    
    

#### Task 2


```python
response = get_commpletion(client, MODEL, system_content, hourly_wages_prompt)
print(f"\n{BOLD_BEGIN}Answer: {BOLD_END}{response}\n")
```

    
    [1mAnswer: [0mHere's how to calculate your yearly income:
    
    **Step 1: Calculate Weekly Earnings**
    
    * Multiply your hourly rate by the number of hours you work per week: $117.79/hour * 30 hours/week = $3533.70/week
    
    **Step 2: Calculate Yearly Earnings**
    
    * Multiply your weekly earnings by the number of weeks in a year (52): $3533.70/week * 52 weeks/year = $183,748.40/year
    
    **Therefore, your yearly income would be $183,748.40.** 
    
    
    

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

    
    [1mGenerated Python code:[0m```python
    import ray
    import random
    
    # Initialize Ray
    ray.init()
    
    # Define a function to simulate a random point within a unit square
    @ray.remote
    def simulate_point():
      """Simulates a random point within a unit square."""
      x = random.uniform(0, 1)
      y = random.uniform(0, 1)
      return x, y
    
    # Define a function to calculate PI using the Monte Carlo method
    @ray.remote
    def calculate_pi(num_points):
      """Calculates PI using the Monte Carlo method."""
      inside_circle = 0
      for _ in range(num_points):
        x, y = simulate_point.remote()
        # Check if the point lies within the unit circle
        if x**2 + y**2 <= 1:
          inside_circle += 1
      return inside_circle
    
    # Number of points to simulate
    num_points = 1000000
    
    # Distribute the simulations across multiple workers
    results = [calculate_pi.remote(num_points // 100) for _ in range(100)]
    
    # Collect the results from the workers
    inside_counts = ray.get(results)
    
    # Calculate the total number of points inside the circle
    total_inside_circle = sum(inside_counts)
    
    # Calculate PI using the Monte Carlo formula
    pi_estimate = 4 * (total_inside_circle / num_points)
    
    # Print the estimated value of PI
    print("Estimated value of PI:", pi_estimate)
    ```
    
    **Explanation:**
    
    1. **Initialization:** The code starts by initializing Ray using `ray.init()`. This creates the necessary infrastructure for distributed execution.
    2. **`simulate_point` Function:** This function simulates a random point within a unit square. It generates random x and y coordinates between 0 and 1 using `random.uniform()`.
    3. **`calculate_pi` Function:** This function uses the Monte Carlo method to estimate PI.
       - It takes the total number of points (`num_points`) as input.
       - It simulates a specified number of points using `simulate_point.remote()` and checks if each point lies within the unit circle (using `x**2 + y**2 <= 1`).
       - The count of points inside the circle is accumulated.
       - The function returns the count of points inside the circle.
    4. **Distributed Simulations:**
       - The code divides the total number of simulations (`num_points`) into smaller batches (100 batches in this case) to distribute the workload across multiple workers.
       - It uses a list comprehension to create a list of `calculate_pi.remote()` calls, each with a portion of the total points.
    5. **Result Collection:**
       - `ray.get(results)` collects the results from all the remote tasks, which are the counts of points inside the circle from each worker.
    6. **PI Calculation:**
       - The code sums up the counts from all workers (`total_inside_circle`) to get the total number of points inside the circle.
       - It then calculates the estimated value of PI using the formula: `4 * (total_inside_circle / num_points)`.
    7. **Output:** Finally, the estimated value of PI is printed.
    
    **Key Points:**
    
    - **Ray Tasks:** The `@ray.remote` decorator defines functions that can be executed on remote workers.
    - **Task Distribution:** The code divides the work into smaller tasks and distributes them to different workers, making use of Ray's distributed execution capabilities.
    - **Result Aggregation:** `ray.get()` efficiently gathers results from all the workers, enabling a unified calculation of PI.
    
    This code provides a straightforward example of using Ray to perform distributed computations for estimating the value of PI using the Monte Carlo method.
    
    
    

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

    
    [1mGenerated SQL code: [0m```sql
    SELECT
      c.user_id,
      u.f_name,
      u.l_name,
      u.company,
      u.title,
      SUM(c.clicks) AS total_clicks
    FROM clicks c
    JOIN users u ON c.user_id = u.user_id
    GROUP BY c.user_id, u.f_name, u.l_name, u.company, u.title
    ORDER BY total_clicks DESC;
    ```
    
    **Explanation:**
    
    1. **`SELECT` Clause:**
       - Selects the following columns:
         - `c.user_id`: The user ID from the `clicks` table.
         - `u.f_name`: The first name from the `users` table.
         - `u.l_name`: The last name from the `users` table.
         - `u.company`: The company name from the `users` table.
         - `u.title`: The job title from the `users` table.
         - `SUM(c.clicks) AS total_clicks`: Calculates the sum of clicks for each user and aliases it as `total_clicks`.
    
    2. **`FROM` Clause:**
       - Specifies the tables involved: `clicks` (aliased as `c`) and `users` (aliased as `u`).
    
    3. **`JOIN` Clause:**
       - Uses an inner join to connect the two tables based on the common column `user_id`.
    
    4. **`GROUP BY` Clause:**
       - Groups the results by `user_id`, `f_name`, `l_name`, `company`, and `title`. This ensures that clicks are aggregated correctly for each unique user.
    
    5. **`ORDER BY` Clause:**
       - Sorts the results in descending order of `total_clicks` to show the users with the most clicks first.
    
    
    

## All this is amazing! 😜 Feel the wizardy prompt power 🧙‍♀️
