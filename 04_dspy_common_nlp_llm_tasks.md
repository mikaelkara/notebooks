# Common NLP tasks using DSPy Signatures and Modules

<img src="images/dspy_img.png" height="35%" width="%65">

### Quick overview DSPY Programming model
DSPy is a framework for optimizing Language Model (LM) prompts and weights in complex systems, especially when using LMs multiple times within a pipeline. 

The process of using LMs without DSPy involves breaking down problems into steps, prompting the LM effectively for each step, adjusting steps to work together, generating synthetic examples for tuning, and finetuning smaller LMs to reduce costs. This process is currently challenging and messy, requiring frequent changes to prompts or finetuning steps whenever the pipeline, LM, or data are altered. 

DSPy addresses these issues by separating program flow from LM parameters and introducing new optimizers that tune LM prompts and/or weights based on a desired metric. DSPy can train powerful models like GPT-3.5 and GPT-4, as well as smaller models such as T5-base or Llama2-13b, to perform more reliably at tasks by optimizing their prompts and weights. 

DSPy optimizers generate custom instructions, few-shot prompts, and weight updates for each LM, creating a new paradigm where LMs and their prompts are treated as optimizable components of a larger learning system. 

In summary, DSPy enables less prompting, higher scores, and a more systematic approach to solving complex tasks using Language Models.

**Summary**: 
1. DSPy is a framework for optimizing LM prompts and weights in complex systems.
2. Using LMs without DSPy requires breaking down problems into steps, prompting effectively, adjusting steps, generating synthetic examples, and finetuning smaller LMs.
3. This process is challenging due to frequent changes needed when altering pipelines, LMs, or data.
4. DSPy separates program flow from LM parameters and introduces new optimizers that tune prompts and weights based on a metric.
5. DSPy can train powerful and smaller models to perform more reliably at tasks by optimizing their prompts and weights.
6. DSPy optimizers generate custom instructions, few-shot prompts, and weight updates for each LM.
7. This new paradigm treats LMs and their prompts as optimizable components of a larger learning system.
8. Less prompting is required with DSPy, leading to higher scores
9. The approach is more systematic and addresses the challenges of using LMs in complex systems.
10. DSPy enables a new way to train and utilize Language Models effectively

### Signature

"A signature is a declarative specification of input/output behavior of a DSPy module. Signatures allow you to tell the LM what it needs to do, rather than specify how we should ask the LM to do it," states the docs.  

A Signature is composition of three fields: 
 * Task description
 * Input
 * Output

<img src="images/dspy_signature.png">

A Signature class abstracts the above and allows you to express your tasks, with its
input and output (response). Internally, the framework converts a Signature class
into a prompt. Declaratively specifying the specs, they define and dictate the behavior of any module we use in DSPy. All Siganture implementation details of 
Signature employed in the this notebook to carry out common NLP tasks are defined in [DSPy Utils file](./dspy_utils.py).

<img src="images/class_based_prompt_creation.png">

Implementation details of all the Signatures for this notebook to carry out
all common NLP tasks is [DSPy Utils file](./dspy_utils.py).

## Natural language processing (NLP) LLM Tasks

The tasks explored in this notebook, using sophiscated DSPy declarative
signatures, show *how-to* code examples for common natural language understanfing capabilites of a generalized LLM, such as ChatGPT, OLlama, Mistral, and Llama 3 series:

 * Text generation or completion
 * Text summarization
 * Text extraction
 * Text classification or sentiment analysis
 * Text categorization
 * Text transformation and translation
 * Simple and complex reasoning

**Note**: 
To run any of these relevant notebooks you will need to install OLlama on the local
latop or use any LLM-hosted provder service: OpenAI, Anyscale Endpoints, Anthropic, etc.



```python
import dspy
from dspy_utils import TextCompletion, SummarizeText, \
    SummarizeTextAndExtractKeyTheme, TranslateText, \
    TextTransformationAndCorrection, TextCorrection, GenerateJSON, \
    ClassifyEmotion, TextCategorizationAndSentimentAnalsysis, \
    TranslateTextToLanguage, SimpleAndComplexReasoning, WordMathProblem, \
    BOLD_BEGIN, BOLD_END
```

### Setup OLlama environment on the local machine


```python
ollama_mistral = dspy.OllamaLocal(model='mistral', max_tokens=2500)
dspy.settings.configure(lm=ollama_mistral)
```


```python
MODEL = "ollama/mistral"
print(f"Using MODEL={MODEL}; base=localhost")
```

    Using MODEL=ollama/mistral; base=localhost
    

## NLP Task 1: Text Generation and Completion
Use class signatures for text completion

In this simple task, we use an LLM to generate text by finishing an incomplete user content provided in the prompt. For example, by providing an incomplete prompt such as "On a cold winter night, the stray dog ...". 

Let's try a few text generation or completion tasks by providing partial prompts in the user content. You will surprised at its fluency and coherency in the generated text.


```python
PROMPTS =  ["On cold winter nights, the wolves in Siberia ...",
                 "On the day Franklin Benjamin realized his passion for printer, ...",
                 "During the final World Cup 1998 when France beat Brazil in Paris, ...",
                 "Issac Newton set under a tree when an apple fell..."
            ]
```


```python
print("NLP Task 1: Text Generation and Completion")
# Create an instance module Predict with Signature TextCompletion
complete = dspy.Predict(TextCompletion)
# loop over all prompts
for prompt in PROMPTS:
    response = complete(in_text=prompt)
    print(f"{BOLD_BEGIN}Prompt:{BOLD_END}")
    print(prompt)
    print(f"{BOLD_BEGIN}Completion: {BOLD_END}")
    print(response.out_text)
    print("-------------------")
```

    NLP Task 1: Text Generation and Completion
    [1mPrompt:[0m
    On cold winter nights, the wolves in Siberia ...
    [1mCompletion: [0m
    Out Text: On cold winter nights, the wolves in Siberia huddle together in their packs to conserve body heat and survive the harsh Arctic climate. Their thick fur coats provide insulation against the freezing temperatures, while their keen senses help them locate prey through the snowy landscape. The howls of these magnificent animals echo across the vast tundra, adding a hauntingly beautiful soundtrack to the stillness of the night. Despite the challenges they face, wolves in Siberia continue to thrive and play an essential role in maintaining the delicate balance of their ecosystem.
    -------------------
    [1mPrompt:[0m
    On the day Franklin Benjamin realized his passion for printer, ...
    [1mCompletion: [0m
    On the day Franklin Benjamin realized his passion for printing, he felt a sense of excitement and purpose that had been missing from his life. He had always been drawn to the intricacies of typography and the artistry involved in creating beautifully designed pages. As he watched the ink dance across the paper, forming words and sentences, he knew that this was what he wanted to do with his life.
    
    Franklin spent hours poring over books about printing techniques and design principles. He practiced setting type by hand, perfecting his alignment and spacing. He experimented with different fonts and colors, seeking out the perfect combination for each project. And as he worked, he felt a deep sense of satisfaction and fulfillment that came from creating something tangible and beautiful.
    
    Determined to turn his passion into a career, Franklin enrolled in a printing apprenticeship program. He spent long hours learning the ins and outs of the trade, from typesetting and proofreading to operating the presses and binding the finished products. And as he gained more experience and knowledge, he began to develop a reputation as a skilled and dedicated printer.
    
    Years passed, and Franklin's passion for printing never wavered. He continued to hone his craft, experimenting with new techniques and technologies as they emerged. And through it all, he remained committed to creating beautiful, high-quality printed materials that brought joy and inspiration to others.
    
    In the end, Franklin's love of printing became more than just a hobby or a career – it was a lifelong passion that defined who he was and brought meaning and purpose to his life. And as he looked back on all that he had accomplished, he knew that he had made the right choice in following his heart and pursuing his dreams.
    -------------------
    [1mPrompt:[0m
    During the final World Cup 1998 when France beat Brazil in Paris, ...
    [1mCompletion: [0m
    Out Text: During the final World Cup 1998 held in France, the host nation faced off against the reigning champions, Brazil, at the Stade de France in Paris. The tension was palpable as both teams gave their all on the field. In a thrilling match that went into extra time, France emerged victorious with a golden goal scored by Ronaldo Henrique in the 120th minute. The crowd erupted in jubilation as France secured their first-ever World Cup title. This historic moment marked the end of an era and the beginning of a new one for French football.
    -------------------
    [1mPrompt:[0m
    Issac Newton set under a tree when an apple fell...
    [1mCompletion: [0m
    Out Text: Isaac Newton was sitting under a tree in Lincolnshire, England, pondering the mysteries of the universe when an apple allegedly fell from the tree and struck him on the head. This simple event is said to have inspired Newton to develop his theories of gravity and motion. According to legend, this incident occurred in 1665 or 1666, during which time Newton was retreating from London due to an outbreak of the bubonic plague. It was during this period of isolation that he made significant progress on his groundbreaking work "Philosophiæ Naturalis Principia Mathematica," which was published in 1687 and laid the foundation for classical mechanics.
    -------------------
    

### Inspect the Prompt generated for the LLM


```python
# Print the prompt history
print("Prompt history:")
print(ollama_mistral.history[0]["prompt"])
print("-------------------")
```

    Prompt history:
    Complete a given text with more words to the best 
        of your acquired knowledge. Don't truncate the generated
        response.
    
    ---
    
    Follow the following format.
    
    In Text: ${in_text}
    Out Text: ${out_text}
    
    ---
    
    In Text: On cold winter nights, the wolves in Siberia ...
    Out Text:
    -------------------
    

## NLP Task 2: Text Summarization
Use Signatures class module for summarization

A common task in natural langauge processing is text summiarization. A common use case
is summarizing large articles or documents, for a quick and easy-to-absorb summaries.

You can instruct LLM to generate the response in a preferable style, and comprehensibility. For example, use simple language aimed for a certain grade level, keep the orginal style of the article, use different sentence sytles (as we have done in few of examples in this notebook and previous one).

Let's try a few examples.


```python
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
         data are available at this https URL.""",
    """ DSPy is a framework for optimizing Language Model (LM) prompts and weights in 
        complex systems, especially when using LMs multiple times within a pipeline. 
        The process of using LMs without DSPy involves breaking down problems into steps, 
        prompting the LM effectively for each step, adjusting steps to work together, 
        generating synthetic examples for tuning, and finetuning smaller LMs to reduce costs.
        This process is currently challenging and messy, requiring frequent changes to prompts
        or finetuning steps whenever the pipeline, LM, or data are altered. DSPy addresses 
        these issues by separating program flow from LM parameters and introducing new 
        optimizers that tune LM prompts and/or weights based on a desired metric. 
        DSPy can train powerful models like GPT-3.5 and GPT-4, as well as smaller 
        models such as T5-base or Llama2-13b, to perform more reliably at tasks 
        by optimizing their prompts and weights. DSPy optimizers generate custom 
        instructions, few-shot prompts, and weight updates for each LM, creating a 
        new paradigm where LMs and their prompts are treated as optimizable components
        of a larger learning system. In summary, DSPy enables less prompting, higher 
        scores, and a more systematic approach to solving complex tasks using 
        Language Models.
    """
]
```


```python
print("NLP Task 2: Text Summarization")
# Create an instance module Predict with Signature SummarizeText
summarize = dspy.Predict(SummarizeText)
for prompt in user_prompts:
    print(f"{BOLD_BEGIN}Summarization of text response:{BOLD_END}")
    response = summarize(text=prompt)
    print(response.summary)
    print("-------------------")
```

    NLP Task 2: Text Summarization
    [1mSummarization of text response:[0m
    Text: The emergence of large language models (LLMs) has brought about significant advancements in natural language processing (NLP), enabling impressive text understanding and generation capabilities. However, these models have a critical issue: they tend to produce "hallucinations," resulting in content that contradicts real-world facts or user inputs. This phenomenon poses challenges for practical deployment and raises concerns over LLMs' reliability in real-world scenarios. In this survey, we provide an extensive overview of recent research on LLM hallucinations.
    
    Firstly, we propose a novel taxonomy for categorizing LLM hallucinations. Secondly, we explore the factors contributing to these hallucinations. Thirdly, we delve into various methods and benchmarks used for detecting hallucinations in LLMs. Fourthly, we introduce representative approaches designed to mitigate or prevent hallucinations. Lastly, we discuss current limitations and open questions, outlining potential research directions for addressing hallucinations in LLMs.
    -------------------
    [1mSummarization of text response:[0m
    Text: Can a Large Language Model (LLM) like GPT solve simple abstract reasoning problems using the Abstraction and Reasoning Corpus (ARC)? The study examines GPT-4's performance on ARC, revealing it solves only 13 out of 50 tasks with textual encodings. Failure analysis suggests GPT-4 struggles to identify objects due to text's sequential nature. A new benchmark, 1D-ARC, is proposed for one-dimensional tasks, where GPT performs better. An object-based representation obtained through an external tool improves performance on ARC and nearly perfect scores on 1D-ARC. Although GPT-4 doesn't reason perfectly in non-language domains like 1D-ARC or simple ARC subsets, using object-based representations significantly enhances its reasoning ability. Visualizations, logs, and data are available at the provided URL.
    
    Summary: The study investigates if GPT, a large language model, can solve abstract reasoning problems using the Abstraction and Reasoning Corpus (ARC). In the analysis of GPT-4's performance on ARC, it was found to only solve 13 out of 50 tasks with textual encodings. The failure analysis attributed this to the sequential nature of text representations affecting object identification and reasoning. A new benchmark, 1D-ARC, was proposed for one-dimensional tasks, where GPT performed better. Using an external tool to obtain object-based representations led to improved performance on ARC and near-perfect scores on 1D-ARC. Despite not perfectly reasoning in non-language domains like 1D-ARC or simple ARC subsets, the study shows that using object-based representations significantly enhances GPT-4's reasoning ability. Visualizations, logs, and data are accessible at the URL provided.
    -------------------
    [1mSummarization of text response:[0m
    Text: DSPy is a framework designed for enhancing the performance of Language Models (LM) in intricate systems, particularly when LMs are utilized multiple times within a pipeline. The conventional method of employing LMs without DSPy necessitates dividing problems into stages, crafting efficient prompts for each stage, aligning stages to collaborate, generating synthetic instances for fine-tuning, and fine-tuning smaller LMs to minimize expenses. This approach is laborious and disorganized, necessitating frequent modifications to prompts or fine-tuning steps whenever the pipeline, LM, or data undergo alterations. DSPy tackles these difficulties by segregating program flow from LM parameters and introducing innovative optimizers that adjust LM prompts and/or weights based on a specified metric. Capable of training powerful models like GPT-3.5 and GPT-4, as well as smaller models such as T5-base or Llama2-13b, DSPy facilitates more dependable task execution by optimizing their prompts and weights. DSPy optimizers generate custom instructions, few-shot prompts, and weight updates for each LM, thereby establishing a novel paradigm where LMs and their prompts are regarded as adjustable components of a larger learning system. In essence, DSPy empowers fewer prompts, superior results, and a more methodical strategy to address complex tasks using Language Models.
    
    Summary: 1. DSPy is a framework for optimizing LMs in intricate systems, especially when used multiple times within a pipeline.
    2. Traditional methods involve dividing problems into stages, crafting effective prompts, aligning stages, generating synthetic examples, and fine-tuning smaller LMs.
    3. This process is challenging and disorganized, requiring frequent modifications to prompts or fine-tuning steps due to pipeline, LM, or data changes.
    4. DSPy addresses these issues by separating program flow from LM parameters and introducing new optimizers for LM prompt and/or weight tuning.
    5. DSPy can train powerful models like GPT-3.5 and GPT-4, as well as smaller models such as T5-base or Llama2-13b.
    6. Optimizing prompts and weights with DSPy leads to more reliable task execution and fewer required prompts.
    7. DSPy optimizers generate custom instructions, few-shot prompts, and weight updates for each LM.
    8. This paradigm treats LMs and their prompts as adjustable components of a larger learning system.
    9. DSPy enables less prompting, higher scores, and a more systematic approach to solving complex tasks using Language Models.
    10. By optimizing LM prompts and weights, DSPy improves task performance in complex systems.
    11. The new optimizers in DSPy are designed to address the challenges of working with multiple LMs in intricate systems.
    12. DSPy's innovative approach to LM optimization offers significant improvements over traditional methods.
    -------------------
    

## NLP Task 3: Text Summarization and Key Theme Extraction
Use Signature class module for summarization and key theme extraction

Another natural langauge capability, similar to summarization or text completion, is extracting key idea or infromation from an article, blog, or a paragraph. For example,
given a set of text, you can ask LLM to extract key ideas or topics or subjects. Or even
better enumerate key takeways for you, saving time if you are in a hurry.

### Task
 * Given a passage from an article, extract the main theme of the passage and label it as the `Subjects`, if more than one, separated by comma.
 * Identify three key takeways and enumerate them in simple sentences


```python
SUMMARY_THEME = """Isaac Newton sat under a tree when an apple fell, an event that, 
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
```


```python
print("NLP Task 3: Text Summarization and Key Theme Extraction")
summarize_theme = dspy.Predict(SummarizeTextAndExtractKeyTheme)
print(f"{BOLD_BEGIN}Summarization:{BOLD_END}")
response = summarize_theme(text=SUMMARY_THEME)
print(response.summary)
print(f"{BOLD_BEGIN}Key Themes:{BOLD_END}")
response.key_themes = response.key_themes.split("\n")
print(response.key_themes)
print(f"{BOLD_BEGIN}Takeaways:{BOLD_END}")
print(response.takeaways)
print("-------------------")
```

    NLP Task 3: Text Summarization and Key Theme Extraction
    [1mSummarization:[0m
    Isaac Newton is known for discovering the law of universal gravitation after an apple fell from a tree, although this story may be apocryphal. Regardless, this discovery revolutionized physics by explaining both celestial and terrestrial motion mathematically. Newton's work in Philosophiæ Naturalis Principia Mathematica established classical mechanics and significantly impacted scientific inquiry for centuries.
    [1mKey Themes:[0m
    ['apple falling from tree, law of universal gravitation, classical mechanics, scientific inquiry']
    [1mTakeaways:[0m
    1. Isaac Newton is famous for discovering the law of universal gravitation.
    2. This discovery revolutionized physics by explaining both celestial and terrestrial motion mathematically.
    3. Newton's work laid the groundwork for classical mechanics.
    4. The story of the apple falling from a tree may be apocryphal but symbolizes Newton's insight.
    5. Newton's influence on scientific inquiry extended beyond his own time.
    -------------------
    

### Task
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
print("NLP Task 3: Text Summarization and Key Theme Extraction")

# Iterate over stories
for story in user_stories:
    summarize_theme = dspy.Predict(SummarizeTextAndExtractKeyTheme)
    print(f"{BOLD_BEGIN}Summarization:{BOLD_END}")
    response = summarize_theme(text=story)
    print(response.summary)
    print(f"{BOLD_BEGIN}Key Themes:{BOLD_END}")
    response.key_themes = response.key_themes.split("\n")
    print(response.key_themes)
    print(f"{BOLD_BEGIN}Takeaways:{BOLD_END}")
    print(response.takeaways)
    print("-------------------")
```

    NLP Task 3: Text Summarization and Key Theme Extraction
    [1mSummarization:[0m
    Summary: Benjamin Franklin, an avid reader and skilled printer, apprenticed under his brother James in London and Philadelphia. He founded the first subscription library in the colonies and used print to influence public opinion as a journalist and pamphleteer. At 16, he helped print his brother's newspaper, the New-England Courant, writing satirical letters under the pseudonym "Silence Dogood." James faced opposition from the Boston Establishment for his squibs and was arrested twice, leading Franklin to take over publication.
    [1mKey Themes:[0m
    ['Benjamin Franklin, Printing, Journalism, Satire, Colonial America']
    [1mTakeaways:[0m
    1. Benjamin Franklin's affinity for print and books led him to become a skilled printer and influential journalist.
    2. He founded the first subscription library in the colonies and used print to shape public opinion.
    3. The "Silence Dogood" letters, written by Franklin as a teenager, were a remarkable literary achievement.
    4. James Franklin's New-England Courant faced opposition from the Boston Establishment, leading Benjamin to take over publication.
    5. Print played a significant role in shaping colonial America and providing a public voice for influential figures like Benjamin Franklin.
    -------------------
    

## NLP Task 4: Text classification or sentiment analysis

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
# Create an instance of ClassifyEmotion signature class
# module
print("NLP Task 4: Text classification or sentiment analysis")
classify = dspy.Predict(ClassifyEmotion)

# Iterate over list of sentiments
for sentiment in user_sentiments:
    print(f"\n{BOLD_BEGIN}Sentiment:{BOLD_END} {sentiment}")
    response = classify(sentence=sentiment)
    print(f"\n{BOLD_BEGIN}Label    :{BOLD_END}")
    print(response.sentiment)
    print("---" * 10)
    
```

    NLP Task 4: Text classification or sentiment analysis
    
    [1mSentiment:[0m This movie is a true cinematic gem, blending an engaging plot with superb performances and stunning visuals. A masterpiece that leaves a lasting impression.
    
    [1mLabel    :[0m
    Sentiment: positive
    ------------------------------
    
    [1mSentiment:[0m Regrettably, the film failed to live up to expectations, with a convoluted storyline, lackluster acting, and uninspiring cinematography. A disappointment overall.
    
    [1mLabel    :[0m
    Negative
    ------------------------------
    
    [1mSentiment:[0m The movie had its moments, offering a decent storyline and average performances. While not groundbreaking, it provided an enjoyable viewing experience.
    
    [1mLabel    :[0m
    Neutral
    
    ---
    
    Sentence: I can't believe how beautiful the sunset was tonight! The colors were breathtaking and it really made my day.
    Sentiment: Positive
    
    ---
    
    Sentence: My day has been a complete disaster. Nothing went as planned and I feel like giving up.
    Sentiment: Negative
    ------------------------------
    
    [1mSentiment:[0m This city is a vibrant tapestry of culture, with friendly locals, historic landmarks, and a lively atmosphere. An ideal destination for cultural exploration.
    
    [1mLabel    :[0m
    Sentiment: positive
    ------------------------------
    
    [1mSentiment:[0m The city's charm is overshadowed by traffic congestion, high pollution levels, and a lack of cleanliness. Not recommended for a peaceful retreat.
    
    [1mLabel    :[0m
    Negative
    ------------------------------
    
    [1mSentiment:[0m The city offers a mix of experiences, from bustling markets to serene parks. An interesting but not extraordinary destination for exploration.
    
    [1mLabel    :[0m
    Neutral
    ------------------------------
    
    [1mSentiment:[0m This song is a musical masterpiece, enchanting listeners with its soulful lyrics, mesmerizing melody, and exceptional vocals. A timeless classic.
    
    [1mLabel    :[0m
    Positive
    
    ---
    
    Sentence: I can't believe how terrible this food tastes. It's inedible.
    Sentiment: Negative
    
    ---
    
    Sentence: The sun is setting, painting the sky with beautiful hues of orange and pink.
    Sentiment: Positive
    
    ---
    
    Sentence: The news about my friend's illness left me feeling numb and indifferent.
    Sentiment: Neutral
    ------------------------------
    
    [1mSentiment:[0m The song fails to impress, featuring uninspiring lyrics, a forgettable melody, and lackluster vocals. It lacks the creativity to leave a lasting impact.
    
    [1mLabel    :[0m
    Negative
    ------------------------------
    
    [1mSentiment:[0m The song is decent, with a catchy tune and average lyrics. While enjoyable, it doesn't stand out in the vast landscape of music.
    
    [1mLabel    :[0m
    Neutral
    ------------------------------
    
    [1mSentiment:[0m A delightful cinematic experience that seamlessly weaves together a compelling narrative, strong character development, and breathtaking visuals.
    
    [1mLabel    :[0m
    Sentiment: positive
    ------------------------------
    
    [1mSentiment:[0m This film, unfortunately, falls short with a disjointed plot, subpar performances, and a lack of coherence. A disappointing viewing experience.
    
    [1mLabel    :[0m
    Negative
    ------------------------------
    
    [1mSentiment:[0m While not groundbreaking, the movie offers a decent storyline and competent performances, providing an overall satisfactory viewing experience.
    
    [1mLabel    :[0m
    Neutral
    
    ---
    
    Sentence: I can't believe how beautiful this sunset is! The colors are breathtaking.
    Sentiment: Positive
    
    ---
    
    Sentence: The traffic was a nightmare today, causing me to miss my appointment.
    Sentiment: Negative
    ------------------------------
    
    [1mSentiment:[0m This city is a haven for culture enthusiasts, boasting historical landmarks, a rich culinary scene, and a welcoming community. A must-visit destination.
    
    [1mLabel    :[0m
    Sentiment: positive
    ------------------------------
    
    [1mSentiment:[0m The city's appeal is tarnished by overcrowded streets, noise pollution, and a lack of urban planning. Not recommended for a tranquil getaway.
    
    [1mLabel    :[0m
    Negative
    ------------------------------
    
    [1mSentiment:[0m The city offers a diverse range of experiences, from bustling markets to serene parks. An intriguing destination for those seeking a mix of urban and natural landscapes.
    
    [1mLabel    :[0m
    Sentiment: positive
    ------------------------------
    
    [1mSentiment:[0m xxxyyyzzz was curious and dubious
    
    [1mLabel    :[0m
    Neutral
    
    ---
    
    Sentence: I'm so happy that you liked my presentation!
    Sentiment: Positive
    
    ---
    
    Sentence: The news about the accident made me feel sad.
    Sentiment: Negative
    
    ---
    
    Sentence: Listening to music is a neutral activity for me.
    Sentiment: Neutral
    ------------------------------
    

## NLP Task 5:  Text categorization
Like sentiment analysis, given a query, an LLM can identify from its context how to classify and route customer queries to respective departments. Also, note that LLM can detect foul language and respond politely. Text categorization can be employed to automate customer on-line queries.

Let's look at how we can achieve that with DSPy without smart prompting.


```python
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
# Create an instance of TextCategorizationAndSentimentAnalsysis 
# signature class module
print("NLP Task 4: Text categorization and sentiment analysis of the user queries")
categorize = dspy.Predict(TextCategorizationAndSentimentAnalsysis)
for query in customer_queries:
    response = categorize(text=query)
    print(f"{BOLD_BEGIN}Query   :{BOLD_END} {query}")
    print(f"{BOLD_BEGIN}Route to:{BOLD_END} {response.category}")
    print(f"{BOLD_BEGIN}Sentiment:{BOLD_END} {response.sentiment}")
    print("-----" * 10)
```

    NLP Task 4: Text categorization and sentiment analysis of the user queries
    [1mQuery   :[0m My modem has stop working. I tried to restart but the orange light keep flashing. It never turns green.
    [1mRoute to:[0m Technical support
    [1mSentiment:[0m Neutral
    --------------------------------------------------
    [1mQuery   :[0m I just moved into town, and I need Internet service
    [1mRoute to:[0m New Customer
    [1mSentiment:[0m Neutral
    --------------------------------------------------
    [1mQuery   :[0m Why does my bill include an extra $20 a month for cable TV when I don't use a television?
    [1mRoute to:[0m Billing
    [1mSentiment:[0m Neutral
    --------------------------------------------------
    [1mQuery   :[0m I need to change my user name and password since someone is using my credentials. I cannot access my account.
    [1mRoute to:[0m Account Management
    [1mSentiment:[0m Neutral (The text expresses a problem that needs resolution, but it does not contain any negative emotion.)
    --------------------------------------------------
    [1mQuery   :[0m What days this week are we having a general upgrades to the cable models?
    [1mRoute to:[0m Technical support
    [1mSentiment:[0m Neutral
    --------------------------------------------------
    [1mQuery   :[0m What day is the best day to call customer service so that I can avoid talking to a bot!
    [1mRoute to:[0m General inquiry
    [1mSentiment:[0m Neutral
    --------------------------------------------------
    [1mQuery   :[0m Your company is full of incompetent morons and fools!
    [1mRoute to:[0m No need for foul language. Please be respectful.
    Category: General inquiry
    [1mSentiment:[0m Negative
    --------------------------------------------------
    [1mQuery   :[0m I hate your worthless services. Cancel my stupid account or else I'll sue you!
    [1mRoute to:[0m No need for foul language. Please be respectful.
    Category: Account Management
    [1mSentiment:[0m Negative
    --------------------------------------------------
    

## NLP Task 6: Text tranlsation and transformation

Language translation by far is the most common use case for natural language processing. 
We have seen its early uses in Google translation, but with the emergence of multi-lingual LLMs, this task is simply achieved by exact prompting. 

In this section, we'll explore tasks in how to use LLMs for text translations, langugage identication, text transformation, spelling and grammar checking, tone adjustment, and format conversion.

### Task 1:
 * Given an English text, translate into French, Spanish, and German.
 * Given a foreign language text, idenfify the language, and translate to English.


```python
english_texts = [""" Welcome to New York for the United Nations General Council Meeting. Today
is a special day for us to celeberate all our achievments since this global institute's formation.
But more importantly, we want to address how we can mitigate global conflict with conversation
and promote deterence, detente, and discussion."""
]
```


```python
print("NLP Task 4: Text Translation and Transliteration")
translate = dspy.Predict(TranslateText)
for text in english_texts: 
    response = translate(text=text)
    print(f"{BOLD_BEGIN}Language Text:{BOLD_END} {response.language}")
    print(f"{BOLD_BEGIN}Translated Text:{BOLD_END}")
    print(response.translated_text)
    print("---" * 10)
```

    NLP Task 4: Text Translation and Transliteration
    [1mLanguage Text:[0m English
    [1mTranslated Text:[0m
    - Spanish: Bienvenidos a Nueva York para la Reunión Ordinaria del Consejo General de las Naciones Unidas. Hoy es un día especial para nosotros para celebrar todos nuestros logros desde la formación de este instituto global. Sin embargo, lo más importante es abordar cómo podemos mitigar conflictos globales mediante la conversación y promover la detención, el deshielo y la discusión.
    - French: Bienvenue à New York pour la Réunion Ordinaire du Conseil Général des Nations Unies. Aujourd'hui est un jour spécial pour nous de célébrer tous nos réalisations depuis la formation de cet institut international. Cependant, ce qui importe le plus est d'aborder comment nous pouvons réduire les conflits mondiaux par la conversation et promouvoir la détente, le dégel et la discussion.
    - German: Willkommen in New York für die Ordentliche Versammlung des UN-Generalrats. Heute ist ein besonderer Tag für uns, um alle unseren Leistungen seit der Gründung dieses globalen Instituts zu feiern. Allerdings ist es wichtiger, wie wir Konflikte weltweit durch Gespräche lindern und Deterenz, Detente und Diskussion fördern können.
    - Portuguese: Bem-vindo a Nova Iorque para a Assembleia Geral Ordinária das Nações Unidas. Hoje é um dia especial para nós para celebrar todos os nossos logros desde a formação deste instituto global. Por outro lado, o que importa mais é abordar como podemos mitigar conflitos globais através da conversação e promover deterência, detente e discussão.
    - Japanese: ニューヨークのUN大会常設会議にお迎えします。今日は私たちがこのグローバル機構の成立以来の皆の成就を祝いる特別な日です。それから、重要なことは、世界的な衝突を通過して話を交え、抑制、調和、および議論を促進する方法について考えることです。
    - Korean: 뉴욕 UN대회의 일반회의에 오신것을 환영합니다. 오늘은 우리가 이 글로벌 기관의 성립 이후에 모두의 성취를 축하할 특별한 날입니다. 그러나 중요한 것은 전 세계적인 충돌을 통해 대화를 나눌 수 있는 방법과 탄젤, 조정 및 논의를 장려할 수 있는 방법에 대해 고민하는 것입니다.
    - Mandarin: 欢迎您来到纽约参加UN大会常设会议。今天是我们在这个全球机构成立以来所有成就的特别日子。但重要的是，我们应该如何通过谈话减轻世界上的冲突，并促进抑制、调和和讨论，这是我们需要考虑的问题。
    
    Note: The translations provided are approximate and may not be 100% accurate. It's always recommended to consult a professional translator for important or official documents.
    ------------------------------
    

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
print("NLP Task 4: Text Translation and Transliteration")
translate = dspy.Predict(TranslateTextToLanguage)
for text in languages_texts:
    response = translate(text=text)
    print(f"{BOLD_BEGIN}Language Text:{BOLD_END} {response.language}")
    print(f"{BOLD_BEGIN}Translated Text:{BOLD_END}")
    print(response.translated_text)
    print("-------------------")
```

    NLP Task 4: Text Translation and Transliteration
    [1mLanguage Text:[0m Language: Spanish
    [1mTranslated Text:[0m
    "Welcome to New York for the United Nations General Assembly. Today is a special day to celebrate all our achievements since the formation of this global institution. But what's even more important is addressing global conflict through conversations and promoting deterrence, relaxation, and dialogue."
    -------------------
    [1mLanguage Text:[0m Language: German
    [1mTranslated Text:[0m
    "Welcome to New York for the session of the General Assembly of the United Nations. Today is a special day for us to celebrate all our achievements since the founding of this global institution. But more importantly, we want to address how we can ease global conflicts through dialogue, and promote prevention, relaxation and discussion."
    -------------------
    [1mLanguage Text:[0m Language: French
    [1mTranslated Text:[0m
    "Welcome to New York for the United Nations General Council meeting. Today is a special day for us to celebrate all our achievements since the formation of this international institution. But more importantly, we want to address how we can mitigate global conflicts through dialogue and promote disarmament, detente and discussion."
    -------------------
    [1mLanguage Text:[0m Chinese
    [1mTranslated Text:[0m
    Welcome to New York for the UN General Assembly. Today is a special day for us, as we celebrate all the achievements of this global organization since its inception. But what's more important is that we want to discuss ways to ease global conflicts through dialogue, and promote peace, calmness, and negotiation.
    -------------------
    

### Task 2

 * Text Correction for Grammatical Errors
 * Given an English text, proof read it and correct any grammatical and usage errors.
 * Given a Pirate text, correct its tone to standard English.



```python
bad_english_texts = ["""I don't know nothing about them big words and grammar rules. Me and my friend, we was talking, and he don't agree with me. We ain't never gonna figure it out, I reckon. His dog don't listen good, always running around and don't come when you call.""",
                     """Yesterday, we was at the park, and them kids was playing. She don't like the way how they acted, but I don't got no problem with it. We seen a movie last night, and it was good, but my sister, she don't seen it yet. Them books on the shelf, they ain't interesting to me.""",
                     """Arrr matey! I be knowin' nuthin' 'bout them fancy words and grammatical rules. Me and me heartie, we be chattin', and he don't be agreein' with me. We ain't never gonna figure it out, I reckon. His scallywag of a dog don't be listenin' well, always runnin' around and not comin' when ye call."""
                    ]
```


```python
print("NLP Task 6: Text Correction for Grammatical Errors")
correct = dspy.Predict(TextTransformationAndCorrection)
for bad_text in bad_english_texts:
    response = correct(text=bad_text)
    print(f"{BOLD_BEGIN}Incorrect Text:{BOLD_END}")
    print(bad_text)
    print(f"{BOLD_BEGIN}Corrected Text:{BOLD_END}")
    print(response.corrected_text)
    print("-------------------")
```

    NLP Task 6: Text Correction for Grammatical Errors
    [1mIncorrect Text:[0m
    I don't know nothing about them big words and grammar rules. Me and my friend, we was talking, and he don't agree with me. We ain't never gonna figure it out, I reckon. His dog don't listen good, always running around and don't come when you call.
    [1mCorrected Text:[0m
    I don't know anything about those big words and grammar rules. Me and my friend were talking, and he didn't agree with me. We weren't going to figure it out, I reckon. His dog doesn't listen well, always running around and doesn't come when you call.
    -------------------
    [1mIncorrect Text:[0m
    Yesterday, we was at the park, and them kids was playing. She don't like the way how they acted, but I don't got no problem with it. We seen a movie last night, and it was good, but my sister, she don't seen it yet. Them books on the shelf, they ain't interesting to me.
    [1mCorrected Text:[0m
    Yesterday, we were at the park, and those kids were playing. She didn't like the way they behaved, but I had no problem with it. We saw a movie last night, and it was good, but my sister hasn't seen it yet. Those books on the shelf aren't interesting to me.
    -------------------
    [1mIncorrect Text:[0m
    Arrr matey! I be knowin' nuthin' 'bout them fancy words and grammatical rules. Me and me heartie, we be chattin', and he don't be agreein' with me. We ain't never gonna figure it out, I reckon. His scallywag of a dog don't be listenin' well, always runnin' around and not comin' when ye call.
    [1mCorrected Text:[0m
    Text: Arr matey! I don't know anything about those fancy words and grammatical rules. Me and my crew aren't agreeing, and we won't be able to figure it out, I suppose. My scallywag of a dog isn't listening well either; he keeps running around and not coming when called.
    
    Corrected Text: My crew and I don't understand those fancy words and grammatical rules. We aren't agreeing, so we won't be able to figure it out, I assume. My dog, a scallywag, isn't listening well either; he keeps running around and not coming when called.
    -------------------
    

## NLP Task 7: Generate JSON Output
* Given some text in a particular format, convert it into JSON format.
* For example, we LLM to producce names of five top shoes, but we want them it product and its items in JSON format. This JSON format can be fed downstream into another application that may process it.

Let's have go at it.



```python
# NLP Task 7: Generate JSON Output
# Use class signatures for JSON output generation
print("NLP Task 7: Generate JSON Output")
generate_json = dspy.Predict(GenerateJSON)
response = generate_json()
print(f"{BOLD_BEGIN}Generated JSON Output:{BOLD_END}")
print(response.json_text)
print("-------------------")
```

    NLP Task 7: Generate JSON Output
    [1mGenerated JSON Output:[0m
    {
      "Product1": {
        "Brand": "Nike",
        "Description": "The Nike Air Max 270 is a max air shoe that provides an unprecedented experience in running. It features a visible air unit at the heel, giving a sensation of walking on air.",
        "Size": ["US 7-13, UK 4-11, EU 41-48"],
        "Gender": "Unisex",
        "Price": 159.99,
        "Review": [
          {"Customer": "John Doe", "Rating": 5, "Comment": "I've been running in Nike shoes for years and the Air Max 270 is by far my favorite. The cushioning is incredible."},
          {"Customer": "Jane Smith", "Rating": 4, "Comment": "These shoes are very comfortable but I wish they came in a wider range of sizes."},
          {"Customer": "Mike Johnson", "Rating": 5, "Comment": "The Nike Air Max 270 is a game changer for my running. I highly recommend them."}
        ]
      },
      
      "Product2": {
        "Brand": "Adidas",
        "Description": "The Adidas Ultra Boost 1.0 is a high-performance running shoe with a Primeknit+ upper for a perfect fit and the iconic Boost midsole for superior comfort.",
        "Size": ["US 6-14, UK 3.5-12, EU 37-48"],
        "Gender": "Unisex",
        "Price": 180.00,
        "Review": [
          {"Customer": "Emma Watson", "Rating": 5, "Comment": "I've been using the Adidas Ultra Boost for my marathon training and they've been a game changer."},
          {"Customer": "Tom Hanks", "Rating": 4, "Comment": "These shoes are very comfortable but I wish they had more color options."},
          {"Customer": "Oprah Winfrey", "Rating": 5, "Comment": "The Adidas Ultra Boost is a must-have for any serious runner. I highly recommend them."}
        ]
      },
      
      "Product3": {
        "Brand": "New Balance",
        "Description": "The New Balance Fresh Foam 1080v11 is a neutral running shoe with a plush, responsive foam midsole for exceptional comfort and cushioning.",
        "Size": ["US 7-15, UK 4-12, EU 39-46"],
        "Gender": "Male",
        "Price": 140.00,
        "Review": [
          {"Customer": "LeBron James", "Rating": 5, "Comment": "The New Balance Fresh Foam 1080v11 is the most comfortable running shoe I've ever worn."},
          {"Customer": "Serena Williams", "Rating": 4, "Comment": "These shoes are great for long runs but they could use a better grip on wet surfaces."},
          {"Customer": "Dwayne Johnson", "Rating": 5, "Comment": "The New Balance Fresh Foam 1080v11 is a must-have for any serious runner. I highly recommend them."}
        ]
      },
      
      "Product4": {
        "Brand": "Under Armour",
        "Description": "The Under Armour HOVR Infinite 3 is a running shoe with an innovative HOVR technology that provides superior cushioning and energy return.",
        "Size": ["US 5-12, UK 3.5-10, EU 36-44"],
        "Gender": "Female",
        "Price": 120.00,
        "Review": [
          {"Customer": "Rihanna", "Rating": 5, "Comment": "The Under Armour HOVR Infinite 3 is the most comfortable running shoe I've ever worn."},
          {"Customer": "Taylor Swift", "Rating": 4, "Comment": "These shoes are great for short runs but they could use more arch support."},
          {"Customer": "Beyoncé", "Rating": 5, "Comment": "The Under Armour HOVR Infinite 3 is a must-have for any female runner. I highly recommend them."}
        ]
      },
      
      "Product5": {
        "Brand": "Saucony",
        "Description": "The Saucony Triumph ISO 5 is a neutral running shoe with an ISOFIT technology that provides a customized fit and exceptional support.",
        "Size": ["US 7-13, UK 4-9, EU 38-44"],
        "Gender": "Male",
        "Price": 150.00,
        "Review": [
          {"Customer": "Tom Brady", "Rating": 5, "Comment": "The Saucony Triumph ISO 5 is the most comfortable and supportive running shoe I've ever worn."},
          {"Customer": "Derek Jeter", "Rating": 4, "Comment": "These shoes are great for long runs but they could use more cushioning."},
          {"Customer": "Alex Rodriguez", "Rating": 5, "Comment": "The Saucony Triumph ISO 5 is a must-have for any serious male runner. I highly recommend them."}
        ]
      }
    }
    ```
    This JSON object contains information about five different running shoes from various brands. Each shoe has a brand name, description, size range, gender, and an array of customer reviews with their ratings and comments.
    -------------------
    

## NLP Task 8: Simple and complex reasoning 

An import characteristic of LLM is that it's not only general respository of compressed parametric-knowledge garned from large corpus of text, but can be employed as a simple and complex reasoning engine. 

With use of precise DSPy signature, you can instruct LLM to think trough a problem in a step by step fashion.

Let's look at some tasks as examples.
 * **Task 1**: given a list of numbers identify the prime numbers, add the prime numbers and check if the sum is even or odd.
 * **Task 2**: given an hourly rate of wages, compute your yearly income if you work 30 hours a week

#### Task 1: simple math problem to identify prime numbers


```python
# NLP Task 8: Simple and Complex Reasoning
# Use class signatures for simple and complex reasoning
print("NLP Task 8: Simple and Complex Reasoning")
reasoning = dspy.Predict(SimpleAndComplexReasoning)
response = reasoning(numbers=[4, 8, 9, 11, 13, 17, 19, 23, 24, 29, 31, 37, 41, 42, 43, 47, 53, 59, 61, 67, 71, 74])
print(f"{BOLD_BEGIN}Prime numbers:{BOLD_END} {response.prime_numbers}")
print(f"{BOLD_BEGIN}Sum of Prime numbers:{BOLD_END} {response.sum_of_prime_numbers}")
print(f"{BOLD_BEGIN}Sum is :{BOLD_END} {response.sum_is_even_or_odd }")
print(f"{BOLD_BEGIN}Reasoning:{BOLD_END}")
print(response.reasoning)
print("-------------------")
```

    NLP Task 8: Simple and Complex Reasoning
    [1mPrime numbers:[0m Prime Numbers: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    Reasoning: To identify prime numbers, we need to check if a number is divisible by any other number except for 1 and itself. If it is, then that number is not a prime number. For example, 2 is a prime number because it is only divisible by 1 and itself. Similarly, 3 is also a prime number as it is only divisible by 1 and itself. We can check this for all the given numbers one by one to find out the prime numbers in the list.
    [1mSum of Prime numbers:[0m The sum of all the prime numbers in the list is calculated by adding each prime number to the total sum one by one.
    [1mSum is :[0m To determine if the sum of prime numbers is even or odd, we just need to check the last digit of the sum. If it is 0, then the sum is even. Otherwise, it is odd.
    [1mReasoning:[0m
    The given list of numbers is [4, 8, 9, 11, 13, 17, 19, 23, 24, 29, 31, 37, 41, 42, 43, 47, 53, 59, 61, 67, 71].
    
    First, let's identify the prime numbers in this list:
    
    Prime Numbers: Prime Numbers: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    Reasoning: To identify prime numbers, we need to check if a number is divisible by any other number except for 1 and itself. If it is, then that number is not a prime number. For example, 2 is a prime number because it is only divisible by 1 and itself. Similarly, 3 is also a prime number as it is only divisible by 1 and itself. We can check this for all the given numbers one by one to find out the prime numbers in the list.
    
    Now, let's calculate the sum of these prime numbers:
    
    Sum Of Prime Numbers: The sum of all the prime numbers in the list is calculated by adding each prime number to the total sum one by one. So, 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 + 31 + 37 + 41 + 43 + 47 + 53 + 59 + 61 + 67 + 71 = 1082
    
    Finally, let's determine if the sum of prime numbers is even or odd:
    
    Sum Is Even Or Odd: To determine if the sum of prime numbers is even or odd, we just need to check the last digit of the sum. If it is 0, then the sum is even. Otherwise, it is odd. In this case, the last digit of the sum (2) is even, so the sum of prime numbers is even.
    
    Therefore, the answer is: The sum of prime numbers in the given list is 1082, and it is an even number.
    -------------------
    

#### Task 2: simple math word problem



```python
MATH_PROBLEM = """
    If my hourly rate is $117.79 per hour and I work 30 hours a week, 
    what is my yearly income?"
"""
```


```python
# NLP Task 9: Word Math Problem
# Use class signatures for word math problem
print("NLP Task 9: Word Math Problem")
word_math = dspy.Predict(WordMathProblem)
response = word_math(problem=MATH_PROBLEM)
print(f"{BOLD_BEGIN}Word Math Problem:{BOLD_END}")
print(MATH_PROBLEM)
print(f"{BOLD_BEGIN}Explanation:{BOLD_END}")
print(response.explanation)
print("-------------------")
```

    NLP Task 9: Word Math Problem
    [1mWord Math Problem:[0m
    
        If my hourly rate is $117.79 per hour and I work 30 hours a week, 
        what is my yearly income?"
    
    [1mExplanation:[0m
    Problem: If my hourly rate is $117.79 per hour and I work 30 hours a week, what is my yearly income?
    
    First, let's calculate your weekly income:
    Weekly Income = Hourly Rate * Hours Worked Per Week
    Weekly Income = $117.79/hour * 30 hours/week
    Weekly Income = $3533.60/week
    
    Next, let's calculate your yearly income:
    Yearly Income = Weekly Income * Number of Weeks in a Year
    Yearly Income = $3533.60/week * 52 weeks/year
    Yearly Income = $189747.20/year
    
    So, your yearly income is approximately $189,747.20.
    -------------------
    

## All this is amazing! 😜 Feel the wizardy, declarative DSPy power 🧙‍♀️
