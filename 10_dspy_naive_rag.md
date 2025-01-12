<img src="images/dspy_img.png" height="35%" width="%65">

## Naive Retreival Augmented Generation (RAG)

Amazingly easy and modular, DSPy Modules can be chained or stacked to create 
a pipeline. In our case, building a Naive RAG comprises using `dspy.Signature` and `dspy.ChainOfThought`, and own module class `RAG` (see implementation in [dspy_utils](dspy_utils.py). 

Out of the box, DSPy supports a set of [Retrievers clients](https://dspy-docs.vercel.app/api/category/retrieval-model-clients). For this example,
we will use `dspy.ColBERTv2`.

<img src="images/dspy_rag_pipeline.png">
<img src="images/dspy_rag_flow.png">

[source](https://towardsdatascience.com/intro-to-dspy-goodbye-prompting-hello-programming-4ca1c6ce3eb9)


```python
import warnings
import dspy
import warnings
from dspy_utils import RAG, BOLD_BEGIN, BOLD_END

# Filter out warnings
warnings.filterwarnings("ignore")
```

### Questions to ask the RAG model


```python

QUESTIONS = [
    "What is the capital of Tanzania?",
    "Who was the president of the United States in 1960?",
    "What is the largest mammal?",
    "What is the most populous country?",
    "What is the most widely spoken language?",
    "Which country won the FIFA Football World Cup in 1970?",
    "Which country has won the most FIFA Football World Cups?",
    "Who is the author of the book '1984'?",
    "What is the most popular programming language?",
    "What is dark matter in physics?",
]
```

### Instantiate our Language Model


```python
# Setup OLlama environment on the local machine
ollama_mistral = dspy.OllamaLocal(model='mistral',
                                      max_tokens=2500)
# Instantiate the ColBERTv2 as Retrieval module
colbert_rm = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# Configure the settings
dspy.settings.configure(lm=ollama_mistral, rm=colbert_rm)
```

### Query the RAG 


```python
# Instantiate the RAG module
rag = RAG(num_passages=5)
for idx, question in enumerate(QUESTIONS):
    print(f"{BOLD_BEGIN}Question {idx + 1}: {BOLD_END}{question}")
    response = rag(question=question)
    print(f"{BOLD_BEGIN}Answer    : {BOLD_END}{response.answer}")
    print("-----------------------------\n")
```

    [1mQuestion 1: [0mWhat is the capital of Tanzania?
    [1mAnswer    : [0mDodoma
    -----------------------------
    
    [1mQuestion 2: [0mWho was the president of the United States in 1960?
    [1mAnswer    : [0mThe president of the United States in 1960 was John F. Kennedy.
    -----------------------------
    
    [1mQuestion 3: [0mWhat is the largest mammal?
    [1mAnswer    : [0mThe blue whale is the largest mammal.
    -----------------------------
    
    [1mQuestion 4: [0mWhat is the most populous country?
    [1mAnswer    : [0mChina is the most populous country.
    -----------------------------
    
    [1mQuestion 5: [0mWhat is the most widely spoken language?
    [1mAnswer    : [0mThe most widely spoken languages, depending on the region or country, are English, Japanese, and Turkish.
    -----------------------------
    
    [1mQuestion 6: [0mWhich country won the FIFA Football World Cup in 1970?
    [1mAnswer    : [0mBrazil
    -----------------------------
    
    [1mQuestion 7: [0mWhich country has won the most FIFA Football World Cups?
    [1mAnswer    : [0mBoth Germany and Italy have each won 4 FIFA World Cup championships.
    -----------------------------
    
    [1mQuestion 8: [0mWho is the author of the book '1984'?
    [1mAnswer    : [0mGeorge Orwell
    -----------------------------
    
    [1mQuestion 9: [0mWhat is the most popular programming language?
    [1mAnswer    : [0mJava is one of the most popular programming languages in use, particularly for client-server web applications, with a reported 9 million developers.
    -----------------------------
    
    [1mQuestion 10: [0mWhat is dark matter in physics?
    [1mAnswer    : [0mDark matter is a hypothetical type of matter that has never been directly observed but is inferred from its gravitational effects on visible matter. It does not interact with or emit electromagnetic radiation, making it invisible to the entire electromagnetic spectrum. Cold dark matter is a specific form of dark matter believed to make up approximately 84.54% of the matter in the universe and interacts very weakly with ordinary matter and electromagnetic radiation.
    -----------------------------
    
    

### Inspect history of the prompts


```python
print(ollama_mistral.inspect_history(n=1))
```

    
    
    
    Given a context, question, answer the question.
    
    ---
    
    Follow the following format.
    
    Context: ${context}
    
    Question: ${question}
    
    Reasoning: Let's think step by step in order to ${produce the answer}. We ...
    
    Answer: ${answer}
    
    ---
    
    Context:
    [1] «Dark matter in fiction | Dark matter is defined as hypothetical matter that is undetectable by its emitted radiation, but whose presence can be inferred from gravitational effects on visible matter. It has been used in a variety of fictional media, including computer and video games and books. In such cases, dark matter is usually attributed extraordinary physical or magical properties. Such descriptions are often inconsistent with the known properties of dark matter proposed in physics and cosmology. For example in computer games, it is often used as material for making weapons and items, and is usually depicted as black or a similar color.»
    [2] «Dark Matter (disambiguation) | Dark matter is matter that is undetectable by its emitted radiation, but whose presence can be inferred from gravitational effects.»
    [3] «Dark matter | Dark matter is a hypothetical type of matter distinct from baryonic matter (ordinary matter such as protons and neutrons), neutrinos and dark energy. Dark matter has never been directly observed; however, its existence would explain a number of otherwise puzzling astronomical observations. The name refers to the fact that it does not emit or interact with electromagnetic radiation, such as light, and is thus invisible to the entire electromagnetic spectrum. Although dark matter has not been directly observed, its existence and properties are inferred from its gravitational effects such as the motions of visible matter, gravitational lensing, its influence on the universe's large-scale structure, on galaxies, and its effects on the cosmic microwave background.»
    [4] «Cold dark matter | In cosmology and physics, cold dark matter (CDM) is a hypothetical form of dark matter whose particles moved slowly compared to the speed of light (the "cold" in CDM) since the universe was approximately one year old (a time when the cosmic particle horizon contained the mass of one typical galaxy); and interact very weakly with ordinary matter and electromagnetic radiation (the "dark" in CDM). It is believed that approximately 84.54% of matter in the Universe is dark matter, with only a small fraction being the ordinary baryonic matter that composes stars, planets and living organisms.»
    [5] «Dark•Matter | Dark•Matter is a science fiction/conspiracy theory campaign setting that was originally published in 1999 by Wizards of the Coast as the second campaign setting for the "Alternity" role-playing game. It was written by Wolfgang Baur and Monte Cook. It was later converted to "d20 Modern" rules and published as a stand-alone book in 2006.»
    
    Question: What is dark matter in physics?
    
    Reasoning: Let's think step by step in order to[32m answer the question. We will first examine the context provided to understand what dark matter is in the context of physics. The context states that dark matter is a hypothetical type of matter distinct from baryonic matter, neutrinos and dark energy. It has never been directly observed but its existence is inferred from its gravitational effects on visible matter. Dark matter does not interact with electromagnetic radiation, making it invisible to the entire electromagnetic spectrum. The name refers to its lack of observable emission or interaction with light. Cold dark matter is a specific form of dark matter believed to make up approximately 84.54% of the matter in the universe and interacts very weakly with ordinary matter and electromagnetic radiation.
    
    Answer: Dark matter is a hypothetical type of matter that has never been directly observed but is inferred from its gravitational effects on visible matter. It does not interact with or emit electromagnetic radiation, making it invisible to the entire electromagnetic spectrum. Cold dark matter is a specific form of dark matter believed to make up approximately 84.54% of the matter in the universe and interacts very weakly with ordinary matter and electromagnetic radiation.[0m
    
    
    
    
    
    
    Given a context, question, answer the question.
    
    ---
    
    Follow the following format.
    
    Context: ${context}
    
    Question: ${question}
    
    Reasoning: Let's think step by step in order to ${produce the answer}. We ...
    
    Answer: ${answer}
    
    ---
    
    Context:
    [1] «Dark matter in fiction | Dark matter is defined as hypothetical matter that is undetectable by its emitted radiation, but whose presence can be inferred from gravitational effects on visible matter. It has been used in a variety of fictional media, including computer and video games and books. In such cases, dark matter is usually attributed extraordinary physical or magical properties. Such descriptions are often inconsistent with the known properties of dark matter proposed in physics and cosmology. For example in computer games, it is often used as material for making weapons and items, and is usually depicted as black or a similar color.»
    [2] «Dark Matter (disambiguation) | Dark matter is matter that is undetectable by its emitted radiation, but whose presence can be inferred from gravitational effects.»
    [3] «Dark matter | Dark matter is a hypothetical type of matter distinct from baryonic matter (ordinary matter such as protons and neutrons), neutrinos and dark energy. Dark matter has never been directly observed; however, its existence would explain a number of otherwise puzzling astronomical observations. The name refers to the fact that it does not emit or interact with electromagnetic radiation, such as light, and is thus invisible to the entire electromagnetic spectrum. Although dark matter has not been directly observed, its existence and properties are inferred from its gravitational effects such as the motions of visible matter, gravitational lensing, its influence on the universe's large-scale structure, on galaxies, and its effects on the cosmic microwave background.»
    [4] «Cold dark matter | In cosmology and physics, cold dark matter (CDM) is a hypothetical form of dark matter whose particles moved slowly compared to the speed of light (the "cold" in CDM) since the universe was approximately one year old (a time when the cosmic particle horizon contained the mass of one typical galaxy); and interact very weakly with ordinary matter and electromagnetic radiation (the "dark" in CDM). It is believed that approximately 84.54% of matter in the Universe is dark matter, with only a small fraction being the ordinary baryonic matter that composes stars, planets and living organisms.»
    [5] «Dark•Matter | Dark•Matter is a science fiction/conspiracy theory campaign setting that was originally published in 1999 by Wizards of the Coast as the second campaign setting for the "Alternity" role-playing game. It was written by Wolfgang Baur and Monte Cook. It was later converted to "d20 Modern" rules and published as a stand-alone book in 2006.»
    
    Question: What is dark matter in physics?
    
    Reasoning: Let's think step by step in order to[32m answer the question. We will first examine the context provided to understand what dark matter is in the context of physics. The context states that dark matter is a hypothetical type of matter distinct from baryonic matter, neutrinos and dark energy. It has never been directly observed but its existence is inferred from its gravitational effects on visible matter. Dark matter does not interact with electromagnetic radiation, making it invisible to the entire electromagnetic spectrum. The name refers to its lack of observable emission or interaction with light. Cold dark matter is a specific form of dark matter believed to make up approximately 84.54% of the matter in the universe and interacts very weakly with ordinary matter and electromagnetic radiation.
    
    Answer: Dark matter is a hypothetical type of matter that has never been directly observed but is inferred from its gravitational effects on visible matter. It does not interact with or emit electromagnetic radiation, making it invisible to the entire electromagnetic spectrum. Cold dark matter is a specific form of dark matter believed to make up approximately 84.54% of the matter in the universe and interacts very weakly with ordinary matter and electromagnetic radiation.[0m
    
    
    
    

## All this is amazing! 😜 Feel the wizardy in DSPy Modularity 🧙‍♀️
