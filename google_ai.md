---
keywords: [gemini, GoogleGenerativeAI, gemini-pro]
---
# Google AI


:::caution
You are currently on a page documenting the use of Google models as [text completion models](/docs/concepts/text_llms). Many popular Google models are [chat completion models](/docs/concepts/chat_models).

You may be looking for [this page instead](/docs/integrations/chat/google_generative_ai/).
:::

A guide on using [Google Generative AI](https://developers.generativeai.google/) models with Langchain. Note: It's separate from Google Cloud Vertex AI [integration](/docs/integrations/llms/google_vertex_ai_palm).

## Setting up


To use Google Generative AI you must install the `langchain-google-genai` Python package and generate an API key. [Read more details](https://developers.generativeai.google/).


```python
%pip install --upgrade --quiet  langchain-google-genai
```


```python
from langchain_google_genai import GoogleGenerativeAI
```


```python
from getpass import getpass

api_key = getpass()
```


```python
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)
print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language?"
    )
)
```

    **Pros of Python:**
    
    * **Easy to learn:** Python is a very easy-to-learn programming language, even for beginners. Its syntax is simple and straightforward, and there are a lot of resources available to help you get started.
    * **Versatile:** Python can be used for a wide variety of tasks, including web development, data science, and machine learning. It's also a good choice for beginners because it can be used for a variety of projects, so you can learn the basics and then move on to more complex tasks.
    * **High-level:** Python is a high-level programming language, which means that it's closer to human language than other programming languages. This makes it easier to read and understand, which can be a big advantage for beginners.
    * **Open-source:** Python is an open-source programming language, which means that it's free to use and there are a lot of resources available to help you learn it.
    * **Community:** Python has a large and active community of developers, which means that there are a lot of people who can help you if you get stuck.
    
    **Cons of Python:**
    
    * **Slow:** Python is a relatively slow programming language compared to some other languages, such as C++. This can be a disadvantage if you're working on computationally intensive tasks.
    * **Not as performant:** Python is not as performant as some other programming languages, such as C++ or Java. This can be a disadvantage if you're working on projects that require high performance.
    * **Dynamic typing:** Python is a dynamically typed programming language, which means that the type of a variable can change during runtime. This can be a disadvantage if you need to ensure that your code is type-safe.
    * **Unmanaged memory:** Python uses a garbage collection system to manage memory. This can be a disadvantage if you need to have more control over memory management.
    
    Overall, Python is a very good programming language for beginners. It's easy to learn, versatile, and has a large community of developers. However, it's important to be aware of its limitations, such as its slow performance and lack of performance.
    


```python
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language?"
    )
)
```

    **Pros:**
    
    * **Simplicity and Readability:** Python is known for its simple and easy-to-read syntax, which makes it accessible to beginners and reduces the chance of errors. It uses indentation to define blocks of code, making the code structure clear and visually appealing.
    
    * **Versatility:** Python is a general-purpose language, meaning it can be used for a wide range of tasks, including web development, data science, machine learning, and desktop applications. This versatility makes it a popular choice for various projects and industries.
    
    * **Large Community:** Python has a vast and active community of developers, which contributes to its growth and popularity. This community provides extensive documentation, tutorials, and open-source libraries, making it easy for Python developers to find support and resources.
    
    * **Extensive Libraries:** Python offers a rich collection of libraries and frameworks for various tasks, such as data analysis (NumPy, Pandas), web development (Django, Flask), machine learning (Scikit-learn, TensorFlow), and many more. These libraries provide pre-built functions and modules, allowing developers to quickly and efficiently solve common problems.
    
    * **Cross-Platform Support:** Python is cross-platform, meaning it can run on various operating systems, including Windows, macOS, and Linux. This allows developers to write code that can be easily shared and used across different platforms.
    
    **Cons:**
    
    * **Speed and Performance:** Python is generally slower than compiled languages like C++ or Java due to its interpreted nature. This can be a disadvantage for performance-intensive tasks, such as real-time systems or heavy numerical computations.
    
    * **Memory Usage:** Python programs tend to consume more memory compared to compiled languages. This is because Python uses a dynamic memory allocation system, which can lead to memory fragmentation and higher memory usage.
    
    * **Lack of Static Typing:** Python is a dynamically typed language, which means that data types are not explicitly defined for variables. This can make it challenging to detect type errors during development, which can lead to unexpected behavior or errors at runtime.
    
    * **GIL (Global Interpreter Lock):** Python uses a global interpreter lock (GIL) to ensure that only one thread can execute Python bytecode at a time. This can limit the scalability and parallelism of Python programs, especially in multi-threaded or multiprocessing scenarios.
    
    * **Package Management:** While Python has a vast ecosystem of libraries and packages, managing dependencies and package versions can be challenging. The Python Package Index (PyPI) is the official repository for Python packages, but it can be difficult to ensure compatibility and avoid conflicts between different versions of packages.
    

## Using in a chain


```python
from langchain_core.prompts import PromptTemplate
```


```python
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "How much is 2+2?"
print(chain.invoke({"question": question}))
```

    4
    

## Streaming calls


```python
import sys

for chunk in llm.stream("Tell me a short poem about snow"):
    sys.stdout.write(chunk)
    sys.stdout.flush()
```

    In winter's embrace, a silent ballet,
    Snowflakes descend, a celestial display.
    Whispering secrets, they softly fall,
    A blanket of white, covering all.
    
    With gentle grace, they paint the land,
    Transforming the world into a winter wonderland.
    Trees stand adorned in icy splendor,
    A glistening spectacle, a sight to render.
    
    Snowflakes twirl, like dancers on a stage,
    Creating a symphony, a winter montage.
    Their silent whispers, a sweet serenade,
    As they dance and twirl, a snowy cascade.
    
    In the hush of dawn, a frosty morn,
    Snow sparkles bright, like diamonds reborn.
    Each flake unique, in its own design,
    A masterpiece crafted by the divine.
    
    So let us revel in this wintry bliss,
    As snowflakes fall, with a gentle kiss.
    For in their embrace, we find a peace profound,
    A frozen world, with magic all around.

### Safety Settings

Gemini models have default safety settings that can be overridden. If you are receiving lots of "Safety Warnings" from your models, you can try tweaking the `safety_settings` attribute of the model. For example, to turn off safety blocking for dangerous content, you can construct your LLM as follows:


```python
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory

llm = GoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=api_key,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)
```

For an enumeration of the categories and thresholds available, see Google's [safety setting types](https://ai.google.dev/api/python/google/generativeai/types/SafetySettingDict).
