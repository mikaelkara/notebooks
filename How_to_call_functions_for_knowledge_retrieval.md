# How to use functions with a knowledge base

This notebook builds on the concepts in the [argument generation](How_to_call_functions_with_chat_models.ipynb) notebook, by creating an agent with access to a knowledge base and two functions that it can call based on the user requirement.

We'll create an agent that uses data from arXiv to answer questions about academic subjects. It has two functions at its disposal:
- **get_articles**: A function that gets arXiv articles on a subject and summarizes them for the user with links.
- **read_article_and_summarize**: This function takes one of the previously searched articles, reads it in its entirety and summarizes the core argument, evidence and conclusions.

This will get you comfortable with a multi-function workflow that can choose from multiple services, and where some of the data from the first function is persisted to be used by the second.

## Walkthrough

This cookbook takes you through the following workflow:

- **Search utilities:** Creating the two functions that access arXiv for answers.
- **Configure Agent:** Building up the Agent behaviour that will assess the need for a function and, if one is required, call that function and present results back to the agent.
- **arXiv conversation:** Put all of this together in live conversation.



```python
!pip install scipy --quiet
!pip install tenacity --quiet
!pip install tiktoken==0.3.3 --quiet
!pip install termcolor --quiet
!pip install openai --quiet
!pip install arxiv --quiet
!pip install pandas --quiet
!pip install PyPDF2 --quiet
!pip install tqdm --quiet
```


```python
import os
import arxiv
import ast
import concurrent
import json
import os
import pandas as pd
import tiktoken
from csv import writer
from IPython.display import display, Markdown, Latex
from openai import OpenAI
from PyPDF2 import PdfReader
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from termcolor import colored

GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"
client = OpenAI()
```

## Search utilities

We'll first set up some utilities that will underpin our two functions.

Downloaded papers will be stored in a directory (we use ```./data/papers``` here). We create a file ```arxiv_library.csv``` to store the embeddings and details for downloaded papers to retrieve against using ```summarize_text```.


```python
directory = './data/papers'

# Check if the directory already exists
if not os.path.exists(directory):
    # If the directory doesn't exist, create it and any necessary intermediate directories
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully.")
else:
    # If the directory already exists, print a message indicating it
    print(f"Directory '{directory}' already exists.")
```

    Directory './data/papers' already exists.
    


```python
# Set a directory to store downloaded papers
data_dir = os.path.join(os.curdir, "data", "papers")
paper_dir_filepath = "./data/arxiv_library.csv"

# Generate a blank dataframe where we can store downloaded files
df = pd.DataFrame(list())
df.to_csv(paper_dir_filepath)
```


```python
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def embedding_request(text):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def get_articles(query, library=paper_dir_filepath, top_k=5):
    """This function gets the top_k articles based on a user's query, sorted by relevance.
    It also downloads the files and stores them in arxiv_library.csv to be retrieved by the read_article_and_summarize.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query = "quantum",
        max_results = 10,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    result_list = []
    for result in client.results(search):
        result_dict = {}
        result_dict.update({"title": result.title})
        result_dict.update({"summary": result.summary})

        # Taking the first url provided
        result_dict.update({"article_url": [x.href for x in result.links][0]})
        result_dict.update({"pdf_url": [x.href for x in result.links][1]})
        result_list.append(result_dict)

        # Store references in library file
        response = embedding_request(text=result.title)
        file_reference = [
            result.title,
            result.download_pdf(data_dir),
            response.data[0].embedding,
        ]

        # Write to file
        with open(library, "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(file_reference)
            f_object.close()
    return result_list

```


```python
# Test that the search is working
result_output = get_articles("ppo reinforcement learning")
result_output[0]

```




    {'title': 'Long-range entanglement from spontaneous non-onsite symmetry breaking',
     'summary': "We explore the states of matter arising from the spontaneous symmetry\nbreaking (SSB) of $\\mathbb{Z}_2$ non-onsite symmetries. In one spatial\ndimension, we construct a frustration-free lattice model exhibiting SSB of a\nnon-onsite symmetry, which features the coexistence of two ground states with\ndistinct symmetry-protected topological (SPT) orders. We analytically prove the\ntwo-fold ground-state degeneracy and the existence of a finite energy gap.\nFixing the symmetry sector yields a long-range entangled ground state that\nfeatures long-range correlations among non-invertible charged operators. We\nalso present a constant-depth measurement-feedback protocol to prepare such a\nstate with a constant success probability in the thermodynamic limit, which may\nbe of independent interest. Under a symmetric deformation, the SSB persists up\nto a critical point, beyond which a gapless phase characterized by a conformal\nfield theory emerges. In two spatial dimensions, the SSB of 1-form non-onsite\nsymmetries leads to a long-range entangled state (SPT soup) - a condensate of\n1d SPT along any closed loops. On a torus, there are four such locally\nindistinguishable states that exhibit algebraic correlations between local\noperators, which we derived via a mapping to the critical $O(2)$ loop model.\nThis provides an intriguing example of `topological quantum criticality'. Our\nwork reveals the exotic features of SSB of non-onsite symmetries, which may lie\nbeyond the framework of topological holography (SymTFT).",
     'article_url': 'http://arxiv.org/abs/2411.05004v1',
     'pdf_url': 'http://arxiv.org/pdf/2411.05004v1'}




```python
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> list[str]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = embedding_request(query)
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["filepath"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n]

```


```python
def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    return pdf_text


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    """Returns successive n-sized chunks from provided text."""
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


def extract_chunk(content, template_prompt):
    """This function applies a prompt to some input content. In this case it returns a summarized chunk of text"""
    prompt = template_prompt + content
    response = client.chat.completions.create(
        model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response.choices[0].message.content


def summarize_text(query):
    """This function does the following:
    - Reads in the arxiv_library.csv file in including the embeddings
    - Finds the closest file to the user's query
    - Scrapes the text out of the file and chunks it
    - Summarizes each chunk in parallel
    - Does one final summary and returns this to the user"""

    # A prompt to dictate how the recursive summarizations should approach the input paper
    summary_prompt = """Summarize this text from an academic paper. Extract any key points with reasoning.\n\nContent:"""

    # If the library is empty (no searches have been performed yet), we perform one and download the results
    library_df = pd.read_csv(paper_dir_filepath).reset_index()
    if len(library_df) == 0:
        print("No papers searched yet, downloading first.")
        get_articles(query)
        print("Papers downloaded, continuing")
        library_df = pd.read_csv(paper_dir_filepath).reset_index()
    library_df.columns = ["title", "filepath", "embedding"]
    library_df["embedding"] = library_df["embedding"].apply(ast.literal_eval)
    strings = strings_ranked_by_relatedness(query, library_df, top_n=1)
    print("Chunking text from paper")
    pdf_text = read_pdf(strings[0])

    # Initialise tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    results = ""

    # Chunk up the document into 1500 token chunks
    chunks = create_chunks(pdf_text, 1500, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    print("Summarizing each chunk of text")

    # Parallel process the summaries
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(text_chunks)
    ) as executor:
        futures = [
            executor.submit(extract_chunk, chunk, summary_prompt)
            for chunk in text_chunks
        ]
        with tqdm(total=len(text_chunks)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for future in futures:
            data = future.result()
            results += data

    # Final summary
    print("Summarizing into overall summary")
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Write a summary collated from this collection of key points extracted from an academic paper.
                        The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
                        User query: {query}
                        The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
                        Key points:\n{results}\nSummary:\n""",
            }
        ],
        temperature=0,
    )
    return response

```


```python
# Test the summarize_text function works
chat_test_response = summarize_text("PPO reinforcement learning sequence generation")

```

    Chunking text from paper
    Summarizing each chunk of text
    

    100%|██████████| 18/18 [00:13<00:00,  1.37it/s]
    

    Summarizing into overall summary
    


```python
print(chat_test_response.choices[0].message.content)

```

    ### Core Argument
    - The paper presents a novel approach to quantum error correction (QEC) through the development of topological stabilizer codes based on continuous variable (CV) systems.
    - It introduces a family of two-dimensional topological stabilizer codes that leverage boson condensation, generalizing existing models and addressing limitations in current QEC codes.
    
    ### Evidence
    - **Topological Stabilizer Codes**: The authors construct codes that utilize continuous variable degrees of freedom, extending models like homological rotor codes and the toric-GKP code.
    - **Boson Condensation**: The codes are derived from a parent stabilizer code based on Rgauge theory, demonstrating the creation of a broad class of topological CV stabilizer codes.
    - **Anyons and Theories**: The codes are characterized by anyon theories from U(1) 2n×U(1)−2m Chern-Simons theories, some of which are non-chiral and do not support gapped boundaries, indicating their unique properties.
    - **Hamiltonian Analysis**: The paper examines Hamiltonians associated with these codes, revealing that they can transition from a gapless to a gapped spectrum through quadratic perturbations.
    - **Scalability**: The research emphasizes the potential for scalable QEC codes that are intrinsic to CV systems, which is crucial for reducing logical error rates in complex quantum computations.
    
    ### Conclusions
    - The introduction of topological stabilizer codes based on continuous variables represents a significant advancement in the field of quantum error correction, providing a framework for developing fault-tolerant codes that can protect against logical errors.
    - The findings suggest that these codes may not be realizable through traditional concatenation methods, indicating the potential for new classes of QEC codes that are fundamentally different from existing finite-dimensional systems.
    - Future research directions include exploring many-body generalizations of quantum codes and the practical implementation of these theoretical models in experimental platforms, such as superconducting qubits.
    

## Configure Agent

We'll create our agent in this step, including a ```Conversation``` class to support multiple turns with the API, and some Python functions to enable interaction between the ```ChatCompletion``` API and our knowledge base functions.


```python
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=functions,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

```


```python
class Conversation:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )
```


```python
# Initiate our get_articles and read_article_and_summarize functions
arxiv_functions = [
    {
        "name": "get_articles",
        "description": """Use this function to get academic papers from arXiv to answer user questions.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            User query in JSON. Responses should be summarized and should include the article URL reference
                            """,
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_article_and_summarize",
        "description": """Use this function to read whole papers and provide a summary for users.
        You should NEVER call this function before get_articles has been called in the conversation.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            Description of the article in plain text based on the user's query
                            """,
                }
            },
            "required": ["query"],
        },
    }
]

```


```python
def chat_completion_with_function_execution(messages, functions=[None]):
    """This function makes a ChatCompletion API call with the option of adding functions"""
    response = chat_completion_request(messages, functions)
    full_message = response.choices[0]
    if full_message.finish_reason == "function_call":
        print(f"Function generation requested, calling function")
        return call_arxiv_function(messages, full_message)
    else:
        print(f"Function not required, responding to user")
        return response


def call_arxiv_function(messages, full_message):
    """Function calling function which executes function calls when the model believes it is necessary.
    Currently extended by adding clauses to this if statement."""

    if full_message.message.function_call.name == "get_articles":
        try:
            parsed_output = json.loads(
                full_message.message.function_call.arguments
            )
            print("Getting search results")
            results = get_articles(parsed_output["query"])
        except Exception as e:
            print(parsed_output)
            print(f"Function execution failed")
            print(f"Error message: {e}")
        messages.append(
            {
                "role": "function",
                "name": full_message.message.function_call.name,
                "content": str(results),
            }
        )
        try:
            print("Got search results, summarizing content")
            response = chat_completion_request(messages)
            return response
        except Exception as e:
            print(type(e))
            raise Exception("Function chat request failed")

    elif (
        full_message.message.function_call.name == "read_article_and_summarize"
    ):
        parsed_output = json.loads(
            full_message.message.function_call.arguments
        )
        print("Finding and reading paper")
        summary = summarize_text(parsed_output["query"])
        return summary

    else:
        raise Exception("Function does not exist and cannot be called")

```

## arXiv conversation

Let's put this all together by testing our functions out in conversation.


```python
# Start with a system message
paper_system_message = """You are arXivGPT, a helpful assistant pulls academic papers to answer user questions.
You summarize the papers clearly so the customer can decide which to read to answer their question.
You always provide the article_url and title so the user can understand the name of the paper and click through to access it.
Begin!"""
paper_conversation = Conversation()
paper_conversation.add_message("system", paper_system_message)

```


```python
# Add a user message
paper_conversation.add_message("user", "Hi, how does PPO reinforcement learning work?")
chat_response = chat_completion_with_function_execution(
    paper_conversation.conversation_history, functions=arxiv_functions
)
assistant_message = chat_response.choices[0].message.content
paper_conversation.add_message("assistant", assistant_message)
display(Markdown(assistant_message))

```

    Function generation requested, calling function
    Getting search results
    Got search results, summarizing content
    


It looks like none of the listed papers directly address Proximal Policy Optimization (PPO) in reinforcement learning. However, if you are looking for a summary of how PPO works, I can provide that information based on general knowledge.

PPO is a popular reinforcement learning algorithm that seeks to optimize an agent's policy in a way that is both effective and stable. The key ideas behind PPO include:

1. **Clipping Objective**: PPO modifies the objective function using a clipped surrogate objective. This avoids excessively large updates, which can destabilize training. The clipped objective restricts how much the probability ratio between the new and old policies can differ, allowing for better control over policy updates.

2. **Policy Gradient**: PPO uses policy gradient methods, where the focus is on directly adjusting the policy based on the agent's performance after taking actions in the environment.

3. **Multiple Epochs**: PPO often utilizes multiple epochs of training on the same batch of data, which improves sample efficiency as it allows the agent to learn from the same data multiple times.

4. **Generalized Advantage Estimation (GAE)**: To reduce the variance of the policy gradient estimates, PPO frequently employs GAE, which provides a more stable estimate of returns.

If you have specific questions about reinforcement learning or want references to further reading, feel free to ask!



```python
# Add another user message to induce our system to use the second tool
paper_conversation.add_message(
    "user",
    "Can you read the PPO sequence generation paper for me and give me a summary",
)
updated_response = chat_completion_with_function_execution(
    paper_conversation.conversation_history, functions=arxiv_functions
)
display(Markdown(updated_response.choices[0].message.content))

```

    Function generation requested, calling function
    Getting search results
    Got search results, summarizing content
    


It seems that I couldn't find any specific paper that discusses "PPO sequence generation." If you meant a particular paper on PPO in reinforcement learning, please provide its title or broad topic. Alternatively, if you want to know about PPO in the context of generating sequences or trajectories, I can provide a detailed explanation based on existing knowledge in reinforcement learning.

If you are referring to a specific recent study, I recommend checking relevant platforms like arXiv or Google Scholar for the latest publications on PPO and sequence generation.

Let me know how you'd like to proceed!

