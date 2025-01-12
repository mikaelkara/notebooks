# Agentic RAG: turbocharge your RAG with query reformulation and self-query! 🚀
_Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_

> This tutorial is advanced. You should have notions from [this other cookbook](advanced_rag) first!

> Reminder: Retrieval-Augmented-Generation (RAG) is “using an LLM to answer a user query, but basing the answer on information retrieved from a knowledge base”. It has many advantages over using a vanilla or fine-tuned LLM: to name a few, it allows to ground the answer on true facts and reduce confabulations, it allows to provide the LLM with domain-specific knowledge, and it allows fine-grained control of access to information from the knowledge base.

But vanilla RAG has limitations, most importantly these two:
- It **performs only one retrieval step**: if the results are bad, the generation in turn will be bad.
- __Semantic similarity is computed with the *user query* as a reference__, which might be suboptimal: for instance, the user query will often be a question and the document containing the true answer will be in affirmative voice, so its similarity score will be downgraded compared to other source documents in the interrogative form, leading to a risk of missing the relevant information.

But we can alleviate these problems by making a **RAG agent: very simply, an agent armed with a retriever tool!**

This agent will: ✅ Formulate the query itself and ✅ Critique to re-retrieve if needed.

So it should naively recover some advanced RAG techniques!
- Instead of directly using the user query as the reference in semantic search, the agent formulates itself a reference sentence that can be closer to the targeted documents, as in [HyDE](https://huggingface.co/papers/2212.10496)
- The agent can the generated snippets and re-retrieve if needed, as in [Self-Query](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery/)

Let's build this system. 🛠️

Run the line below to install required dependencies:


```python
!pip install pandas langchain langchain-community sentence-transformers faiss-cpu "transformers[agents]" --upgrade -q
```

Let's login in order to call the HF Inference API:


```python
from huggingface_hub import notebook_login

notebook_login()
```

We first load a knowledge base on which we want to perform RAG: this dataset is a compilation of the documentation pages for many `huggingface` packages, stored as markdown.


```python
import datasets

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
```

Now we prepare the knowledge base by processing the dataset and storing it into a vector database to be used by the retriever.

We use [LangChain](https://python.langchain.com/) for its excellent vector database utilities.
For the embedding model, we use [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) since it performed well in our `RAG_evaluation` cookbook.


```python
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Split docs and keep only unique ones
print("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)

print(
    "Embedding documents... This should take a few minutes (5 minutes on MacBook with M1 Pro)"
)
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
```

Now the database is ready: let’s build our agentic RAG system!

👉 We only need a `RetrieverTool` that our agent can leverage to retrieve information from the knowledge base.

Since we need to add a vectordb as an attribute of the tool, we cannot simply use the [simple tool constructor](https://huggingface.co/docs/transformers/main/en/agents#create-a-new-tool) with a `@tool` decorator: so we will follow the advanced setup highlighted in the [advanced agents documentation](https://huggingface.co/docs/transformers/main/en/agents_advanced#directly-define-a-tool-by-subclassing-tool-and-share-it-to-the-hub).


```python
from transformers.agents import Tool
from langchain_core.vectorstores import VectorStore


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [
                f"===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )
```

Now it’s straightforward to create an agent that leverages this tool!

The agent will need these arguments upon initialization:
- *`tools`*: a list of tools that the agent will be able to call.
- *`llm_engine`*: the LLM that powers the agent.

Our `llm_engine` must be a callable that takes as input a list of [messages](https://huggingface.co/docs/transformers/main/chat_templating) and returns text. It also needs to accept a `stop_sequences` argument that indicates when to stop its generation. For convenience, we directly use the `HfEngine` class provided in the package to get a LLM engine that calls our [Inference API](https://huggingface.co/docs/api-inference/en/index).

And we use [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) as the llm engine because:
- It has a long 128k context, which is helpful for processing long source documents
- It is served for free at all times on HF's Inference API!

_Note:_ The Inference API hosts models based on various criteria, and deployed models may be updated or replaced without prior notice. Learn more about it [here](https://huggingface.co/docs/api-inference/supported-models).


```python
from transformers.agents import HfApiEngine, ReactJsonAgent

llm_engine = HfApiEngine("Qwen/Qwen2.5-72B-Instruct")

retriever_tool = RetrieverTool(vectordb)
agent = ReactJsonAgent(
    tools=[retriever_tool], llm_engine=llm_engine, max_iterations=4, verbose=2
)
```

Since we initialized the agent as a `ReactJsonAgent`, it has been automatically given a default system prompt that tells the LLM engine to process step-by-step and generate tool calls as JSON blobs (you could replace this prompt template with your own as needed).

Then when its `.run()` method is launched, the agent takes care of calling the LLM engine, parsing the tool call JSON blobs and executing these tool calls, all in a loop that ends only when the final answer is provided.


```python
agent_output = agent.run("How can I push a model to the Hub?")

print("Final output:")
print(agent_output)
```

    [32;20;1m======== New task ========[0m
    [37;1mHow can I push a model to the Hub?[0m
    [38;20mSystem prompt is as follows:[0m
    [38;20mYou are an expert assistant who can solve any task using JSON tool calls. You will be given a task to solve as best you can.
    To do so, you have been given access to the following tools: 'retriever', 'final_answer'
    The way you use the tools is by specifying a json blob, ending with '<end_action>'.
    Specifically, this json should have an `action` key (name of the tool to use) and an `action_input` key (input to the tool).
    
    The $ACTION_JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. It should be formatted in json. Do not try to escape special characters. Here is the template of a valid $ACTION_JSON_BLOB:
    {
      "action": $TOOL_NAME,
      "action_input": $INPUT
    }<end_action>
    
    Make sure to have the $INPUT as a dictionary in the right format for the tool you are using, and do not put variable names as input if you can find the right values.
    
    You should ALWAYS use the following format:
    
    Thought: you should always think about one action to take. Then use the action as follows:
    Action:
    $ACTION_JSON_BLOB
    Observation: the result of the action
    ... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $ACTION_JSON_BLOB must only use a SINGLE action at a time.)
    
    You can use the result of the previous action as input for the next action.
    The observation will always be a string: it can represent a file, like "image_1.jpg".
    Then you can use it as input for the next action. You can do it for instance as follows:
    
    Observation: "image_1.jpg"
    
    Thought: I need to transform the image that I received in the previous observation to make it green.
    Action:
    {
      "action": "image_transformer",
      "action_input": {"image": "image_1.jpg"}
    }<end_action>
    
    To provide the final answer to the task, use an action blob with "action": "final_answer" tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:
    Action:
    {
      "action": "final_answer",
      "action_input": {"answer": "insert your final answer here"}
    }<end_action>
    
    
    Here are a few examples using notional tools:
    ---
    Task: "Generate an image of the oldest person in this document."
    
    Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
    Action:
    {
      "action": "document_qa",
      "action_input": {"document": "document.pdf", "question": "Who is the oldest person mentioned?"}
    }<end_action>
    Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."
    
    
    Thought: I will now generate an image showcasing the oldest person.
    Action:
    {
      "action": "image_generator",
      "action_input": {"prompt": "A portrait of John Doe, a 55-year-old man living in Canada."}
    }<end_action>
    Observation: "image.png"
    
    Thought: I will now return the generated image.
    Action:
    {
      "action": "final_answer",
      "action_input": "image.png"
    }<end_action>
    
    ---
    Task: "What is the result of the following operation: 5 + 3 + 1294.678?"
    
    Thought: I will use python code evaluator to compute the result of the operation and then return the final answer using the `final_answer` tool
    Action:
    {
        "action": "python_interpreter",
        "action_input": {"code": "5 + 3 + 1294.678"}
    }<end_action>
    Observation: 1302.678
    
    Thought: Now that I know the result, I will now return it.
    Action:
    {
      "action": "final_answer",
      "action_input": "1302.678"
    }<end_action>
    
    ---
    Task: "Which city has the highest population , Guangzhou or Shanghai?"
    
    Thought: I need to get the populations for both cities and compare them: I will use the tool `search` to get the population of both cities.
    Action:
    {
        "action": "search",
        "action_input": "Population Guangzhou"
    }<end_action>
    Observation: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
    
    
    Thought: Now let's get the population of Shanghai using the tool 'search'.
    Action:
    {
        "action": "search",
        "action_input": "Population Shanghai"
    }
    Observation: '26 million (2019)'
    
    Thought: Now I know that Shanghai has a larger population. Let's return the result.
    Action:
    {
      "action": "final_answer",
      "action_input": "Shanghai"
    }<end_action>
    
    
    Above example were using notional tools that might not exist for you. You only have acces to those tools:
    
    - retriever: Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query.
        Takes inputs: {'query': {'type': 'string', 'description': 'The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.'}}
        Returns an output of type: string
    
    - final_answer: Provides a final answer to the given problem.
        Takes inputs: {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
        Returns an output of type: any
    
    Here are the rules you should always follow to solve your task:
    1. ALWAYS provide a 'Thought:' sequence, and an 'Action:' sequence that ends with <end_action>, else you will fail.
    2. Always use the right arguments for the tools. Never use variable names in the 'action_input' field, use the value instead.
    3. Call a tool only when needed: do not call the search agent if you do not need information, try to solve the task yourself.
    4. Never re-do a tool call that you previously did with the exact same parameters.
    
    Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
    [0m
    [38;20m===== New step =====[0m
    ===== Calling LLM with this last message: =====
    {'role': <MessageRole.USER: 'user'>, 'content': 'Task: How can I push a model to the Hub?'}
    [38;20m===== Output message of the LLM: =====[0m
    [38;20mThought: To find out how to push a model to the Hub, I need to search the knowledge base for relevant information using the 'retriever' tool.
    Action:
    {
      "action": "retriever",
      "action_input": {"query": "push a model to the Hub"}
    }[0m
    [38;20m===== Extracting action =====[0m
    [33;1m=== Agent thoughts:[0m
    [0mThought: To find out how to push a model to the Hub, I need to search the knowledge base for relevant information using the 'retriever' tool.[0m
    [33;1m>>> Calling tool: 'retriever' with arguments: {'query': 'push a model to the Hub'}[0m
    Retrieved documents:
    ===== Document 0 =====
    # Step 7. Push everything to the Hub
        api.upload_folder(
            repo_id=repo_id,
            folder_path=repo_local_path,
            path_in_repo=".",
        )
    
        print("Your model is pushed to the Hub. You can view your model here: ", repo_url)
    ```
    
    ### .
    
    By using `push_to_hub` **you evaluate, record a replay, generate a model card of your agent and push it to the Hub**.===== Document 1 =====
    ```py
    >>> trainer.push_to_hub()
    ```
    </pt>
    <tf>
    Share a model to the Hub with [`PushToHubCallback`]. In the [`PushToHubCallback`] function, add:
    
    - An output directory for your model.
    - A tokenizer.
    - The `hub_model_id`, which is your Hub username and model name.
    
    ```py
    >>> from transformers import PushToHubCallback
    
    >>> push_to_hub_callback = PushToHubCallback(
    ...     output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
    ... )
    ```===== Document 2 =====
    Let's pretend we've now fine-tuned the model. The next step would be to push it to the Hub! We can do this with the `timm.models.hub.push_to_hf_hub` function.
    
    ```py
    >>> model_cfg = dict(labels=['a', 'b', 'c', 'd'])
    >>> timm.models.hub.push_to_hf_hub(model, 'resnet18-random', model_config=model_cfg)
    ```
    
    Running the above would push the model to `<your-username>/resnet18-random` on the Hub. You can now share this model with your friends, or use it in your own code!
    
    ## Loading a Model===== Document 3 =====
    processor.push_to_hub(hub_model_id)
    trainer.push_to_hub(**kwargs)
    ```
    
    # 4. Inference
    
    Now comes the exciting part, using our fine-tuned model! In this section, we'll show how you can load your model from the hub and use it for inference.===== Document 4 =====
    Finally, if you want, you can push your model up to the hub. Here, we'll push it up if you specified `push_to_hub=True` in the training configuration. Note that in order to push to hub, you'll have to have git-lfs installed and be logged into your Hugging Face account (which can be done via `huggingface-cli login`).
    
    ```python
    kwargs = {
        "finetuned_from": model.config._name_or_path,
        "tasks": "image-classification",
        "dataset": 'beans',
        "tags": ['image-classification'],
    }===== Document 5 =====
    --push_to_hub
    ```===== Document 6 =====
    . The second way to upload a model, though, is to call model.push_to_hub(). So this is more of a once-off method - it's not called regularly during training. You can just call this manually whenever you want to upload a model to the hub. So we recommend running this after the end of training, just to make sure that you have a commit message just to guarantee that this was the final version of the model at the end of training. And it just makes sure that you're working with the definitive end-of-training model and not accidentally using a model that's from a checkpoint somewhere along the way
    [38;20m===== New step =====[0m
    ===== Calling LLM with this last message: =====
    {'role': <MessageRole.TOOL_RESPONSE: 'tool-response'>, 'content': '[OUTPUT OF STEP 0] -> Observation:\nRetrieved documents:\n===== Document 0 =====\n# Step 7. Push everything to the Hub\n    api.upload_folder(\n        repo_id=repo_id,\n        folder_path=repo_local_path,\n        path_in_repo=".",\n    )\n\n    print("Your model is pushed to the Hub. You can view your model here: ", repo_url)\n```\n\n### .\n\nBy using `push_to_hub` **you evaluate, record a replay, generate a model card of your agent and push it to the Hub**.===== Document 1 =====\n```py\n>>> trainer.push_to_hub()\n```\n</pt>\n<tf>\nShare a model to the Hub with [`PushToHubCallback`]. In the [`PushToHubCallback`] function, add:\n\n- An output directory for your model.\n- A tokenizer.\n- The `hub_model_id`, which is your Hub username and model name.\n\n```py\n>>> from transformers import PushToHubCallback\n\n>>> push_to_hub_callback = PushToHubCallback(\n...     output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"\n... )\n```===== Document 2 =====\nLet\'s pretend we\'ve now fine-tuned the model. The next step would be to push it to the Hub! We can do this with the `timm.models.hub.push_to_hf_hub` function.\n\n```py\n>>> model_cfg = dict(labels=[\'a\', \'b\', \'c\', \'d\'])\n>>> timm.models.hub.push_to_hf_hub(model, \'resnet18-random\', model_config=model_cfg)\n```\n\nRunning the above would push the model to `<your-username>/resnet18-random` on the Hub. You can now share this model with your friends, or use it in your own code!\n\n## Loading a Model===== Document 3 =====\nprocessor.push_to_hub(hub_model_id)\ntrainer.push_to_hub(**kwargs)\n```\n\n# 4. Inference\n\nNow comes the exciting part, using our fine-tuned model! In this section, we\'ll show how you can load your model from the hub and use it for inference.===== Document 4 =====\nFinally, if you want, you can push your model up to the hub. Here, we\'ll push it up if you specified `push_to_hub=True` in the training configuration. Note that in order to push to hub, you\'ll have to have git-lfs installed and be logged into your Hugging Face account (which can be done via `huggingface-cli login`).\n\n```python\nkwargs = {\n    "finetuned_from": model.config._name_or_path,\n    "tasks": "image-classification",\n    "dataset": \'beans\',\n    "tags": [\'image-classification\'],\n}===== Document 5 =====\n--push_to_hub\n```===== Document 6 =====\n. The second way to upload a model, though, is to call model.push_to_hub(). So this is more of a once-off method - it\'s not called regularly during training. You can just call this manually whenever you want to upload a model to the hub. So we recommend running this after the end of training, just to make sure that you have a commit message just to guarantee that this was the final version of the model at the end of training. And it just makes sure that you\'re working with the definitive end-of-training model and not accidentally using a model that\'s from a checkpoint somewhere along the way'}
    [38;20m===== Output message of the LLM: =====[0m
    [38;20mThought: I retrieved information on how to push a model to the Hub. Now, I will gather the main points and return the final answer using the `final_answer` tool.
    Action:
    {
      "action": "final_answer",
      "action_input": {"answer": "There are several ways to push a model to the Hub:\n1. Using `trainer.push_to_hub()` method.\n2. Using `model.push_to_hub()` method, which is a once-off approach after training.\n3. Using `timm.models.hub.push_to_hf_hub` function for specific frameworks.\n4. Using `api.upload_folder` with `repo_id` and `folder_path` to upload the entire folder.\n5. Setting `--push_to_hub` flag in the training configuration.\nTo use these methods, ensure you have `git-lfs` installed and are logged into your Hugging Face account using `huggingface-cli login`."}
    }[0m
    [38;20m===== Extracting action =====[0m
    [33;1m=== Agent thoughts:[0m
    [0mThought: I retrieved information on how to push a model to the Hub. Now, I will gather the main points and return the final answer using the `final_answer` tool.[0m
    [33;1m>>> Calling tool: 'final_answer' with arguments: {'answer': 'There are several ways to push a model to the Hub:\n1. Using `trainer.push_to_hub()` method.\n2. Using `model.push_to_hub()` method, which is a once-off approach after training.\n3. Using `timm.models.hub.push_to_hf_hub` function for specific frameworks.\n4. Using `api.upload_folder` with `repo_id` and `folder_path` to upload the entire folder.\n5. Setting `--push_to_hub` flag in the training configuration.\nTo use these methods, ensure you have `git-lfs` installed and are logged into your Hugging Face account using `huggingface-cli login`.'}[0m
    

    Final output:
    There are several ways to push a model to the Hub:
    1. Using `trainer.push_to_hub()` method.
    2. Using `model.push_to_hub()` method, which is a once-off approach after training.
    3. Using `timm.models.hub.push_to_hf_hub` function for specific frameworks.
    4. Using `api.upload_folder` with `repo_id` and `folder_path` to upload the entire folder.
    5. Setting `--push_to_hub` flag in the training configuration.
    To use these methods, ensure you have `git-lfs` installed and are logged into your Hugging Face account using `huggingface-cli login`.
    

## Agentic RAG vs. standard RAG

Does the agent setup make a better RAG system? Well, let's compare it to a standard RAG system using LLM Judge!

We will use [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) for evaluation since it's one of the strongest OS models we tested for LLM judge use cases.


```python
eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
```

Before running the test let's make the agent less verbose.


```python
import logging

agent.logger.setLevel(logging.WARNING) # Let's reduce the agent's verbosity level

eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
```


```python
outputs_agentic_rag = []

for example in tqdm(eval_dataset):
    question = example["question"]

    enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
give a comprehensive answer to the question below.
Respond only to the question asked, response should be concise and relevant to the question.
If you cannot find information, do not give up and try calling your retriever again with different arguments!
Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
Your queries should not be questions but affirmative form sentences: e.g. rather than "How do I load a model from the Hub in bf16?", query should be "load a model from the Hub bf16 weights".

Question:
{question}"""
    answer = agent.run(enhanced_question)
    print("=======================================================")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f'True answer: {example["answer"]}')

    results_agentic = {
        "question": question,
        "true_answer": example["answer"],
        "source_doc": example["source_doc"],
        "generated_answer": answer,
    }
    outputs_agentic_rag.append(results_agentic)
```


```python
from huggingface_hub import InferenceClient

reader_llm = InferenceClient("Qwen/Qwen2.5-72B-Instruct")

outputs_standard_rag = []

for example in tqdm(eval_dataset):
    question = example["question"]
    context = retriever_tool(question)

    prompt = f"""Given the question and supporting documents below, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If you cannot find information, do not give up and try calling your retriever again with different arguments!

Question:
{question}

{context}
"""
    messages = [{"role": "user", "content": prompt}]
    answer = reader_llm.chat_completion(messages).choices[0].message.content

    print("=======================================================")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f'True answer: {example["answer"]}')

    results_agentic = {
        "question": question,
        "true_answer": example["answer"],
        "source_doc": example["source_doc"],
        "generated_answer": answer,
    }
    outputs_standard_rag.append(results_agentic)
```

The evaluation prompt follows some of the best principles shown in [our llm_judge cookbook](llm_judge): it follows a small integer Likert scale, has clear criteria, and a description for each score.


```python
EVALUATION_PROMPT = """You are a fair evaluator language model.

You will be given an instruction, a response to evaluate, a reference answer that gets a score of 3, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 3. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 3}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.
5. Do not score conciseness: a correct answer that covers the question should receive max score, even if it contains additional useless information.

The instruction to evaluate:
{instruction}

Response to evaluate:
{response}

Reference Answer (Score 3):
{reference_answer}

Score Rubrics:
[Is the response complete, accurate, and factual based on the reference answer?]
Score 1: The response is completely incomplete, inaccurate, and/or not factual.
Score 2: The response is somewhat complete, accurate, and/or factual.
Score 3: The response is completely complete, accurate, and/or factual.

Feedback:"""
```


```python
from huggingface_hub import InferenceClient

evaluation_client = InferenceClient("meta-llama/Llama-3.1-70B-Instruct")
```


```python
import pandas as pd

results = {}
for system_type, outputs in [
    ("agentic", outputs_agentic_rag),
    ("standard", outputs_standard_rag),
]:
    for experiment in tqdm(outputs):
        eval_prompt = EVALUATION_PROMPT.format(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        messages = [
            {"role": "system", "content": "You are a fair evaluator language model."},
            {"role": "user", "content": eval_prompt},
        ]

        eval_result = evaluation_client.text_generation(
            eval_prompt, max_new_tokens=1000
        )
        try:
            feedback, score = [item.strip() for item in eval_result.split("[RESULT]")]
            experiment["eval_score_LLM_judge"] = score
            experiment["eval_feedback_LLM_judge"] = feedback
        except:
            print(f"Parsing failed - output was: {eval_result}")

    results[system_type] = pd.DataFrame.from_dict(outputs)
    results[system_type] = results[system_type].loc[~results[system_type]["generated_answer"].str.contains("Error")]
```


```python
DEFAULT_SCORE = 2 # Give average score whenever scoring fails
def fill_score(x):
    try:
        return int(x)
    except:
        return DEFAULT_SCORE

for system_type, outputs in [
    ("agentic", outputs_agentic_rag),
    ("standard", outputs_standard_rag),
]:

    results[system_type]["eval_score_LLM_judge_int"] = (
        results[system_type]["eval_score_LLM_judge"].fillna(DEFAULT_SCORE).apply(fill_score)
    )
    results[system_type]["eval_score_LLM_judge_int"] = (results[system_type]["eval_score_LLM_judge_int"] - 1) / 2

    print(
        f"Average score for {system_type} RAG: {results[system_type]['eval_score_LLM_judge_int'].mean()*100:.1f}%"
    )
```

    Average score for agentic RAG: 86.9%
    Average score for standard RAG: 73.1%
    

**Let us recap: the Agent setup improves scores by 14% compared to a standard RAG!** (from 73.1% to 86.9%)

This is a great improvement, with a very simple setup 🚀

(For a baseline, using Llama-3-70B without the knowledge base got 36%)
