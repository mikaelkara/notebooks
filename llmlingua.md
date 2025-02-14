# LLMLingua Document Compressor

>[LLMLingua](https://github.com/microsoft/LLMLingua) utilizes a compact, well-trained language model (e.g., GPT2-small, LLaMA-7B) to identify and remove non-essential tokens in prompts. This approach enables efficient inference with large language models (LLMs), achieving up to 20x compression with minimal performance loss.

This notebook shows how to use LLMLingua as a document compressor.


```python
%pip install --upgrade --quiet  llmlingua accelerate
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.3.2[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpython -m pip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.
    


```python
# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
```

## Set up the base vector store retriever
Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.


```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader(
    "../../how_to/state_of_the_union.txt",
).load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)
```

    Document 1:
    
    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 
    
    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
    ----------------------------------------------------------------------------------------------------
    Document 2:
    
    As I said last year, especially to our younger transgender Americans, I will always have your back as your President, so you can be yourself and reach your God-given potential. 
    
    While it often appears that we never agree, that isn’t true. I signed 80 bipartisan bills into law last year. From preventing government shutdowns to protecting Asian-Americans from still-too-common hate crimes to reforming military justice.
    ----------------------------------------------------------------------------------------------------
    Document 3:
    
    A former top litigator in private practice. A former federal public defender. And from a family of public school educators and police officers. A consensus builder. Since she’s been nominated, she’s received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democrats and Republicans. 
    
    And if we are to advance liberty and justice, we need to secure the Border and fix the immigration system.
    ----------------------------------------------------------------------------------------------------
    Document 4:
    
    He met the Ukrainian people. 
    
    From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. 
    
    Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland. 
    
    In this struggle as President Zelenskyy said in his speech to the European Parliament “Light will win over darkness.” The Ukrainian Ambassador to the United States is here tonight.
    ----------------------------------------------------------------------------------------------------
    Document 5:
    
    But that trickle-down theory led to weaker economic growth, lower wages, bigger deficits, and the widest gap between those at the top and everyone else in nearly a century. 
    
    Vice President Harris and I ran for office with a new economic vision for America. 
    
    Invest in America. Educate Americans. Grow the workforce. Build the economy from the bottom up  
    and the middle out, not from the top down.
    ----------------------------------------------------------------------------------------------------
    Document 6:
    
    And tonight, I’m announcing that the Justice Department will name a chief prosecutor for pandemic fraud. 
    
    By the end of this year, the deficit will be down to less than half what it was before I took office.  
    
    The only president ever to cut the deficit by more than one trillion dollars in a single year. 
    
    Lowering your costs also means demanding more competition. 
    
    I’m a capitalist, but capitalism without competition isn’t capitalism. 
    
    It’s exploitation—and it drives up prices.
    ----------------------------------------------------------------------------------------------------
    Document 7:
    
    I spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves. 
    
    I’ve worked on these issues a long time. 
    
    I know what works: Investing in crime prevention and community police officers who’ll walk the beat, who’ll know the neighborhood, and who can restore trust and safety. 
    
    So let’s not abandon our streets. Or choose between safety and equal justice.
    ----------------------------------------------------------------------------------------------------
    Document 8:
    
    As I’ve told Xi Jinping, it is never a good bet to bet against the American people. 
    
    We’ll create good jobs for millions of Americans, modernizing roads, airports, ports, and waterways all across America. 
    
    And we’ll do it all to withstand the devastating effects of the climate crisis and promote environmental justice.
    ----------------------------------------------------------------------------------------------------
    Document 9:
    
    Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  
    
    Last year COVID-19 kept us apart. This year we are finally together again. 
    
    Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. 
    
    With a duty to one another to the American people to the Constitution. 
    
    And with an unwavering resolve that freedom will always triumph over tyranny.
    ----------------------------------------------------------------------------------------------------
    Document 10:
    
    As Ohio Senator Sherrod Brown says, “It’s time to bury the label “Rust Belt.” 
    
    It’s time. 
    
    But with all the bright spots in our economy, record job growth and higher wages, too many families are struggling to keep up with the bills.  
    
    Inflation is robbing them of the gains they might otherwise feel. 
    
    I get it. That’s why my top priority is getting prices under control.
    ----------------------------------------------------------------------------------------------------
    Document 11:
    
    I’m also calling on Congress: pass a law to make sure veterans devastated by toxic exposures in Iraq and Afghanistan finally get the benefits and comprehensive health care they deserve. 
    
    And fourth, let’s end cancer as we know it. 
    
    This is personal to me and Jill, to Kamala, and to so many of you. 
    
    Cancer is the #2 cause of death in America–second only to heart disease.
    ----------------------------------------------------------------------------------------------------
    Document 12:
    
    Headaches. Numbness. Dizziness. 
    
    A cancer that would put them in a flag-draped coffin. 
    
    I know. 
    
    One of those soldiers was my son Major Beau Biden. 
    
    We don’t know for sure if a burn pit was the cause of his brain cancer, or the diseases of so many of our troops. 
    
    But I’m committed to finding out everything we can. 
    
    Committed to military families like Danielle Robinson from Ohio. 
    
    The widow of Sergeant First Class Heath Robinson.
    ----------------------------------------------------------------------------------------------------
    Document 13:
    
    He will never extinguish their love of freedom. He will never weaken the resolve of the free world. 
    
    We meet tonight in an America that has lived through two of the hardest years this nation has ever faced. 
    
    The pandemic has been punishing. 
    
    And so many families are living paycheck to paycheck, struggling to keep up with the rising cost of food, gas, housing, and so much more. 
    
    I understand.
    ----------------------------------------------------------------------------------------------------
    Document 14:
    
    When we invest in our workers, when we build the economy from the bottom up and the middle out together, we can do something we haven’t done in a long time: build a better America. 
    
    For more than two years, COVID-19 has impacted every decision in our lives and the life of the nation. 
    
    And I know you’re tired, frustrated, and exhausted. 
    
    But I also know this.
    ----------------------------------------------------------------------------------------------------
    Document 15:
    
    My plan to fight inflation will lower your costs and lower the deficit. 
    
    17 Nobel laureates in economics say my plan will ease long-term inflationary pressures. Top business leaders and most Americans support my plan. And here’s the plan: 
    
    First – cut the cost of prescription drugs. Just look at insulin. One in ten Americans has diabetes. In Virginia, I met a 13-year-old boy named Joshua Davis.
    ----------------------------------------------------------------------------------------------------
    Document 16:
    
    And soon, we’ll strengthen the Violence Against Women Act that I first wrote three decades ago. It is important for us to show the nation that we can come together and do big things. 
    
    So tonight I’m offering a Unity Agenda for the Nation. Four big things we can do together.  
    
    First, beat the opioid epidemic. 
    
    There is so much we can do. Increase funding for prevention, treatment, harm reduction, and recovery.
    ----------------------------------------------------------------------------------------------------
    Document 17:
    
    My plan will not only lower costs to give families a fair shot, it will lower the deficit. 
    
    The previous Administration not only ballooned the deficit with tax cuts for the very wealthy and corporations, it undermined the watchdogs whose job was to keep pandemic relief funds from being wasted. 
    
    But in my administration, the watchdogs have been welcomed back. 
    
    We’re going after the criminals who stole billions in relief money meant for small businesses and millions of Americans.
    ----------------------------------------------------------------------------------------------------
    Document 18:
    
    So let’s not abandon our streets. Or choose between safety and equal justice. 
    
    Let’s come together to protect our communities, restore trust, and hold law enforcement accountable. 
    
    That’s why the Justice Department required body cameras, banned chokeholds, and restricted no-knock warrants for its officers.
    ----------------------------------------------------------------------------------------------------
    Document 19:
    
    I understand. 
    
    I remember when my Dad had to leave our home in Scranton, Pennsylvania to find work. I grew up in a family where if the price of food went up, you felt it. 
    
    That’s why one of the first things I did as President was fight to pass the American Rescue Plan.  
    
    Because people were hurting. We needed to act, and we did. 
    
    Few pieces of legislation have done more in a critical moment in our history to lift us out of crisis.
    ----------------------------------------------------------------------------------------------------
    Document 20:
    
    And we will, as one people. 
    
    One America. 
    
    The United States of America. 
    
    May God bless you all. May God protect our troops.
    

## Doing compression with LLMLingua
Now let’s wrap our base retriever with a `ContextualCompressionRetriever`, using `LLMLinguaCompressor` as a compressor.


```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMLinguaCompressor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)
```

    Document 1:
    
    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
    ----------------------------------------------------------------------------------------------------
    Document 2:
    
    . Numbness. Dizziness.A that would them in a-draped coffin. I One of those soldiers was my Biden We don’t know for sure if a burn pit the cause of brain, or the diseases of so many of our troops But I’m committed to finding out everything we can Committed to military families like Danielle Robinson from Ohio The widow of First Robinson.
    ----------------------------------------------------------------------------------------------------
    Document 3:
    
    <ref#> let� Or between equal Let’ to protect, restore law accountable  why the Justice Department cameras bannedhold and restricted its officers. <
    ----------------------------------------------------------------------------------------------------
    Document 4:
    
    <# The Sergeant Class Combat froms widow us toBut burn pits ravaged Heath’s lungs and body. 
    Danielle says Heath was a fighter to the very end.
    

## QA generation with LLMLingua

We can see what it looks like to use this in the generation step now


```python
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)
```


```python
chain.invoke({"query": query})
```




    {'query': 'What did the president say about Ketanji Brown Jackson',
     'result': "The President mentioned that Ketanji Brown Jackson is one of the nation's top legal minds and will continue Justice Breyer's legacy of excellence."}




```python

```
