## Pinecone Vector Database Semantic Search Example

This is a simple Semantic Search applicaiton using Pinecone vector database
to store embeddings. Pinecode offers a free starter-index as a community edition.
Actually, it's not that bad, as it allows you to index 100K vector embeddings.

Only a single index is allowed in the community edition. The diagram below
shows the process and flow, and the steps in the notebook illustrate simple
cronological steps to create a semantic search application. 

<img src="images/pinecone_vectordb.png">

[source](https://www.pinecone.io/learn/vector-database/)


```python
import os
from pinecone import Pinecone, PodSpec
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
from tqdm.auto import tqdm
```

Utility function to extract relevant information returned
by Pinecone search querty and displaying them.


```python
def extract_and_print_matches(results):
    for result in results['matches']:
        print(f"Score  : {round(result['score'], 2)}")
        print(f"Matches: {result['metadata']['text']}")
        print('-' * 50)
```

### Step 1: Load the IMDB dataset, only use first 50k samples


```python
dataset = load_dataset("imdb", split='train[:50000]')
print(dataset[:1])
```

    {'text': ['I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it\'s not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn\'t have much of a plot.'], 'label': [0]}
    

#### Examine the data


```python
reviews = []
for record in dataset['text']:
    reviews.extend(record.split('\n'))
reviews = list(set(reviews))
print('\n'.join(reviews[:2]))
print('-' * 50)
print(f'Number of reviews: {len(reviews)}')
```

    I sat down to watch this movie with my friends with very low expectations. My expectations were no where near low enough. I honestly could not tell what genre this movie was from watching it, and if it was a comedy, the humor was completely missed. The plot was nonexistent and the acting was horrendous. My friends and I managed to watch approximately 30 to 40 minutes of this film before we turned it off and promptly begged the video store to take it back. I do NOT recommend this movie to anyone unless you are purposely trying to watch the worst movies of all time. I honestly don't know how this film lasted more than a day in theatres and moreover I can not understand why anyone would willing watch it, considering not only it's very uninteresting title but also the lack of any famous actors/actresses in it's cast. This review is not a joke and I honestly think this could possibly be the worst movie ever made. It's certainly the worst movie I've ever had to sit through.
    Michael Callan plays a smarmy photographer who seems, nonetheless, to be regarded as a perfect "catch" by any woman that runs across him; could this have anything to do with the fact that he also co-produced the film? He's a "hero" whom it's very difficult to empathize with, so the movie is in trouble right from the start. However, it's troubles don't end there. It has the production values of a TV-movie (check out that head made of clay or something, near the end), and the ending cheats in a way that I can't reveal, in case anyone wants to see the movie (highly unlikely). Let's just say that the killer knows more than we were let to know he knows. (*1/2)
    --------------------------------------------------
    Number of reviews: 24904
    

### Step 2: Instantiate the sentence transformer embedding model


```python
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```


```python
# Try encoding a sample review
embeddings = model.encode(reviews[0:1])
print(f"vector shape: {embeddings.shape}; vector length:{len(embeddings[0])}")
```

    vector shape: (1, 384); vector length:384
    

### Step 3: Set up Pinecone environment. 

Use the `.env` file to load the Pinecone API key and the environment name, 
which is "gcp-starter." In this case, the GCP starter environment is a community edition of Pinecone available for free.


```python
 _ = load_dotenv(find_dotenv())
api_key = os.getenv("PINECONE_API_KEY")
if api_key is None:
    raise ValueError("Please set the PINECONE_API_KEY environment")

pc = Pinecone(
    api_key=api_key,
    environment="gcp-starter",
    spec=PodSpec(environment="gcp-starter")
) 
```


```python
# check if an index exists in Pinecone
index_name = "starter-index"
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]
existing_indexes
```




    []




```python
# Delete index if one already exists, since Pincone allows only 
# one index in the starter community edition
if index_name in existing_indexes:
    print(f"Index {index_name} already exists. Deleting it.")
    pc.delete_index(index_name)
```

### Step 3: Create an index
And then get a handle or pointer to it


```python
print(f"Creating a new index {index_name}...")
pc.create_index(name=index_name,
        metric="cosine",
        dimension=embeddings.shape[1],
        spec=PodSpec(environment="gcp-starter")
)
# Connect or get a pointer to the index
pindex = pc.Index(index_name)
```

    Creating a new index starter-index...
    

### Step 4: Upsert data into the index.

In our case, we are going to create IMDB review embeddings in batches
and upsert each batch, along with `ids` and `metadata`.


```python
print("Upserting the embeddings into the index...")
batch_size = 500
for i in tqdm(range(0, len(reviews), batch_size)):

    # create minibatches from the dataset
    i_end = min(i+batch_size, len(reviews))
    
    # create IDs each batch
    ids = [str(x) for x in range(i, i_end)]
    
    # create metadata batch as text and insert review
    metadatas = [{'text': text} for text in reviews[i:i_end]]
    batch = reviews[i:i+batch_size]

    # create an embedding for the batch
    embeddings = model.encode(batch)
    records = zip(ids, embeddings, metadatas)
        
    # upsert each batch to Pinecone
    print(f"Upserting {i} to {i_end} records...")

    pindex.upsert(vectors=records)
```

    Upserting the embeddings into the index...
    


      0%|          | 0/50 [00:00<?, ?it/s]


    Upserting 0 to 500 records...
    Upserting 500 to 1000 records...
    Upserting 1000 to 1500 records...
    Upserting 1500 to 2000 records...
    Upserting 2000 to 2500 records...
    Upserting 2500 to 3000 records...
    Upserting 3000 to 3500 records...
    Upserting 3500 to 4000 records...
    Upserting 4000 to 4500 records...
    Upserting 4500 to 5000 records...
    Upserting 5000 to 5500 records...
    Upserting 5500 to 6000 records...
    Upserting 6000 to 6500 records...
    Upserting 6500 to 7000 records...
    Upserting 7000 to 7500 records...
    Upserting 7500 to 8000 records...
    Upserting 8000 to 8500 records...
    Upserting 8500 to 9000 records...
    Upserting 9000 to 9500 records...
    Upserting 9500 to 10000 records...
    Upserting 10000 to 10500 records...
    Upserting 10500 to 11000 records...
    Upserting 11000 to 11500 records...
    Upserting 11500 to 12000 records...
    Upserting 12000 to 12500 records...
    Upserting 12500 to 13000 records...
    Upserting 13000 to 13500 records...
    Upserting 13500 to 14000 records...
    Upserting 14000 to 14500 records...
    Upserting 14500 to 15000 records...
    Upserting 15000 to 15500 records...
    Upserting 15500 to 16000 records...
    Upserting 16000 to 16500 records...
    Upserting 16500 to 17000 records...
    Upserting 17000 to 17500 records...
    Upserting 17500 to 18000 records...
    Upserting 18000 to 18500 records...
    Upserting 18500 to 19000 records...
    Upserting 19000 to 19500 records...
    Upserting 19500 to 20000 records...
    Upserting 20000 to 20500 records...
    Upserting 20500 to 21000 records...
    Upserting 21000 to 21500 records...
    Upserting 21500 to 22000 records...
    Upserting 22000 to 22500 records...
    Upserting 22500 to 23000 records...
    Upserting 23000 to 23500 records...
    Upserting 23500 to 24000 records...
    Upserting 24000 to 24500 records...
    Upserting 24500 to 24904 records...
    


```python
# Check the index stats
print(pindex.describe_index_stats())
```

    {'dimension': 384,
     'index_fullness': 0.24904,
     'namespaces': {'': {'vector_count': 24904}},
     'total_vector_count': 24904}
    

### Step 5: Query the Pinecone indexed vector database

Let's create a long review. I'm a consumate fan of John Le Carre, so 
let's see if we can find any matching reviews.


```python
query = """This is a classic espionage thriller. I loved the movie, it was capitivating, 
            the plot brilliant, based on a true events during the cold war, the characters were 
            well developed, and their actions unpredictable yet justifiable, given circumstances.
            The cast was amazing, and the direction of plot cogent and very well thought out. 
            Recommended to everyone if you love clock and dagger, twists and turns of cold war drama 
            and betrayals, and if you relish how John Le Carre spins his plots 
            in his absorbing novels on cold war espionage tales of spooks and crooks, 
            you shall throughly enjoy this one!"""

```


```python
# create an embedding for the query
query_embedding = model.encode(query).tolist()

# search our index created above
results = pindex.query(vector=query_embedding, top_k=5,
                include_values=False, include_metadata=True)
```


```python
print("Top 5 results for the query:")
```

    Top 5 results for the query:
    


```python
extract_and_print_matches(results)
```

    Score  : 0.63
    Matches: Every now and then there gets released this movie no one has ever heard of and got shot in a very short time with very little money and resource but everybody goes crazy about and turns out to be a surprisingly great one. This also happened in the '50's with quite a few little movies, that not a lot of people have ever heard of. There are really some unknown great surprising little jewels from the '50's that are worth digging out. "Panic in the Streets" is another movie like that that springs to the mind. Both are movies that aren't really like the usual genre flicks from their time and are also made with limited resources.<br /><br />I was really surprised at how much I ended up liking this movie. It was truly a movie that got better and better as it progressed. Like all 'old' movies it tends to begin sort of slow but once you get into the story and it's characters you're in for a real treat with this movie.<br /><br />The movie has a really great story that involves espionage, though the movie doesn't start of like that. It begins as this typical crime-thriller with a touch of film-noir to it. But "Pickup on South Street" just isn't really a movie by the numbers so it starts to take its own directions pretty soon on. It ensures that the movie remains a surprising but above all also really refreshing one to watch.<br /><br />I also really liked the characters within this movie. None of them are really good guys and they all of their flaws and weaknesses. Really humane. It also especially features a great performance from Thelma Ritter, who even received a well deserved Oscar nomination for. It has really got to be one of the greatest female roles I have ever seen.<br /><br />Even despite its somewhat obvious low budget this is simply one great, original, special little movie that deserves to be seen by more!<br /><br />10/10
    --------------------------------------------------
    Score  : 0.59
    Matches: This is a fascinating account of the hunt for the Soviet Union's first known serial killer. I had tuned in, just expecting a half-decent TV movie, but found myself drawn by the compelling way the story was told. As others have said, there is much to admire here that is sadly lacking in many big screen releases.<br /><br />Much of the credit must go to Chris Gerolmo, whose intelligent screenplay and direction draw the viewer in, until it is impossible not to feel emotionally involved. The acting by the whole cast is also superb, especially that of the two leads, Stephen Rea and Donald Sutherland. Their convincing portrayals give their character arcs a great deal of credibility, and the scene where they have their first committee meeting after Perestroika is genuinely touching.<br /><br />If you prefer your crime films with a bit more depth and a little less sheen, I strongly recommend you look out for 'Citizen X'.<br /><br />
    --------------------------------------------------
    Score  : 0.59
    Matches: This is yet another gritty and compelling film directed by Sam Fuller in the early 1950s. This minimalist and fast-working director has something unusual for his earlier films--a cast with some stars. Richard Widmark, Jean Peters and Richard Kiley star in this film about a group of Communist agents who are trying to sneak secrets out of America--and they'll stop at nothing to succeed.<br /><br />The film starts with Peters on a subway car being watched by federal agents. They know she is a link in a long espionage chain. Unknown to everyone is the wild card in the equation--a small-time pickpocket (Widmark) is also on the train and he manages to steal the secrets that Peters is carrying. Widmark thinks it's just another purse he's ransacked--only later does he realize the seriousness of what he's stolen. Now it's Widmark on his own--with Commies and the FBI hot on his trail.<br /><br />Widmark and the rest are exceptional and the film is gripping from start to finish. Although she didn't get top billing, a special mention should be made of Thelma Ritter. This supporting actress had perhaps the performance of her lifetime as a stool pigeon. Seldom was she given this much of a chance to act and I was impressed by her ability to play a broken down and sad old lady.<br /><br />As far as the script and directing go, they are very good--but with one small exception. At first, I loved the way Widmark and Peters interacted. It's one of the few times on film you'll see a woman punched square in the mouth! Now THAT'S tough. Later, inexplicably, they become amazingly close--too close to be believable. Still, with so much great drama and such an effective Noir-like film, this can be overlooked. See this film.
    --------------------------------------------------
    Score  : 0.59
    Matches: I thought this film would be a lot better then it was. It sounded like a spoof off of the spy gener, and the start of it reminded me of Pleasantvil, but this film came up short.<br /><br />The plot is just to ridiculous. The KGB and Soviet Union in Russia have started up a spy school to teach their spies' how to act like Americans, but the town they set up in it for training is a bit dated, so they grab two yanks from the US to spice things up. I don't know, but this seems just to out there. It gets really odd when next to no one in this all Russian town speaks in a Russian accent. Someone screwed up in the casting job.<br /><br />Also, for a comedy this is painfully dry. There is one, two funny spots tops, and they are nothing to sing and dance about. The film in the end will likely put you to sleep.<br /><br />And, as a twisted punch in the face, this film is so pro the US it makes me sick. The movie keeps on saying again and again, the US is God and Russia is the devil. This is the kind of smear campaign that was done against the Japanese in World War 2. It's films like these that makes everyone think that the US is full of itself.<br /><br />This gets a 4 out of 10, and I'm being kind. It should really get a one, but the dance scene was funny, but then again it dragged far to long to be really funny.
    --------------------------------------------------
    Score  : 0.58
    Matches: This movie was excellent. It details the struggle between a committed detective against the dedicated ignorance of the corrupted communist regime in Russia during the 80's. I give this movie high marks for it's no-holds-barred look into the birth and development of forensic investigation in a globally isolated (thanks to the "Regime") community. This is a graphic movie. It presents an unsensationalized picture of violence and it's tragic remains. Nothing is "candy-coated" with overdone blood or gore to separate us from the cruel reality on the screen. This movie is based on Russian serial killer Andrei Chikatilo. I'm familiar enough with the true story to have a very deep appreciation for how real they kept the film. It's not a comedy, but for those who appreciate dry and dark humor, this movie is a must-see.
    --------------------------------------------------
    

### Step 6 (optional): Remove the index
Since only a single index is allowed, we might as well remove it at the end,
and recreate a new one next time if needed or use a different dataset to index


```python
# Delete the index
print(f"Deleting the index {index_name}...")
pc.delete_index(index_name)
print("Done!")
```

    Deleting the index starter-index...
    Done!
    
