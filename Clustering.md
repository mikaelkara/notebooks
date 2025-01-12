## K-means Clustering in C# using OpenAI
We use a simple k-means algorithm to demonstrate how clustering can be done. Clustering can help discover valuable, hidden groupings within the data. The dataset is created in the [Get_embeddings_from_dataset Notebook](Get_embeddings_from_dataset.ipynb) Notebook.


```csharp
#r "nuget: Azure.AI.OpenAI, 1.0.0-beta.14"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Azure.AI.OpenAI, 1.0.0-beta.14</span></li></ul></div></div>



```csharp
#r "nuget:Microsoft.DotNet.Interactive.AIUtilities, 1.0.0-beta.24129.1"
```


```csharp
using Microsoft.DotNet.Interactive;
using Microsoft.DotNet.Interactive.AIUtilities;
using Azure;
using Azure.AI.OpenAI;
```

## Run this cell, it will prompt you for the apiKey, endPoint, and chatDeployment


```csharp
var azureOpenAIKey = await Kernel.GetPasswordAsync("Provide your OPEN_AI_KEY");

// Your endpoint should look like the following https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/
var azureOpenAIEndpoint = await Kernel.GetInputAsync("Provide the OPEN_AI_ENDPOINT");

// Enter the deployment name you chose when you deployed the model.
var chatDeployment = await Kernel.GetInputAsync("Provide chat deployment name");
```


```csharp
OpenAIClient client = new (new Uri(azureOpenAIEndpoint), new AzureKeyCredential(azureOpenAIKey.GetClearTextPassword()));
```


```csharp
#r "nuget: Microsoft.ML,  3.0.0"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Microsoft.ML, 3.0.0</span></li></ul></div></div>



```csharp

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
```


```csharp
public class DataRow{
    public string ProducIt {get;set;} 
    public string UserId {get;set;} 
    public int Score {get;set;} 
    public string Summary {get;set;} 
    public string Text {get;set;} 
    public int TokenCount {get;set;} 
    [VectorType(1536)]
    public float[] Embedding {get;set;} 
};
```


```csharp
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;

var filePath = Path.Combine("..","..","..","Data","fine_food_reviews_with_embeddings_1k.json");

var foodReviewsData = JsonSerializer.Deserialize<DataRow[]>(File.ReadAllText(filePath));
```

### 1. Find the clusters using K-means
We show the simplest use of K-means. You can pick the number of clusters that fits your use case best.

First, a new instance of the `MLContext` class is created. 

Next, the `LoadFromEnumerable` method of the `Data` property of the `context` object is called to load the `foodReviewsData` into an `IDataView` object, which is a flexible, efficient way of describing tabular data (numeric and text).

A pipeline is then defined using the `Clustering.Trainers.KMeans` method of the `context` object. This method creates a new K-Means clustering trainer. The first argument is the name of the feature column (in this case, "Embedding"), and the `numberOfClusters` parameter is set to 4, indicating that the algorithm should group the data into 4 clusters.

The `Fit` method is then called on the pipeline, passing in the `idv` object. This trains the model on the loaded data and returns the trained model.

The `Transform` method is then called on the `model` object, passing in the `idv` object. This applies the trained model to the loaded data, assigning each data point to a cluster.

Finally, the `GetClusterCentroids` method is called on the `Model` property of the `model` object. This method retrieves the centroids of the clusters identified by the model. The centroids are stored in the `centroids` variable.



```csharp
var context = new MLContext();
var idv = context.Data.LoadFromEnumerable(foodReviewsData);
var pipeline =  context.Clustering.Trainers.KMeans("Embedding", numberOfClusters: 4);
var model = pipeline.Fit(idv);
var clusteredData = model.Transform(idv);

VBuffer<float>[] centroids = default;
model.Model.GetClusterCentroids(ref centroids, out var _);
```

### 2. Text samples in the clusters & naming the clusters
Let's show samples from each cluster. We'll use GPT to name the clusters, based on a random sample of 5 reviews from that cluster.
Iterating over the clusters' centroids we find the most relevant reviewes using `CosineSimilarityComparer`. The we randomly pick 5 for each cluster.


```csharp
var rnd = new Random(42);

var examples = centroids.Select(c => {
    var embedding = c.GetValues().ToArray();
    var samples = foodReviewsData
        .ScoreBySimilarityTo(embedding, new CosineSimilarityComparer<float[]>(v => v), r => r.Embedding )
        .OrderByDescending(e => e.Score)
        .Select(e => e.Value)
        .Take(200)
        .Shuffle()
        .Take(5);

    return new {
            CenstroidEmbedding = embedding,
            Reviews = samples
            };
    }
).ToArray();
```

Using the 5 random samples of each cluster we ask GPT for the common theme


```csharp
foreach (var example in examples)
{
    var prompt =
$"""
What do the following customer reviews have in common?
Customer reviews:
{string.Join("\n", example.Reviews.Select(r => $"{r.Score}, {r.Summary}: {r.Text}"))}
Theme:
""";
    var options= new ChatCompletionsOptions{
        Messages ={ new ChatRequestUserMessage(prompt)},
        Temperature = 0f,
        DeploymentName = chatDeployment
    };

    var response = await client.GetChatCompletionsAsync(options);
    var theme = response.Value.Choices.FirstOrDefault()?.Message?.Content;
    var text = new StringBuilder($"Cluster theme : {theme}");
    foreach (var review in example.Reviews)
    {
        text.AppendLine();
        text.AppendLine($"{review.Score}, {review.Summary}: {review.Text}");
    }
    text.ToString().Display();
}
```


    Cluster theme : The common theme among these customer reviews is that all of the products mentioned are considered to be healthy and good-tasting.
    5, Favorite chew toy!: This is the second one of these antlers that I've ordered (the first one was stolen by a friend's dog, because he liked it so much!).  My dog absolutely loves these antlers and carries them all over the house with her.  It is the perfect size for her (she's about 30 pounds) and provides hours of chewing time.  I highly recommend this to anyone with a dog who loves to chew and is looking for something other than a bone for them to chew on.
    
    5, back to nature oatmeal cookies: Back to nature cookies are great cookies to buy ...they taste great and are more healthier for you ...cant beat that ..they are awesome ..we love all of the varieties ...these are all the cookies i buy ..all the kids that come to my house they will eat the entire box in one sitting ..and i love the fact they are getting better ingredients without all the bad stuff
    
    5, Paws up!: We try different treats for our dogs and they seem to like the Holistic Select biscuits very much.  This has good quality ingredients just like their kibbles which our dogs like as well.
    
    5, Deee-licious...and NO heartburn!: Regular semolina shells give me heartburn that have killed 2 horses, bless their hearts.<br /><br />You won't have the spike in insulin, then drop, causing you to want to sleep for a few hours.<br /><br />The shells come in a large plastic bag inside a non-descript box.<br /><br />Freshness, thanks to putting them in baggies, was as good on the 1st day as on the last day, which, coincidentally, was 2.5 months later!<br /><br />Whole Foods would charge at least twice this much, probably more.<br /><br />No heartburn.  Plenty of shells at my beckon call.  Great price.<br /><br />You're welcome.
    
    5, Dogs Love Them!: My Maltese and Cavalier King Charles love these treats!  I feel good about feeding them a healthier treat.<br />Not made in China!




    Cluster theme : All of the customer reviews have a positive sentiment and praise the product or service being reviewed.
    5, Excellent Tea: This is the best tea i have had yet. It reminds me of a fine wine,very smooth and full bodied.I use milk and honey with it and i a cant get over how good the tea tastes.I finally found a great tea thanks to amazon and the reviews from consumers on products like these.
    
    5, Wow! Clean and fresh tasting: I was curious as to how this would taste - and I have to say they did a great job.  It taste clean and is refreshing, to top it off its also very good for me to drink.  I love natural products and this one fits into my narrow box of acceptability and it's taste was a pleasant surprise.<br />Jesus' Blessings and Peace
    
    5, Best coffee ever!: In my opinion this is the best coffee ever!  I've been drinking coffee for 50 plus years and this is what I serve to myself and friends.  However, I wish I could find this grind in a pound size, so I could make a full pot rather than just a single cup.
    
    5, Excellent Quality , Best Source of CLAs out there: This is a high quality Ghee , superior tasting and above all made from grass fed cows' milk and certified organic by the USDA.<br />  We use it over lentils ,  eggs , sauteed veggies , quinoa , rice , you name it and we use it.  Tried it on whole grain toast too...YUMMY !! Being from grass fed cows' milk , it is rich in CLAs and boosts immunity and energy and is known to fight and obesity , especially trimming the waistline. This is my second time buying this and I am a consumer for life. I also introduced it to my friends and they are digging the taste.<br />  I really hope AMAZON continues to sell this product  ( prime delivery really makes it all the more special ).<br /> Please try it for getting healthier and also the incredible taste.  Highly recommended.
    
    3, eh,,ok, but...: I love popcorn, and this product is good except for the fact that it does not and will not give you the taste that you are looking for that you get from movie theaters. Even with the matching seasoning salt, it doesn't do the justice. But its ok,just not buttery at all.




    Cluster theme : The common theme in these customer reviews is that the customers are satisfied with the flavor and overall quality of the product (tea or coffee) they purchased. However, there are some minor complaints or suggestions for improvement mentioned in a few of the reviews.
    5, So delicious!!: I love Green Mountain French Vanilla Cream coffee! The flavor is amazing and all I add in French Vanilla Creamer :)
    
    5, Great coffee, free shipping, great price: We love this coffee. We are not coffee connoisseurs but do enjoy a good cup of Joe. It works wonderfully in our french press and makes a nice, fresh, coffee shop grade cup.
    
    5, Deeeee-lish!: For far too long I was a devotee of the Starbucks roast.  This is so much better.  Rich, tasty, strong coffee without the BITTER bite.  Lovely crema.  Works great in my automatic espresso machine, too.  Love, love. love Lavazza.
    
    5, Tasty!: I wish these came in larger quantities, if I could have a cuppa everyday I would. So refreshing and tasty!
    
    2, Fair to midland: I was skeptical about ordering what with the trouble of the cups exploding, but I didn't have any of that trouble. I did find, however, that it just wasn't full-bodied enough for me.  It left me wanting.  I can usually brew an 8 oz cup from my K-cups but truthfully even brewing at 6 oz I still didn't get the rich flavor I craved. So I've been finishing up the box by brewing it at 2 or 4 oz and adding a different K-cup brand to the mug.  I'd stir clear of this one.




    Cluster theme : The common theme among these customer reviews is that they all discuss the quality or value of a food product.
    5, Licorice Pastels a Hit: Very delicious and enjoyed by many people.  These pastels do not melt and do not require special storage or handling.
    
    5, Party Peanuts: Great product for the price. Mix with the Asian rice crackers for an excellent snack.  Big container lasts a long time. Only lightly slighted. Peanuts are whole and large.
    
    5, Great for HS lunch: Great for HS lunch, kid enjoy as a snack also, will buy again. Salted chips are good too, tried them too.
    
    5, Great product, but too much of it.: This pearls were exactly what I was looking for and made delicious tapioca pudding. My only complaint is that you have to order so many boxes. What am I going to do with all that tapioca?? Two or three would've lasted me a long time.
    
    5, Great Cookies: These cookies are WONDERFUL!  So happy to be able to enjoy a sweet, GF Snickerdoodle!  The cookies are flavorful and soft.  I think even folks that don't follow a GF diet would enjoy.


