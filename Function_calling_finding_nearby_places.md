# Function calling for nearby places: Leveraging the Bing Maps Local Search api

This notebook is centered around the integration of the Bing Maps Local Search api to enhance location-based searches. Please note that while we focus on the Bing Maps Local Search api in this instance, there are numerous other APIs you could explore and apply in a similar fashion.

We'll explore the application of three main components:

- Bing Maps Local Search api: This API provides real-time data about nearby places. It factors in various data points such as ratings, types of venues, costs, and more from the locations around you.

- Function calling: A single command such as \"I'm hungry\" or \"I want to visit a museum\" activates the function which invokes the Bing Maps Local Search api to identify suitable venues.

This notebook introduces two primary use cases:

- API integration with function calling: Understand how to integrate and call Bing Maps Local Search api effectively to source real-time data of various places using function calling.

Please note that while this system is highly versatile, its effectiveness may vary based on user preferences and available place data. For the purposes of this notebook, the customer data is fake and the location is hardcoded. "

## Setup

Bing Maps Local Search api

To use the Bing Maps Local Search api, you'll need:

- Microsoft Account: If you don't already have one, you will need to create a Microsoft account.

- Bing Maps Local Search api API Key: The API key is a unique identifier that is used to authenticate requests associated with your project. You can get your API key from the [Bing Maps Dev Center](https://www.bingmapsportal.com/Application). 


## Installation
Install the Azure Open AI SDK using the below command.


```csharp
#r "nuget: Azure.AI.OpenAI, 1.0.0-beta.14"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Azure.AI.OpenAI, 1.0.0-beta.14</span></li></ul></div></div>



```csharp
#r "nuget:Microsoft.DotNet.Interactive.AIUtilities, 1.0.0-beta.24129.1"

using Microsoft.DotNet.Interactive;
using Microsoft.DotNet.Interactive.AIUtilities;
```

## Run this cell, it will prompt you for the apiKey, endPoint, gtpDeployment, and bingMap


```csharp
var azureOpenAIKey = await Kernel.GetPasswordAsync("Provide your OPEN_AI_KEY");

// Your endpoint should look like the following https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/
var azureOpenAIEndpoint = await Kernel.GetInputAsync("Provide the OPEN_AI_ENDPOINT");

// Enter the deployment name you chose when you deployed the model.
var gptDeployment = await Kernel.GetInputAsync("Provide GPT  deployment name");

var bingMapsKey = await Kernel.GetPasswordAsync("Provide your BING_MAPS_KEY");
```

### Import namesapaces and create an instance of `OpenAiClient` using the `azureOpenAIEndpoint` and the `azureOpenAIKey`


```csharp
using Azure;
using Azure.AI.OpenAI;
```


```csharp
OpenAIClient client = new (new Uri(azureOpenAIEndpoint), new AzureKeyCredential(azureOpenAIKey.GetClearTextPassword()));
```


```csharp
public enum PlaceType{
    Bars,
    BarsGrillsAndPubs,
    BelgianRestaurants,
    BreweriesAndBrewPubs,
    BritishRestaurants,
    BuffetRestaurants,
    CafeRestaurants,
    CaribbeanRestaurants,
    ChineseRestaurants,
    CocktailLounges,
    CoffeeAndTea,
    Delicatessens,
    DeliveryService,
    Diners,
    DiscountStores,
    Donuts,
    FastFood,
    FrenchRestaurants,
    FrozenYogurt,
    GermanRestaurants,
    GreekRestaurants,
    Grocers,
    Grocery,
    HawaiianRestaurants,
    HungarianRestaurants,
    IceCreamAndFrozenDesserts,
    IndianRestaurants,
    ItalianRestaurants,
    JapaneseRestaurants,
    Juices,
    KoreanRestaurants,
    LiquorStores,
    MexicanRestaurants,
    MiddleEasternRestaurants,
    Pizza,
    PolishRestaurants,
    PortugueseRestaurants,
    Pretzels,
    Restaurants,
    RussianAndUkrainianRestaurants,
    Sandwiches,
    SeafoodRestaurants,
    SpanishRestaurants,
    SportsBars,
    SteakHouseRestaurants,
    Supermarkets,
    SushiRestaurants,
    TakeAway,
    Taverns,
    ThaiRestaurants,
    TurkishRestaurants,
    VegetarianAndVeganRestaurants,
    VietnameseRestaurants,
    AmusementParks,
    Attractions,
    Carnivals,
    Casinos,
    LandmarksAndHistoricalSites,
    MiniatureGolfCourses,
    MovieTheaters,
    Museums,
    Parks,
    SightseeingTours,
    TouristInformation,
    Zoos,
    AntiqueStores,
    Bookstores,
    CDAndRecordStores,
    ChildrensClothingStores,
    CigarAndTobaccoShops,
    ComicBookStores,
    DepartmentStores,
    FleaMarketsAndBazaars,
    FurnitureStores,
    HomeImprovementStores,
    JewelryAndWatchesStores,
    KitchenwareStores,
    MallsAndShoppingCenters,
    MensClothingStores,
    MusicStores,
    OutletStores,
    PetShops,
    PetSupplyStores,
    SchoolAndOfficeSupplyStores,
    ShoeStores,
    SportingGoodsStores,
    ToyAndGameStores,
    VitaminAndSupplementStores,
    WomensClothingStores,
    BanksAndCreditUnions,
    Hospitals,
    HotelsAndMotels,
    Parking
}
```

This ustility functions helps translating a `GptFunction` to the `ChatCompletionsFunctionToolDefinition` type


```csharp
using System.Text.Json;

public ChatCompletionsFunctionToolDefinition CreateToolDefinition(GptFunction function)
{
    var functionDefinition = new ChatCompletionsFunctionToolDefinition{
        Name = function.Name,
    };
    var json = JsonDocument.Parse(function.JsonSignature.ToString()).RootElement;
    functionDefinition.Parameters = BinaryData.FromString(json.GetProperty("parameters").ToString());
    return functionDefinition;
}
```

## Create a `GptFunction`

A `GptFunction` is an object  that can be used to create a `ChatCompletionOption` and later execute the logic in the `delegate` passed in the constructor.

Let's create ones that will take as input a location and a placeType. They will be used to search a matching location using the [Bing Map Local Search Api](https://learn.microsoft.com/bingmaps/rest-services/locations/local-search).


```csharp
using System.Web;
var findFunction = GptFunction.Create("find",async (string currentLocation, PlaceType placeType) =>{
    var apiKey = bingMapsKey.GetClearTextPassword();
    var httpClient = new System.Net.Http.HttpClient();
    var url = new Uri($"https://dev.virtualearth.net/REST/v1/LocalSearch/?query={HttpUtility.UrlEncode(currentLocation)}&type={placeType}&key={apiKey}");
 
    var response = await httpClient.GetAsync(url);
    return await response.Content.ReadAsStringAsync();   
},enumsAsString: true);

```

### Create ProvideReccomendations
This function will use the `GptFunction` and `ChatGPT` to answer a user question by calling the function with parameters generated by the LLM.



```csharp
public async Task<string> ProvideReccomendations(string userQuestion){
    var response = await client.GetChatCompletionsAsync(new ChatCompletionsOptions{
        Messages={
                    new ChatRequestSystemMessage("You are a sophisticated AI assistant, a specialist in user intent detection and interpretation. Your task is to perceive and respond to the user's needs, even when they're expressed in an indirect or direct manner. You excel in recognizing subtle cues: for example, if a user states they are 'hungry', you should assume they are seeking nearby dining options such as a restaurant or a cafe. If they indicate feeling 'tired', 'weary', or mention a long journey, interpret this as a request for accommodation options like hotels or guest houses. However, remember to navigate the fine line of interpretation and assumption: if a user's intent is unclear or can be interpreted in multiple ways, do not hesitate to politely ask for additional clarification. Use only values from the nums in the functions."),
                    new ChatRequestUserMessage(userQuestion)
        },
        Tools = { CreateToolDefinition(findFunction) },
        DeploymentName = gptDeployment,
    });

    var toolCall =  response.Value.Choices[0].Message.ToolCalls.Cast<ChatCompletionsFunctionToolCall>().First(tc => tc.Name == findFunction.Name);

    toolCall.Arguments.Display();
    
    var results = await  ((Task<string>) findFunction.Execute(toolCall.Arguments));
    return results;
}
```


```csharp
var response = await ProvideReccomendations("I am hungry in Seattle, actually Capitol Hill. At the moment I would appreaciate something local and cheap. Maybe a pub? Don't know what is the best to go for. I am open to any idea, What do you suggest?");


```


    {
      "currentLocation": "Capitol Hill, Seattle",
      "placeType": "BarsGrillsAndPubs"
    }


Now the full loop


```csharp
// setup the call exposing the tool to llm
public async Task<string> Ask(string userQuestion){
    var messages = new List<ChatRequestMessage>{
                    new ChatRequestSystemMessage("You are a sophisticated AI assistant, a specialist in user intent detection and interpretation. Your task is to perceive and respond to the user's needs, even when they're expressed in an indirect or direct manner. You excel in recognizing subtle cues: for example, if a user states they are 'hungry', you should assume they are seeking nearby dining options such as a restaurant or a cafe. If they indicate feeling 'tired', 'weary', or mention a long journey, interpret this as a request for accommodation options like hotels or guest houses. However, remember to navigate the fine line of interpretation and assumption: if a user's intent is unclear or can be interpreted in multiple ways, do not hesitate to politely ask for additional clarification. Use only values from the nums in the functions."),
                    new ChatRequestUserMessage(userQuestion)
    };

    // create the initial request
    var request = new ChatCompletionsOptions{
       
        Tools = { CreateToolDefinition(findFunction) },
        DeploymentName = gptDeployment,
    };

    foreach (var chatRequestMessage in messages)
    {
        request.Messages.Add(chatRequestMessage);
    }

    var response = await client.GetChatCompletionsAsync(request);

    // we need to track the messages and tool calls to re-evaluate the response
    while (response.Value.Choices.Any(c => c.FinishReason == CompletionsFinishReason.ToolCalls))
    {
        var needToEval = false;
        
        
        foreach (var choice in response.Value.Choices)
        {

            // proceed with the llm tool calls
            if (choice?.Message?.ToolCalls is { } toolCalls)
            {
                // there are tool calls in the response, build assistant message and execute the tool
                var assistantMessage = new ChatRequestAssistantMessage(choice.Message.Content);
                foreach (var messageToolCall in choice.Message.ToolCalls)
                {
                    assistantMessage.ToolCalls.Add(messageToolCall);
                }

                assistantMessage.FunctionCall = choice.Message.FunctionCall;

                // add the assistant message to the messages, this one contains the tool calls that the llm wants to execute
                messages.Add(assistantMessage);

                // execute the tool calls   
                foreach (var toolCall in toolCalls)
                {
                    if (toolCall is ChatCompletionsFunctionToolCall functionToolCall)
                    {
                        // make sure we are executing the right tool!!
                        if (functionToolCall.Name == findFunction.Name)
                        {
                            var id = functionToolCall.Id;
                            var toolResult = await  ((Task<string>) findFunction.Execute(functionToolCall.Arguments));
                        
                            messages.Add(new ChatRequestToolMessage(
                                toolResult as string ?? JsonSerializer.Serialize(toolResult), functionToolCall.Id));

                            // we need to re-evaluate the response with the new messages, this will enable the llm to continue
                            needToEval = true;
                        }
                    }
                }
            }
        }

        if (needToEval)
        {
            // we can now re-evaluate the response with the new messages and get final response
           new ChatCompletionsOptions{
                Tools = { CreateToolDefinition(findFunction) },
                DeploymentName = gptDeployment,
            };

            foreach (var chatRequestMessage in messages)
            {
                request.Messages.Add(chatRequestMessage);
            }

            response = await client.GetChatCompletionsAsync(request);
        }
    }

    // no more tool call to handle, we can now display the final response
    var message = response.Value.Choices[0].Message.Content;
    Console.WriteLine(message);

    return message;
}
```


```csharp
await Ask("I am hungry in Seattle, actually Capitol Hill. At the moment I would appreaciate something local and cheap. Maybe a pub? Don't know what is the best to go for. I am open to any idea, What do you suggest?");
```

    Here are some local and affordable pubs in Capitol Hill, Seattle that you might like:
    
    1. [Capitol Hill Comedy / Bar](https://comedyslashbar.com/): 210 Broadway E, Seattle, WA, 98125. Contact: (206) 390-9152
    
    2. [Barrio Mexican Kitchen & Bar](https://www.barriorestaurant.com/): 1420 12th Ave, Seattle, WA, 98122. Contact: (206) 588-8105
    
    3. [Mezcaleria Oaxaca Capitol Hill](https://mercadoluna.com/mezcaleria-oaxaca/): 422 E Pine St, Seattle, WA, 98122. Contact: (206) 324-0506
    
    4. [Capitol Cider](https://www.capitolcider.com/): 818 E Pike St, Seattle, WA, 98122. Contact: (206) 397-3564
    
    5. [GOLD BAR](https://goldbarseattle.com/Info): 1418 E OLIVE Way, Seattle, WA, 98122. Contact: (206) 402-3473
    
    Please note that prices and availability may vary. Make sure to check their websites or contact them for more information. Enjoy your meal!
    

