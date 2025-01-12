# Autmatically labelling Github issues


```csharp
#r "nuget: Azure.AI.OpenAI, 1.0.0-beta.14"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Azure.AI.OpenAI, 1.0.0-beta.14</span></li></ul></div></div>



```csharp
#r "nuget: Octokit, 9.0.0"
#r "nuget: Octokit.Reactive, 9.0.0"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Octokit, 9.0.0</span></li><li><span>Octokit.Reactive, 9.0.0</span></li></ul></div></div>



```csharp
#r "nuget:Microsoft.DotNet.Interactive.AIUtilities, 1.0.0-beta.24129.1"
```


```csharp
using Azure;
using Azure.AI.OpenAI;
using Microsoft.DotNet.Interactive;
using Microsoft.DotNet.Interactive.AIUtilities;
using Octokit;
```


```csharp
var azureOpenAIKey = await Kernel.GetPasswordAsync("Provide your OPEN_AI_KEY");
var azureOpenAIEndpoint = await Kernel.GetInputAsync("Provide the OPEN_AI_ENDPOINT");
var chatDeployment = await Kernel.GetInputAsync("Provide chat deployment name");
var embeddingDeployment = await Kernel.GetInputAsync("Provide chat embedding name");
```

## Access to GitHub
You will need access token with rights to query and update issues.


```csharp
var githubKey = await Kernel.GetPasswordAsync("Provide your Github api key");
var repoName = await Kernel.GetInputAsync("Provide repo");
var org = await Kernel.GetInputAsync("Provide org");
```


```csharp
OpenAIClient openAIClient = new (new Uri(azureOpenAIEndpoint), new AzureKeyCredential(azureOpenAIKey.GetClearTextPassword()));
```


```csharp
var options = new ApiOptions();
var gitHubClient = new GitHubClient(new ProductHeaderValue("notebook"));

if (!string.IsNullOrEmpty(githubKey.GetClearTextPassword())) {
    Console.WriteLine("Using GitHub API token");
    var tokenAuth = new Credentials(githubKey.GetClearTextPassword());
    gitHubClient.Credentials = tokenAuth;
} else {
    Console.WriteLine("Using anonymous GitHub API");
}
```

    Using GitHub API token



```csharp
var allLabels = await gitHubClient.Issue.Labels.GetAllForRepository(org, repoName);
```


```csharp
allLabels.DisplayTable();
```


<table><thead><tr><td><span>Id</span></td><td><span>Url</span></td><td><span>Name</span></td><td><span>NodeId</span></td><td><span>Color</span></td><td><span>Description</span></td><td><span>Default</span></td></tr></thead><tbody><tr><td><div class="dni-plaintext"><pre>4773058988</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Accessibility</td><td>Area-Accessibility</td><td>LA_kwDODgj8L88AAAABHH8ZrA</td><td>5319e7</td><td>Relating to UI accessibility issues</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>5511620623</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-API</td><td>Area-API</td><td>LA_kwDODgj8L88AAAABSISoDw</td><td>5319e7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>4803279709</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Auth</td><td>Area-Auth</td><td>LA_kwDODgj8L88AAAABHkw7XQ</td><td>5319e7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>2094097123</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Automation</td><td>Area-Automation</td><td>MDU6TGFiZWwyMDk0MDk3MTIz</td><td>5319e7</td><td>Relating to non-interactive execution of notebooks and scripts</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>4084666155</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Azure%20Data%20Studio</td><td>Area-Azure Data Studio</td><td>LA_kwDODgj8L87zdw8r</td><td>5319e7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>1907988999</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Build%20&amp;%20Infrastructure</td><td>Area-Build &amp; Infrastructure</td><td>MDU6TGFiZWwxOTA3OTg4OTk5</td><td>5319e7</td><td>Relating to this repo&#39;s build and infrastructure</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>2065909664</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-C%23</td><td>Area-C#</td><td>MDU6TGFiZWwyMDY1OTA5NjY0</td><td>5319e7</td><td>Specific to C#</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>2110504572</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Docker</td><td>Area-Docker</td><td>MDU6TGFiZWwyMTEwNTA0NTcy</td><td>5319e7</td><td>Specific to docker</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>1801690166</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Documentation</td><td>Area-Documentation</td><td>MDU6TGFiZWwxODAxNjkwMTY2</td><td>5319e7</td><td>Improvements or additions to documentation</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>1835518355</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-F%23</td><td>Area-F#</td><td>MDU6TGFiZWwxODM1NTE4MzU1</td><td>5319e7</td><td>Specific to F#</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>2272303722</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Formatting</td><td>Area-Formatting</td><td>MDU6TGFiZWwyMjcyMzAzNzIy</td><td>5319e7</td><td>Data and object formatting as HTML and plaintext</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>4436670966</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Getting%20Started</td><td>Area-Getting Started</td><td>LA_kwDODgj8L88AAAABCHI59g</td><td>5319E7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>6047789224</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-HTTP</td><td>Area-HTTP</td><td>LA_kwDODgj8L88AAAABaHnwqA</td><td>5319E7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>1836998938</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Installation</td><td>Area-Installation</td><td>MDU6TGFiZWwxODM2OTk4OTM4</td><td>5319e7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>2094233835</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-JavaScript%20HTML%20CSS</td><td>Area-JavaScript HTML CSS</td><td>MDU6TGFiZWwyMDk0MjMzODM1</td><td>5319e7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>2071777991</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Jupyter%20Kernel</td><td>Area-Jupyter Kernel</td><td>MDU6TGFiZWwyMDcxNzc3OTkx</td><td>5319e7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>4750560388</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Jupyter%20sub-kernels</td><td>Area-Jupyter sub-kernels</td><td>LA_kwDODgj8L88AAAABGyfMhA</td><td>5319E7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>1985413182</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Language%20Services</td><td>Area-Language Services</td><td>MDU6TGFiZWwxOTg1NDEzMTgy</td><td>5319e7</td><td>IntelliSense, LSP, and related</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>4688323202</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Localization%20and%20Globalization</td><td>Area-Localization and Globalization</td><td>LA_kwDODgj8L88AAAABF3Iigg</td><td>5319e7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td><div class="dni-plaintext"><pre>2518739187</pre></div></td><td>https://api.github.com/repos/dotnet/interactive/labels/Area-Messaging%20/%20scheduling%20/%20comms</td><td>Area-Messaging / scheduling / comms</td><td>MDU6TGFiZWwyNTE4NzM5MTg3</td><td>5319e7</td><td></td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td colspan="7"><i>(41 more)</i></td></tr></tbody></table><style>
.dni-code-hint {
    font-style: italic;
    overflow: hidden;
    white-space: nowrap;
}
.dni-treeview {
    white-space: nowrap;
}
.dni-treeview td {
    vertical-align: top;
    text-align: start;
}
details.dni-treeview {
    padding-left: 1em;
}
table td {
    text-align: start;
}
table tr { 
    vertical-align: top; 
    margin: 0em 0px;
}
table tr td pre 
{ 
    vertical-align: top !important; 
    margin: 0em 0px !important;
} 
table th {
    text-align: start;
}
</style>


The code below is using the Octokit library, which is a .NET client for interacting with the GitHub API.

The first part of the code is creating a new instance of `RepositoryIssueRequest` named `last6Months`. This object is used to specify the parameters for a request to fetch issues from a GitHub repository. In this case, the `Filter` property is set to `IssueFilter.All`, which means that the request will return all issues regardless of their state (open, closed, etc.). The `Since` property is set to a date that is six months prior to the current date (`DateTimeOffset.UtcNow.Subtract(TimeSpan.FromDays(30*6))`). This means that the request will return only the issues that have been updated in the last six months.

The second part of the code is making an asynchronous request to fetch all issues for a specific repository. The `GetAllForRepository` method of the `Issue` class in the `gitHubClient` object is used to make this request. The `org` and `repoName` variables are used to specify the organization and the name of the repository from which to fetch the issues. The method returns a list of all issues in the specified repository. The `await` keyword is used to wait for the method to complete execution before moving on to the next line of code. This is necessary because the method is asynchronous, meaning it runs in the background and may not complete immediately.


```csharp
var last6Months = new RepositoryIssueRequest
{
    Filter = IssueFilter.All,
    Since = DateTimeOffset.UtcNow.Subtract(TimeSpan.FromDays(30*6))
};
var allIssues = await gitHubClient.Issue.GetAllForRepository(org, repoName);
```


```csharp
if(allIssues.Count(i => i.Labels.Count == 0) == 0){
    "No issues without labels, no need to proceed!".Display();
}
```


```csharp
public record IssueWithEmbedding(Issue Issue, float[] Embedding);
```

With a `foreach` loop that iterates over chunks of issues. The `Chunk(16)` method is used to divide the `allIssues` collection into smaller collections (or chunks) of 16 issues each. This is done to manage memory usage when processing large collections.

Inside the loop, for each chunk of issues, the code first concatenates the title and body of each issue and truncates the resulting string to a maximum of 8191 tokens using the `tokenizer.TruncateByTokenCount(s,8191)` method. The resulting strings are then converted to an array.

Next, the code makes an asynchronous request to an AI service (likely OpenAI) to generate embeddings for the text of each issue in the chunk. The `GetEmbeddingsAsync` method of the `openAIClient` object is used to make this request. The method takes an instance of `EmbeddingsOptions` as a parameter, which specifies the deployment of the embedding model and the text to be embedded.

The response from the AI service is then processed to extract the embeddings. The `Value.Data` property of the response contains the embeddings, which are converted to arrays and stored in the `embeddings` variable.

Finally, the code creates a new instance of `IssueWithEmbedding` for each issue in the chunk, associating each issue with its corresponding embedding. These instances are added to the `issuesWithEmbeddings` collection for further processing.


```csharp
var issuesWithEmbeddings = new List<IssueWithEmbedding>();

var tokenizer = await Tokenizer.CreateAsync(TokenizerModel.ada2);

foreach(var chunk in allIssues.Chunk(16)){
    var text = chunk.Select(i => i.Title + "\n" + i.Body).Select(s => tokenizer.TruncateByTokenCount(s,8191)).ToArray();
    var response = await openAIClient.GetEmbeddingsAsync(new EmbeddingsOptions(embeddingDeployment, text));

    var embeddings = response.Value.Data.Select(e => e.Embedding.ToArray()).ToArray();
    for(var i = 0; i < chunk.Length; i++){
        issuesWithEmbeddings.Add(new IssueWithEmbedding(chunk[i], embeddings[i]));
    }
}
```


    Azure.RequestFailedException: Requests to the Embeddings_Create Operation under Azure OpenAI API version 2023-09-01-preview have exceeded call rate limit of your current OpenAI S0 pricing tier. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.


    Status: 429 (Too Many Requests)


    ErrorCode: 429


    


    Content:


    {"error":{"code":"429","message": "Requests to the Embeddings_Create Operation under Azure OpenAI API version 2023-09-01-preview have exceeded call rate limit of your current OpenAI S0 pricing tier. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit."}}


    


    Headers:


    x-rate-limit-reset-tokens: REDACTED


    x-ms-client-request-id: 6286698b-1edb-47b5-a2f9-fdf0a1e53ed7


    apim-request-id: REDACTED


    Strict-Transport-Security: REDACTED


    X-Content-Type-Options: REDACTED


    policy-id: REDACTED


    x-ms-region: REDACTED


    x-ratelimit-remaining-requests: REDACTED


    Date: Wed, 06 Dec 2023 12:49:31 GMT


    Content-Length: 312


    Content-Type: application/json


    


       at Azure.Core.HttpPipelineExtensions.ProcessMessageAsync(HttpPipeline pipeline, HttpMessage message, RequestContext requestContext, CancellationToken cancellationToken)


       at Azure.AI.OpenAI.OpenAIClient.GetEmbeddingsAsync(EmbeddingsOptions embeddingsOptions, CancellationToken cancellationToken)


       at Submission#14.<<Initialize>>d__0.MoveNext()


    --- End of stack trace from previous location ---


       at Microsoft.CodeAnalysis.Scripting.ScriptExecutionState.RunSubmissionsAsync[TResult](ImmutableArray`1 precedingExecutors, Func`2 currentExecutor, StrongBox`1 exceptionHolderOpt, Func`2 catchExceptionOpt, CancellationToken cancellationToken)


The following cell is filtering the `issuesWithEmbeddings` collection into two separate lists based on the number of labels each issue has.

The first line of the code is creating a new list named `noLabels`. This list is populated with the issues from the `issuesWithEmbeddings` collection that have no labels. This is determined by the lambda expression `i => i.Issue.Labels.Count == 0` in the `Where` method, which checks if the `Labels` property of the `Issue` object has a `Count` of 0.

The second line of the code is creating another list named `labelled`. This list is populated with the issues from the `issuesWithEmbeddings` collection that have one or more labels. This is determined by the lambda expression `i => i.Issue.Labels.Count > 0` in the `Where` method, which checks if the `Labels` property of the `Issue` object has a `Count` greater than 0.

In both cases, the `ToList` method is used to convert the filtered enumerable collections to lists.


```csharp
var noLabels = issuesWithEmbeddings.Where(i => i.Issue.Labels.Count == 0).ToList();
var labelled = issuesWithEmbeddings.Where(i => i.Issue.Labels.Count > 0).ToList();
```


```csharp
public class LabelWithEmbeddings{
    public Label Label {get;set;}
    public float[] Embedding {get;set;}
    public List<IssueWithEmbedding> Issues {get;init ;} = new();
}
```


```csharp
var labelsWithEmbeddings = new List<LabelWithEmbeddings>();
```


```csharp
foreach(var label in allLabels.Where(e => e.Name.Contains("Area-"))){
    var issues = labelled.Where(i => i.Issue.Labels.Any(l => l.Name == label.Name)).ToList();
    if(issues.Count > 0){
        var labelWithEmbeddings = new LabelWithEmbeddings{
            Label = label,
            Issues = issues
        };
       labelsWithEmbeddings.Add(labelWithEmbeddings);
    }
}
```


```csharp
foreach(var label in labelsWithEmbeddings){
    label.Embedding = label.Issues.Select(i => i.Embedding).Centroid();
}
```


```csharp
var suggestions = new Dictionary<IssueWithEmbedding, LabelWithEmbeddings[]>();
foreach(var issue in noLabels){
    var suggestedLabels = labelsWithEmbeddings.ScoreBySimilarityTo(issue.Embedding, new CosineSimilarityComparer<float[]>(f => f), l => l.Embedding)
    .OrderByDescending( s => s.Score)
    .Where(s => s.Score > 0.85)
    .Take(5)
    .ToArray();
    suggestions.Add(issue, suggestedLabels.Select(s => s.Value).ToArray());
}
```

Then we suggest labels for GitHub issues based on their embeddings. 

The code starts by creating a new dictionary named `suggestions`. The keys in this dictionary are instances of `IssueWithEmbedding` and the values are arrays of `LabelWithEmbeddings`.

Next, the code enters a `foreach` loop that iterates over each issue in the `noLabels` list. For each issue, the code calculates the similarity between the issue's embedding and the embeddings of all labels using the `ScoreBySimilarityTo` method. This method likely calculates the cosine similarity, a measure of similarity between two non-zero vectors, between the issue's embedding and each label's embedding. The `CosineSimilarityComparer<float[]>(f => f)` is used to specify how to calculate the cosine similarity.

The resulting scores are then ordered in descending order, filtered to include only scores greater than 0.85, and the top 5 scores are selected. This means that the code is suggesting the top 5 labels that have a similarity score greater than 0.85 with the issue's embedding.

Finally, the issue and its suggested labels are added to the `suggestions` dictionary. The `Select(s => s.Key).ToArray()` part of the code is used to extract the labels (which are the keys in the score dictionary) and convert them to an array.


```csharp
suggestions.Select(e => new {
    Issue = e.Key.Issue.Title,
    SuggestedLabels = e.Value.Select(l => l.Label.Name).ToArray()

}).DisplayTable();
```


<table><thead><tr><td><span>Issue</span></td><td><span>SuggestedLabels</span></td></tr></thead><tbody><tr><td>Issues with input prompt docs</td><td><div class="dni-plaintext"><pre>[ Area-F#, Area-PowerShell, Area-Packages and Extensions, Area-Documentation, Area-Polyglot Notebooks Extension ]</pre></div></td></tr><tr><td>.net interactive stuck loading nuget packagees</td><td><div class="dni-plaintext"><pre>[ Area-Packages and Extensions, Area-Installation, Area-F#, Area-Documentation, Area-PowerShell ]</pre></div></td></tr><tr><td>Wont run under .NET 8</td><td><div class="dni-plaintext"><pre>[ Area-Installation, Area-Packages and Extensions, Area-Documentation, Area-PowerShell, Area-F# ]</pre></div></td></tr><tr><td>Issues once .net 8 is installed.</td><td><div class="dni-plaintext"><pre>[ Area-Installation, Area-Packages and Extensions, Area-F#, Area-PowerShell, Area-Documentation ]</pre></div></td></tr><tr><td>Outputs from dotnet-repl are not displayed in by VS Code extension</td><td><div class="dni-plaintext"><pre>[ Area-Polyglot Notebooks Extension, Area-F#, Area-Jupyter Kernel, Area-Documentation, Area-PowerShell ]</pre></div></td></tr><tr><td>Printing values from R Type Provider prints garbled output</td><td><div class="dni-plaintext"><pre>[ Area-Formatting, Area-F# ]</pre></div></td></tr><tr><td>Polyglot Notebook: [DevExE2E][Regression] The kernelName and language show as csharp in the created Untitled-1.ipynb contents.</td><td><div class="dni-plaintext"><pre>[ Area-Polyglot Notebooks Extension, Area-F#, Area-JavaScript HTML CSS, Area-Jupyter Kernel, Area-Installation ]</pre></div></td></tr><tr><td>Polyglot Notebook: [DevExE2E][Regression][intermittent]When running the cells one by one, test can&#39;t be stopped and always hang in running status.</td><td><div class="dni-plaintext"><pre>[ Area-Polyglot Notebooks Extension, Area-Installation, Area-JavaScript HTML CSS, Area-PowerShell, Area-Packages and Extensions ]</pre></div></td></tr><tr><td>Polyglot Notebook: [DevExE2E][Regression] After stopping the cell, no variable is shown in POLYGLOT NOTEBOOK: VARIABLES page. </td><td><div class="dni-plaintext"><pre>[ Area-Polyglot Notebooks Extension, Area-JavaScript HTML CSS, Area-Installation, Area-PowerShell, Area-Jupyter Kernel ]</pre></div></td></tr><tr><td>Failed to connect to python kernel on mac</td><td><div class="dni-plaintext"><pre>[ Area-Python, Area-Jupyter Kernel, Area-Installation, Area-PowerShell, Area-Packages and Extensions ]</pre></div></td></tr><tr><td>Don&#39;t use markdown preview to show .net version alert</td><td><div class="dni-plaintext"><pre>[ Area-Polyglot Notebooks Extension, Area-Installation, Area-Language Services, Area-Documentation, Area-JavaScript HTML CSS ]</pre></div></td></tr><tr><td>Command &#39;Polyglot Notebook: Create new blank notebook&#39; resulted in an error</td><td><div class="dni-plaintext"><pre>[ Area-Installation, Area-PowerShell, Area-Packages and Extensions, Area-Documentation, Area-Polyglot Notebooks Extension ]</pre></div></td></tr><tr><td>[pre-release] Polygot notebook for PowerShell adds bad characters in output cell</td><td><div class="dni-plaintext"><pre>[ Area-Polyglot Notebooks Extension, Area-Installation, Area-PowerShell, Area-F#, Area-Formatting ]</pre></div></td></tr><tr><td>find a better name for the text/plain+summary MIME type</td><td></td></tr><tr><td>Python implementation for Polyglot kernel APIs</td><td></td></tr></tbody></table><style>
.dni-code-hint {
    font-style: italic;
    overflow: hidden;
    white-space: nowrap;
}
.dni-treeview {
    white-space: nowrap;
}
.dni-treeview td {
    vertical-align: top;
    text-align: start;
}
details.dni-treeview {
    padding-left: 1em;
}
table td {
    text-align: start;
}
table tr { 
    vertical-align: top; 
    margin: 0em 0px;
}
table tr td pre 
{ 
    vertical-align: top !important; 
    margin: 0em 0px !important;
} 
table th {
    text-align: start;
}
</style>

