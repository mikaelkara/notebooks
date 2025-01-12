# Long Document Content Extraction
GPT-3 can help us extract key figures, dates or other bits of important content from documents that are too big to fit into the context window. One approach for solving this is to chunk the document up and process each chunk separately, before combining into one list of answers. 
In this notebook we'll run through this approach:
- Load in a long PDF and pull the text out
- Create a prompt to be used to extract key bits of information
- Chunk up our document and process each chunk to pull any answers out
- Combine them at the end
- This simple approach will then be extended to three more difficult questions
## Approach
- **Setup**: Take a PDF, a Formula 1 Financial Regulation document on Power Units, and extract the text from it for entity extraction. We'll use this to try to extract answers that are buried in the content.
- **Simple Entity Extraction**: Extract key bits of information from chunks of a document by:
    - Creating a template prompt with our questions and an example of the format it expects
    - Create a function to take a chunk of text as input, combine with the prompt and get a response
    - Run a script to chunk the text, extract answers and output them for parsing
- **Complex Entity Extraction**: Ask some more difficult questions which require tougher reasoning to work out


```csharp
#r "nuget: Azure.AI.OpenAI, 1.0.0-beta.14"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Azure.AI.OpenAI, 1.0.0-beta.12</span></li></ul></div></div>



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

### Install `itext` pacakge
Parsing pdfs using [iTextCore](https://itextpdf.com/products/itext-core) from nuget


```csharp
#r "nuget: itext7, 8.0.2"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>itext7, 8.0.2</span></li></ul></div></div>



```csharp
using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;
using iText.Kernel.Pdf.Canvas.Parser.Listener;
using System.IO;

public string ReadPdfFile(string fileName)
{
    var pages = new List<string>();
    using (PdfDocument pdfDoc = new PdfDocument(new PdfReader(fileName)))
    {            
        for (int i = 1; i <= pdfDoc.GetNumberOfPages(); i++)
        {
            PdfPage page = pdfDoc.GetPage(i);
            ITextExtractionStrategy strategy = new SimpleTextExtractionStrategy();
            string currentText = PdfTextExtractor.GetTextFromPage(page, strategy);
            pages.Add(currentText);
        }          
    }
    return string.Join(Environment.NewLine, pages);
}
```


```csharp
var filePath = Path.Combine("..","..","..","Data","fia_f1_power_unit_financial_regulations_issue_1_-_2022-08-16.pdf");
```


```csharp
var cleanText = ReadPdfFile(filePath).Replace("  ", " ").Replace("\n", "; ").Replace(';',' ');
```

## Simple Entity Extraction


```csharp
var tokenizer = await Tokenizer.CreateAsync(TokenizerModel.gpt35);
var chunks = tokenizer.ChunkByTokenCount(cleanText, 1000, true);
```


```csharp
var extractions = new List<string>();

foreach (var chunk in chunks)
{
    var prompt = 
$"""
Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output "Not specified".
When you extract a key piece of information, include the closest page number.
Use the following format:
0. Who is the author
1. What is the amount of the "Power Unit Cost Cap" in USD, GBP and EUR
2. What is the value of External Manufacturing Costs in USD
3. What is the Capital Expenditure Limit in USD

Document: \"\"\"{chunk}\"\"\"

0. Who is the author: Tom Anderson (Page 1)
1.
""";
    var options= new ChatCompletionsOptions{
        Messages ={ new ChatRequestUserMessage(prompt)},
        Temperature = 0f,
        DeploymentName = chatDeployment,
    };

    var response = await client.GetChatCompletionsAsync(options);
    var extraction = response.Value.Choices.FirstOrDefault()?.Message?.Content;
    extractions.AddRange(extraction.Split(new []{"\n"}, StringSplitOptions.RemoveEmptyEntries));
}
```

The `extractions` object is a collection that contains the extracted information. Each item in the collection is a string that represents a piece of information extracted from the document.

The `Where` method is a LINQ (Language Integrated Query) method that is used to filter the collection. The method takes a lambda expression `p => !p.Contains("Not specified")` as an argument. This expression is a function that takes an item from the collection (represented by `p`) and returns `true` if the item does not contain the string "Not specified", and `false` otherwise. In other words, this method filters out any items in the collection that contain the string "Not specified".

The `DisplayTable` method is then called on the filtered collection. This method displays the items in the collection as a table. Each item in the collection is displayed as a row in the table.

In summary, this code is used to filter out any items in the `extractions` collection that contain the string "Not specified", and then display the remaining items as a table.


```csharp
extractions.Where(p => !p.Contains("Not specified")).DisplayTable();
```


<table><thead><tr><td><span>value</span></td></tr></thead><tbody><tr><td>What is the amount of the &quot;Power Unit Cost Cap&quot; in USD, GBP and EUR: </td></tr><tr><td>- USD: $95,000,000 (Page 2)</td></tr><tr><td>- GBP: &#163;76,459 (Page 2)</td></tr><tr><td>- EUR: €90,210 (Page 2)</td></tr><tr><td>3. What is the Capital Expenditure Limit in USD: US Dollars 30,000,000 (Page 2)</td></tr></tbody></table><style>
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


## Complex Entity Extraction


```csharp
var extractions = new List<string>();

foreach (var chunk in chunks)
{
    var prompt = 
$"""
Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output "Not specified".
When you extract a key piece of information, include the closest page number.
Use the following format:
0. Who is the author
1. How is a Minor Overspend Breach calculated
2. How is a Major Overspend Breach calculated
3. Which years do these financial regulations apply to

Document: \"\"\"{chunk}\"\"\"\

0. Who is the author: Tom Anderson (Page 1)
1.
""";
    var options= new ChatCompletionsOptions{
        Messages ={ new ChatRequestUserMessage(prompt)},
        Temperature = 0f,
        DeploymentName = chatDeployment
    };

    var response = await client.GetChatCompletionsAsync(options);
    var extraction = response.Value.Choices.FirstOrDefault()?.Message?.Content;
    extractions.AddRange(extraction.Split(new []{"\n"}, StringSplitOptions.RemoveEmptyEntries));
}
```


```csharp
extractions.Where(p => !p.Contains("Not specified")).DisplayTable();
```


<table><thead><tr><td><span>value</span></td></tr></thead><tbody><tr><td>3. Which years do these financial regulations apply to: From 1 January 2023 onwards (Page 1)</td></tr><tr><td>3. Which years do these financial regulations apply to: 31 December 2023, 31 December 2024, 31 December 2025, 31 December 2026 and each subsequent Full Year Reporting Period (Page 2)</td></tr><tr><td>3. Which years do these financial regulations apply to: 16 August 2022 (Page 0)</td></tr><tr><td>3. Which years do these financial regulations apply to: 2022 (Page 24)</td></tr><tr><td>How is a Minor Overspend Breach calculated: A &quot;Minor Overspend Breach&quot; arises when a Power Unit Manufacturer&#39;s Relevant Costs exceed the Power Unit Cost Cap by less than 5% (Page 1)</td></tr><tr><td>2. How is a Major Overspend Breach calculated: A &quot;Material Overspend Breach&quot; arises when a Power Unit Manufacturer&#39;s Relevant Costs exceed the Power Unit Cost Cap by 5% or more (Page 1)</td></tr><tr><td>3. Which years do these financial regulations apply to: 2026 (Page 1)</td></tr><tr><td>3. Which years do these financial regulations apply to: 2023, 2024, 2025, 2026 onwards (Page 1)</td></tr><tr><td>3. Which years do these financial regulations apply to: 2026 to 2030 seasons (inclusive) (Page 2)</td></tr><tr><td>3. Which years do these financial regulations apply to: 2022 (Page 47)</td></tr></tbody></table><style>
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


## Consolidation

We've been able to extract the first two answers safely, while the third was confounded by the date that appeared on every page, though the correct answer is in there as well.

To tune this further you can consider experimenting with:
- A more descriptive or specific prompt
- If you have sufficient training data, fine-tuning a model to find a set of outputs very well
- The way you chunk your data - we have gone for 1000 tokens with no overlap, but more intelligent chunking that breaks info into sections, cuts by tokens or similar may get better results

However, with minimal tuning we have now answered 6 questions of varying difficulty using the contents of a long document, and have a reusable approach that we can apply to any long document requiring entity extraction. Look forward to seeing what you can do with this!
