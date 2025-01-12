```
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Document Processing with Gemini

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/document-processing/document_processing.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fuse-cases%2Fdocument-processing%2Fdocument_processing.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>       
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/document-processing/document_processing.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/document-processing/document_processing.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Holt Skinner](https://github.com/holtskinner), [Drew Gillson](https://github.com/drewgillson) |

## Overview

In today's information-driven world, the volume of digital documents generated daily is staggering. From emails and reports to legal contracts and scientific papers, businesses and individuals alike are inundated with vast amounts of textual data. Extracting meaningful insights from these documents efficiently and accurately has become a paramount challenge.

Document processing involves a range of tasks, including text extraction, classification, summarization, and translation, among others. Traditional methods often rely on rule-based algorithms or statistical models, which may struggle with the nuances and complexities of natural language.

Generative AI offers a promising alternative to understand, generate, and manipulate text using natural language prompting. Gemini on Vertex AI allows these models to be used in a scalable manner through:

- [Vertex AI Studio](https://cloud.google.com/generative-ai-studio) in the Cloud Console
- [Vertex AI REST API](https://cloud.google.com/vertex-ai/docs/reference/rest)
- [Vertex AI SDK for Python](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk-ref)
- [Other client libraries](https://cloud.google.com/vertex-ai/docs/start/client-libraries)

This notebook focuses on using the **Vertex AI SDK for Python** to call the Gemini API in Vertex AI with the Gemini 1.5 Flash model.

For more information, see the [Generative AI on Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) documentation.


### Objectives

In this tutorial, you will learn how to use the Gemini API in Vertex AI with the Vertex AI SDK for Python to process PDF documents.

You will complete the following tasks:

- Install the Vertex AI SDK for Python
- Use the Gemini API in Vertex AI to interact with Gemini 1.5 Flash (`gemini-1.5-flash`) model:
  - Extract structured entities from an unstructured document
  - Classify document types
  - Combine classification and entity extraction into a single workflow
  - Summarize documents


### Costs

This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.


## Getting Started


### Install Vertex AI SDK for Python



```
%pip install --upgrade --user --quiet google-cloud-aiplatform
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
# Restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the following cell to authenticate your environment. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).



```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
# Define project information
PROJECT_ID = "YOUR_PROJECT_ID"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI
import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries



```
import json

from IPython.display import Markdown, display_pdf
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
```

### Load the Gemini 1.5 Flash model

Gemini 1.5 Flash (`gemini-1.5-flash`) is a multimodal model that supports multimodal prompts. You can include text, image(s), and video in your prompt requests and get text or code responses.


```
model = GenerativeModel(
    "gemini-1.5-flash",
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    },
)
# This Generation Config sets the model to respond in JSON format.
generation_config = GenerationConfig(
    temperature=0.0, response_mime_type="application/json"
)
```

### Define helper function

Define helper function to print the multimodal prompt


```
PDF_MIME_TYPE = "application/pdf"


def print_multimodal_prompt(contents: list) -> None:
    """
    Given contents that would be sent to Gemini,
    output the full multimodal prompt for ease of readability.
    """
    for content in contents:
        if not isinstance(content, Part):
            print(content)
        elif content.inline_data:
            display_pdf(content.inline_data.data)
        elif content.file_data:
            gcs_url = (
                "https://storage.googleapis.com/"
                + content.file_data.file_uri.replace("gs://", "").replace(" ", "%20")
            )
            print(f"PDF URL: {gcs_url}")


# Send Google Cloud Storage Document to Vertex AI
def process_document(
    prompt: str,
    file_uri: str,
    mime_type: str = PDF_MIME_TYPE,
    generation_config: GenerationConfig | None = None,
    print_prompt: bool = False,
    print_raw_response: bool = False,
) -> str:
    # Load file directly from Google Cloud Storage
    file_part = Part.from_uri(
        uri=file_uri,
        mime_type=mime_type,
    )

    # Load contents
    contents = [file_part, prompt]

    # Send to Gemini
    response = model.generate_content(contents, generation_config=generation_config)

    if print_prompt:
        print("-------Prompt--------")
        print_multimodal_prompt(contents)

    if print_raw_response:
        print("\n-------Raw Response--------")
        print(response)

    return response.text
```

## Entity Extraction

[Named Entity Extraction](https://en.wikipedia.org/wiki/Named-entity_recognition) is a technique of Natural Language Processing to identify specific fields and values from unstructured text. For example, you can find key-value pairs from a filled out form, or get all of the important data from an invoice categorized by the type.

### Extract entities from an invoice

In this example, you will use a sample invoice and get all of the information in JSON format.

This is the prompt to be sent to Gemini along with the PDF document. Feel free to edit this for your specific use case.


```
invoice_extraction_prompt = """You are a document entity extraction specialist. Given a document, your task is to extract the text value of the following entities:
{
	"amount_paid_since_last_invoice": "",
	"carrier": "",
	"currency": "",
	"currency_exchange_rate": "",
	"delivery_date": "",
	"due_date": "",
	"freight_amount": "",
	"invoice_date": "",
	"invoice_id": "",
	"line_items": [
		{
			"amount": "",
			"description": "",
			"product_code": "",
			"purchase_order": "",
			"quantity": "",
			"unit": "",
			"unit_price": ""
		}
	],
	"net_amount": "",
	"payment_terms": "",
	"purchase_order": "",
	"receiver_address": "",
	"receiver_email": "",
	"receiver_name": "",
	"receiver_phone": "",
	"receiver_tax_id": "",
	"receiver_website": "",
	"remit_to_address": "",
	"remit_to_name": "",
	"ship_from_address": "",
	"ship_from_name": "",
	"ship_to_address": "",
	"ship_to_name": "",
	"supplier_address": "",
	"supplier_email": "",
	"supplier_iban": "",
	"supplier_name": "",
	"supplier_payment_ref": "",
	"supplier_phone": "",
	"supplier_registration": "",
	"supplier_tax_id": "",
	"supplier_website": "",
	"total_amount": "",
	"total_tax_amount": "",
	"vat": [
		{
			"amount": "",
			"category_code": "",
			"tax_amount": "",
			"tax_rate": "",
			"total_amount": ""
		}
	]
}

- The JSON schema must be followed during the extraction.
- The values must only include text found in the document
- Do not normalize any entity value.
- If an entity is not found in the document, set the entity value to null.
"""
```


```
# Download a PDF from Google Cloud Storage
! gsutil cp "gs://cloud-samples-data/generative-ai/pdf/invoice.pdf" ./invoice.pdf
```


```
# Load file bytes
with open("invoice.pdf", "rb") as f:
    file_part = Part.from_data(data=f.read(), mime_type="application/pdf")

# Load contents
contents = [file_part, invoice_extraction_prompt]

# Send to Gemini with GenerationConfig
response = model.generate_content(contents, generation_config=generation_config)
```


```
print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Raw Response--------")
print(response.text)
```

This response can then be parsed as JSON into a Python dictionary for use in other applications.


```
print("\n-------Parsed Entities--------")
json_object = json.loads(response.text)
print(json_object)
```

You can see that Gemini extracted all of the relevant fields from the document.

### Extract entities from a payslip

Let's try with another type of document, a payslip or paystub.


```
payslip_extraction_prompt = """You are a document entity extraction specialist. Given a document, your task is to extract the text value of the following entities:
{
"earning_item": [
{
"earning_rate": "",
"earning_hours": "",
"earning_type": "",
"earning_this_period": ""
}
],
"direct_deposit_item": [
{
"direct_deposit": "",
"employee_account_number": ""
}
],
"current_deduction": "",
"ytd_deduction": "",
"employee_id": "",
"employee_name": "",
"employer_name": "",
"employer_address": "",
"federal_additional_tax": "",
"federal_allowance": "",
"federal_marital_status": "",
"gross_earnings": "",
"gross_earnings_ytd": "",
"net_pay": "",
"net_pay_ytd": "",
"ssn": "",
"pay_date": "",
"pay_period_end": "",
"pay_period_start": "",
"state_additional_tax": "",
"state_allowance": "",
"state_marital_status": "",
"tax_item": [
{
"tax_this_period": "",
"tax_type": "",
"tax_ytd": ""
}
]
}

- The JSON schema must be followed during the extraction.
- The values must only include text strings found in the document.
- Generate null for missing entities.
"""
```


```
response_text = process_document(
    payslip_extraction_prompt,
    "gs://cloud-samples-data/generative-ai/pdf/earnings_statement.pdf",
    generation_config=generation_config,
    print_prompt=True,
)
```


```
print("\n-------Parsed Entities--------")
json_object = json.loads(response_text)
print(json_object)
```

## Document Classification

Document classification is the process for identifying the type of document. For example, invoice, W-2, receipt, etc.

In this example, you will use a sample tax form (W-9) and get the specific type of document from a specified list.


```
classification_prompt = """You are a document classification assistant. Given a document, your task is to find which category the document belongs to from the list of document categories provided below.

 1040_2019
 1040_2020
 1099-r
 bank_statement
 credit_card_statement
 expense
 form_1120S_2019
 form_1120S_2020
 investment_retirement_statement
 invoice
 paystub
 property_insurance
 purchase_order
 utility_statement
 w2
 w9
 driver_license

Which category does the above document belong to? Answer with one of the predefined document categories only.
"""
```


```
response_text = process_document(
    classification_prompt,
    "gs://cloud-samples-data/generative-ai/pdf/w9.pdf",
    print_prompt=True,
)
```


```
print("\n-------Document Classification--------")
print(response_text)
```

You can see that Gemini successfully categorized the document.

### Chaining Classification and Extraction

These techniques can also be chained together to extract any number of document types. For example, if you have multiple types of documents to process, you can send each document to Gemini with a classification prompt, then based on that output, you can write logic to decide which extraction prompt to use.


```
generic_document_prompt = """You are a document entity extraction specialist. Given a document, your task is to extract the text value of the following entities:

{}

- The JSON schema must be followed during the extraction.
- The values must only include text found in the document
- Do not normalize any entity value.
- If an entity is not found in the document, set the entity value to null.
"""

w2_extraction_prompt = generic_document_prompt.format(
    """
{
    "ControlNumber": "",
    "EIN": "",
    "EmployeeAddress_City": "",
    "EmployeeAddress_State": "",
    "EmployeeAddress_StreetAddressOrPostalBox": "",
    "EmployeeAddress_Zip": "",
    "EmployeeName_FirstName": "",
    "EmployeeName_LastName": "",
    "EmployerAddress_City": "",
    "EmployerAddress_State": "",
    "EmployerAddress_StreetAddressOrPostalBox": "",
    "EmployerAddress_Zip": "",
    "EmployerName": "",
    "EmployerStateIdNumber_Line1": "",
    "FederalIncomeTaxWithheld": "",
    "FormYear": "",
    "MedicareTaxWithheld": "",
    "MedicareWagesAndTips": "",
    "SocialSecurityTaxWithheld": "",
    "SocialSecurityWages": "",
    "StateIncomeTax_Line1": "",
    "StateWagesTipsEtc_Line1": "",
    "State_Line1": "",
    "WagesTipsOtherCompensation": "",
    "a_Code": "",
    "a_Value": "",
}
"""
)

drivers_license_prompt = generic_document_prompt.format(
    """
{
    "Address": "",
    "Date Of Birth": "",
    "Document Id": "",
    "Expiration Date": "",
    "Family Name": "",
    "Given Names": "",
    "Issue Date": "",
}
"""
)

# Map classification types to extraction prompts
classification_to_prompt = {
    "invoice": invoice_extraction_prompt,
    "w2": w2_extraction_prompt,
    "driver_license": drivers_license_prompt,
}
```


```
gcs_uris = [
    "gs://cloud-samples-data/documentai/SampleDocuments/US_DRIVER_LICENSE_PROCESSOR/dl3.pdf",
    "gs://cloud-samples-data/documentai/SampleDocuments/INVOICE_PROCESSOR/google_invoice.pdf",
    "gs://cloud-samples-data/documentai/SampleDocuments/FORM_W2_PROCESSOR/2020FormW-2.pdf",
]

for gcs_uri in gcs_uris:
    print(f"\nFile: {gcs_uri}\n")

    # Send to Gemini with Classification Prompt
    doc_classification = process_document(classification_prompt, gcs_uri).strip()

    print(f"Document Classification: {doc_classification}")

    # Get Extraction prompt based on Classification
    extraction_prompt = classification_to_prompt.get(doc_classification)

    if not extraction_prompt:
        print(f"Document does not belong to a specified class {doc_classification}")
        continue

    # Send to Gemini with Extraction Prompt
    extraction_response_text = process_document(
        extraction_prompt,
        gcs_uri,
        generation_config=generation_config,
        print_prompt=True,
    ).strip()

    print("\n-------Extracted Entities--------")
    json_object = json.loads(extraction_response_text)
    print(json_object)
```

## Document Question Answering

Gemini can be used to answer questions about a document.

This example answers a question about the Transformer model paper "Attention is all you need".


```
qa_prompt = """What is attention in the context of transformer models? Give me the answer first, followed by an explanation."""
```


```
# Send Q&A Prompt to Gemini
response_text = process_document(
    qa_prompt,
    "gs://cloud-samples-data/generative-ai/pdf/1706.03762v7.pdf",
)

print(f"Answer: {response_text}")
```

## Document Summarization

Gemini can also be used to summarize or paraphrase a document's contents. Your prompt can specify how detailed the summary should be or specific formatting, such as bullet points or paragraphs.


```
summarization_prompt = """You are a very professional document summarization specialist. Given a document, your task is to provide a detailed summary of the content of the document.

If it includes images, provide descriptions of the images.
If it includes tables, extract all elements of the tables.
If it includes graphs, explain the findings in the graphs.
Do not include any numbers that are not mentioned in the document.
"""
```


```
# Send Summarization Prompt to Gemini
response_text = process_document(
    summarization_prompt,
    "gs://cloud-samples-data/generative-ai/pdf/fdic_board_meeting.pdf",
)

print(f"Summarization: {response_text}")
```

## Table parsing from documents

Gemini can parse contents of a table and return it in a structured format, such as HTML or markdown.


```
table_extraction_prompt = """What is the html code of the table in this document?"""
```


```
# Send Table Extraction Prompt to Gemini
response_text = process_document(
    table_extraction_prompt,
    "gs://cloud-samples-data/generative-ai/pdf/salary_table.pdf",
)
display(Markdown(response_text))
```

## Document Translation

Gemini can translate documents between languages. This example translates meeting notes from English into French and Spanish.


```
translation_prompt = """Translate the first paragraph into French and Spanish. Label each paragraph with the target language."""
```


```
# Send Translation Prompt to Gemini
response_text = process_document(
    translation_prompt,
    "gs://cloud-samples-data/generative-ai/pdf/fdic_board_meeting.pdf",
)

print(response_text)
```

## Document Comparison

Gemini can compare and contrast the contents of multiple documents. This example finds the changes in the IRS Form 1040 between 2013 and 2023.

Note: when working with multiple documents, the order can matter and should be specified in your prompt.


```
comparison_prompt = """The first document is from 2013, the second one from 2023. How did the standard deduction evolve?"""
```


```
# Send Comparison Prompt to Gemini
file_part1 = Part.from_uri(
    uri="gs://cloud-samples-data/generative-ai/pdf/form_1040_2013.pdf",
    mime_type=PDF_MIME_TYPE,
)

file_part2 = Part.from_uri(
    uri="gs://cloud-samples-data/generative-ai/pdf/form_1040_2023.pdf",
    mime_type=PDF_MIME_TYPE,
)

# Load contents
contents = [file_part1, file_part2, comparison_prompt]

# Send to Gemini
response = model.generate_content(contents)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("-------Output--------")
print(response.text)
```
