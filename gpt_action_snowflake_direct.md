# GPT Actions - Snowflake direct
## Introduction

This page provides an instruction & guide for developers building a GPT Action for a specific application. Before you proceed, make sure to first familiarize yourself with the following information:



* [Introduction to GPT Actions](https://platform.openai.com/docs/actions)
* [Introduction to GPT Actions Library](https://platform.openai.com/docs/actions/actions-library)
* [Example of Building a GPT Action from Scratch](https://platform.openai.com/docs/actions/getting-started)

This particular GPT Action provides an overview of how to connect to a Snowflake Data Warehouse. This Action takes a user’s question, scans the relevant tables to gather the data schema, then writes a SQL query to answer the user’s question.

Note: This cookbook returns back a [ResultSet SQL statement](https://docs.snowflake.com/en/developer-guide/sql-api/handling-responses#getting-the-data-from-the-results), rather than the full result that is not limited by GPT Actions application/json payload limit. For production and advanced use-case, a middleware is required to return back a CSV file. You can follow instructions in the [GPT Actions - Snowflake Middleware cookbook](../gpt_action_snowflake_middleware) to implement this flow instead.

### Value + Example Business Use Cases


Value: Users can now leverage ChatGPT's natural language capability to connect directly to Snowflake’s Data Warehouse.

Example Use Cases:



* Data scientists can connect to tables and run data analyses using ChatGPT's Data Analysis
* Citizen data users can ask basic questions of their transactional data
* Users gain more visibility into their data & potential anomalies

## Application Information

### Application Key Links


Check out these links from the application before you get started:

* Application Website: [https://app.snowflake.com/](https://app.snowflake.com/)
* Application API Documentation: [https://docs.snowflake.com/en/developer-guide/sql-api/intro](https://docs.snowflake.com/en/developer-guide/sql-api/intro)

### Application Prerequisites

Before you get started, make sure you go through the following steps in your application environment:

* Provision a Snowflake Data Warehouse
* Ensure that the user authenticating into Snowflake via ChatGPT has access to the database, schemas, and tables with the necessary role

## ChatGPT Steps


### Custom GPT Instructions

Once you've created a Custom GPT, copy the text below in the Instructions panel. Have questions? Check out [Getting Started Example](https://platform.openai.com/docs/actions/getting-started) to see how this step works in more detail.


```python
**Context**: You are an expert at writing Snowflake SQL queries. A user is going to ask you a question. 

**Instructions**:
1. No matter the user's question, start by running `runQuery` operation using this query: "SELECT column_name, table_name, data_type, comment FROM {database}.INFORMATION_SCHEMA.COLUMNS" 
-- Assume warehouse = "<insert your default warehouse here>", database = "<insert your default database here>", unless the user provides different values 
2. Convert the user's question into a SQL statement that leverages the step above and run the `runQuery` operation on that SQL statement to confirm the query works. Add a limit of 100 rows
3. Now remove the limit of 100 rows and return back the query for the user to see
4. Use the <your_role> role when querying Snowflake
5. Run each step in sequence. Explain what you are doing in a few sentences, run the action, and then explain what you learned. This will help the user understand the reason behind your workflow. 

**Additional Notes**: If the user says "Let's get started", explain that the user can provide a project or dataset, along with a question they want answered. If the user has no ideas, suggest that we have a sample flights dataset they can query - ask if they want you to query that
```


### OpenAPI Schema

Once you've created a Custom GPT, copy the text below in the Actions panel. Update the servers url to match your Snowflake Account Name url plus `/api/v2` as described [here](https://docs.snowflake.com/en/user-guide/organizations-connect#standard-account-urls). Have questions? Check out [Getting Started Example](https://platform.openai.com/docs/actions/getting-started) to see how this step works in more detail.


```python
openapi: 3.1.0
info:
  title: Snowflake Statements API
  version: 1.0.0
  description: API for executing statements in Snowflake with specific warehouse and role settings.
servers:
  - url: 'https://<orgname>-<account_name>.snowflakecomputing.com/api/v2'


paths:
  /statements:
    post:
      summary: Execute a SQL statement in Snowflake
      description: This endpoint allows users to execute a SQL statement in Snowflake, specifying the warehouse and roles to use.
      operationId: runQuery
      tags:
        - Statements
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                warehouse:
                  type: string
                  description: The name of the Snowflake warehouse to use for the statement execution.
                role:
                  type: string
                  description: The Snowflake role to assume for the statement execution.
                statement:
                  type: string
                  description: The SQL statement to execute.
              required:
                - warehouse
                - role
                - statement
      responses:
        '200':
          description: Successful execution of the SQL statement.
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  data:
                    type: object
                    additionalProperties: true
        '400':
          description: Bad request, e.g., invalid SQL statement or missing parameters.
        '401':
          description: Authentication error, invalid API credentials.
        '500':
          description: Internal server error.
```

## Authentication Instructions

Below are instructions on setting up authentication with this 3rd party application. Have questions? Check out [Getting Started Example](https://platform.openai.com/docs/actions/getting-started) to see how this step works in more detail.

### Pre-Action Steps

Before you set up authentication in ChatGPT, please take the following steps in Snowflake.

### 1. Optional: Configure IP Whitelisting for ChatGPT
Snowflake accounts with network policies that limit connections by IP, may require exceptions to be added for ChatGPT.
* Review the Snowflake documentation on [Network Policies](https://docs.snowflake.com/en/user-guide/network-policies)
* Go to the Snowflake Worksheets
* Create a network rule with the ChatGPT IP egress ranges listed [here](https://platform.openai.com/docs/actions/production/ip-egress-ranges)
* Create a corresponding Network Policy


```python
## Example with ChatGPT IPs as of October 23, 2024
## Make sure to get the current IP ranges from https://platform.openai.com/docs/actions/production
CREATE NETWORK RULE chatgpt_network_rule
  MODE = INGRESS
  TYPE = IPV4
  VALUE_LIST = ('23.102.140.112/28',
                '13.66.11.96/28',
                '104.210.133.240/28',
                '70.37.60.192/28',
                '20.97.188.144/28',
                '20.161.76.48/28',
                '52.234.32.208/28',
                '52.156.132.32/28',
                '40.84.220.192/28',
                '23.98.178.64/28',
                '51.8.155.32/28',
                '20.246.77.240/28',
                '172.178.141.0/28',
                '172.178.141.192/28',
                '40.84.180.128/28');

CREATE NETWORK POLICY chatgpt_network_policy
  ALLOWED_NETWORK_RULE_LIST = ('chatgpt_network_rule');
```

Network policies can be applied at the account, security integration, and user level. The most specific network policy overrides the more general network policies. Depending on how these policies are applied, you may need to alter the policies for individual users in addition to the security integration. If you face this issue, you may encounter Snowflake's error code 390422.

### 2. Set up the Security Integration
* Review the Snowflake OAuth Overview: [https://docs.snowflake.com/en/user-guide/oauth-snowflake-overview](https://docs.snowflake.com/en/user-guide/oauth-snowflake-overview)
* Create new OAuth credentials through a [Security Integration](https://docs.snowflake.com/en/sql-reference/sql/create-security-integration-oauth-snowflake) - you will need a new one for each OAuth app/custom GPT since Snowflake Redirect URIs are 1-1 mapped to Security Integrations


```python
CREATE SECURITY INTEGRATION CHATGPT_INTEGRATION
  TYPE = OAUTH
  ENABLED = TRUE
  OAUTH_CLIENT = CUSTOM
  OAUTH_CLIENT_TYPE = 'CONFIDENTIAL'
  OAUTH_REDIRECT_URI = 'https://oauth.pstmn.io/v1/callback' --- // this is a temporary value while testing your integration. You will replace this with the value your GPT provides
  OAUTH_ISSUE_REFRESH_TOKENS = TRUE
  OAUTH_REFRESH_TOKEN_VALIDITY = 7776000
  NETWORK_POLICY = chatgpt_network_policy; --- // this line should only be included if you followed step 1 above
```


* Retrieve your OAuth Client ID, Auth URL, and Token URL



```python
DESCRIBE SECURITY INTEGRATION CHATGPT_INTEGRATION;
```


You’ll find the required information in these 3 columns:


![../../../images/snowflake_direct_oauth.png](../../../images/snowflake_direct_oauth.png)


* Retrieve your OAuth Client Secret using SHOW_OAUTH_CLIENT_SECRETS


```python
SELECT 
trim(parse_json(SYSTEM$SHOW_OAUTH_CLIENT_SECRETS('CHATGPT_INTEGRATION')):OAUTH_CLIENT_ID) AS OAUTH_CLIENT_ID
, trim(parse_json(SYSTEM$SHOW_OAUTH_CLIENT_SECRETS('CHATGPT_INTEGRATION')):OAUTH_CLIENT_SECRET) AS OAUTH_CLIENT_SECRET;
```

Now is a good time to [test your Snowflake integration in Postman](https://community.snowflake.com/s/article/How-to-configure-postman-for-testing-SQL-API-with-OAuth). If you configured a network policy for your security integration, ensure that it includes the IP of the machine you're using to test.

### In ChatGPT

In ChatGPT, click on "Authentication" and choose "OAuth". Enter in the information below.

| Form Field | Value  |
| -------- | -------- |
| Authentication Type   | OAuth   |
| Client ID   | OAUTH_CLIENT_ID from SHOW_OAUTH_CLIENT_SECRETS   |
| Client Secret   | OAUTH_CLIENT_SECRET from SHOW_OAUTH_CLIENT_SECRETS   |
| Authorization URL   | OAUTH_AUTHORIZATION_ENDPOINT from DESCRIBE SECURITY INTEGRATION |
| Token URL   | OAUTH_TOKEN_ENDPOINT from DESCRIBE SECURITY INTEGRATION   |
| Scope   | session:role:your_role*   |
| Token Exchange Method   | Default (POST Request)   |


*Snowflake scopes pass the role in the format `session:role:<your_role>` for example `session:role:CHATGPT_INTEGRATION_ROLE`. It's possible to leave this empty and specify the role in the instructions, but by adding it here it becomes included in OAuth Consent Request which can sometimes be more reliable. 

### Post-Action Steps


Once you've set up authentication in ChatGPT, follow the steps below in the application to finalize the Action.

* Copy the callback URL from the GPT Action
* Update the Redirect URI in your Security Integration to the callback URL provided in ChatGPT.



```python
ALTER SECURITY INTEGRATION CHATGPT_INTEGRATION SET OAUTH_REDIRECT_URI='https://chat.openai.com/aip/<callback_id>/oauth/callback';
```

### FAQ & Troubleshooting

* The callback url can change if you update the YAML, double check it is correct when making changes.
* _Callback URL Error:_ If you get a callback URL error in ChatGPT, pay close attention to the Post-Action Steps above. You need to add the callback URL directly into your Security Integration for the action to authenticate correctly
* _Schema calls the wrong warehouse or database:_ If ChatGPT calls the wrong warehouse or database, consider updating your instructions to make it more explicit either (a) which warehouse / database should be called or (b) to require the user provide those exact details before it runs the query



_Are there integrations that you’d like us to prioritize? Are there errors in our integrations? File a PR or issue in our github, and we’ll take a look._

