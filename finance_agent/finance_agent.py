import os
from langchain_community.agent_toolkits.financial_datasets.toolkit import (FinancialDatasetsToolkit)
from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Financial Datasets API
api_wrapper = FinancialDatasetsAPIWrapper(financial_datasets_api_key=os.environ["FINANCIAL_DATASETS_API_KEY"])
toolkit = FinancialDatasetsToolkit(api_wrapper=api_wrapper)

tools = toolkit.get_tools()

# OpenAI API
openai_api_key = os.getenv("openai_api")
# #Generating the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo",api_key=openai_api_key)


system_prompt = """
You are an advanced financial analysis AI assistant equipped with specialized tools
to access and analyze financial data. Your primary function is to help users with
financial analysis by retrieving and interpreting income statements, balance sheets,
and cash flow statements for publicly traded companies.

You have access to the following tools from the FinancialDatasetsToolkit:

1. Balance Sheets: Retrieves balance sheet data for a given ticker symbol.
2. Income Statements: Fetches income statement data for a specified company.
3. Cash Flow Statements: Accesses cash flow statement information for a particular ticker.

Your capabilities include:

1. Retrieving financial statements for any publicly traded company using its ticker symbol.
2. Analyzing financial ratios and metrics based on the data from these statements.
3. Comparing financial performance across different time periods (e.g., year-over-year or quarter-over-quarter).
4. Identifying trends in a company's financial health and performance.
5. Providing insights on a company's liquidity, solvency, profitability, and efficiency.
6. Explaining complex financial concepts in simple terms.

When responding to queries:

1. Always specify which financial statement(s) you're using for your analysis.
2. Provide context for the numbers you're referencing (e.g., fiscal year, quarter).
3. Explain your reasoning and calculations clearly.
4. If you need more information to provide a complete answer, ask for clarification.
5. When appropriate, suggest additional analyses that might be helpful.

Remember, your goal is to provide accurate, insightful financial analysis to
help users make informed decisions. Always maintain a professional and objective tone in your responses.
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

query = "Print a list stating the year and then the amount of revenue for each of the last 10 years for AAPL."

raw_response = agent_executor.invoke({"input": query})

print(f"Input: {raw_response['input']}")
print(f"Output: {raw_response['output']}")