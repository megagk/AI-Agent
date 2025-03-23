import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# Generating the llm
openai_api_key = os.getenv("openai_api")
llm = ChatOpenAI(model="gpt-3.5-turbo",api_key=openai_api_key)

# Structuring the research response 
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# structured_output = llm.with_structured_output(AnswerWithJustification)

# Will parse the result into a 'pydantic' format utilising the ResearchResponse Class above.
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
    #.partial allows you to partially fill in the prompt using above instructions.
).partial(format_instructions=parser.get_format_instructions())

# the tools that the agent will utilise.
tools = [search_tool, wiki_tool, save_tool]

# Calling the agent with the appropriate llm, prompt and the tools defined above.
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# agent_executor generates the response...
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose allows you to view the thought process here.

query = str(input("What can I help you research? "))
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output")["text"][0])
    # structured_response = parser.parse(raw_response)
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)