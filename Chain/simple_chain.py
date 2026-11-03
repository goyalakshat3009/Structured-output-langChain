
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define prompt
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

# Define Groq model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

# Output parser
parser = StrOutputParser()

# Create LCEL chain
chain = prompt | model | parser

# Execute chain
result = chain.invoke({"topic": "cricket"})

print(result)

# Display chain graph
chain.get_graph().print_ascii()
