
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Define the model
model = ChatOllama(
    model="llama3.2",
    temperature=0.1
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="""
Give me 5 facts about {topic}.

{format_instruction}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

# Create LCEL chain
chain = template | model | parser

# Invoke chain
result = chain.invoke({"topic": "black hole"})

print(result)
