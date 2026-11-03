
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

# Define the model
model = ChatOllama(
    model="llama3.2",
    temperature=0.1
)

# Define schema
schema = [
    ResponseSchema(
        name="fact_1",
        description="Fact 1 about the topic"
    ),
    ResponseSchema(
        name="fact_2",
        description="Fact 2 about the topic"
    ),
    ResponseSchema(
        name="fact_3",
        description="Fact 3 about the topic"
    ),
]

parser = StructuredOutputParser.from_response_schemas(schema)

# Define prompt
template = PromptTemplate(
    template="""
Give 3 facts about {topic}.

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
