
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda
)

load_dotenv()

# Prompt 1: Generate Report
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# Prompt 2: Summarize Report
prompt2 = PromptTemplate(
    template='Summarize the following text:\n{text}',
    input_variables=['text']
)

# Ollama Model
model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7
)

parser = StrOutputParser()

# Report Generation Chain
report_gen_chain = prompt1 | model | parser

# Conditional Branch
branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 300,
        prompt2 | model | parser
    ),
    RunnablePassthrough()
)

# Final Chain
final_chain = RunnableSequence(
    report_gen_chain,
    branch_chain
)

# Invoke
result = final_chain.invoke({
    'topic': 'Russia vs Ukraine'
})

print(result)

# Display Graph
final_chain.get_graph().print_ascii()
