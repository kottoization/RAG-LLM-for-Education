from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from tools.edu_tools import (
    wikipedia_search,
    define_word,
    calculator,
    document_search,
    current_date,
    detect_language,
)

DEFAULT_PROMPT = PromptTemplate.from_template(
    """
Answer the following question as best as you can using the provided tools.
You have access to the following tools:

{tools}

Use the following format:
Question: {input}
Thought: you should think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Always respond in {language}. If any tool returns text in a different language,
translate it to {language} before giving the final answer.

{agent_scratchpad}"""
)


def create_agent(model_name: str = "gpt-3.5-turbo") -> AgentExecutor:
    """Create an agent executor with default tools."""
    tools = [
        wikipedia_search,
        define_word,
        calculator,
        document_search,
        current_date,
        detect_language,
    ]
    llm = ChatOpenAI(model=model_name, temperature=0)
    agent = create_react_agent(llm, tools, DEFAULT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_agent(question: str) -> str:
    """Run the default agent on a question and return the answer."""
    executor = create_agent()
    from tools.language_handler import LanguageHandler
    lang = LanguageHandler.choose_or_detect(question)
    result = executor.invoke({"input": question, "language": lang})
    output = result["output"]
    output = LanguageHandler.ensure_language(output, lang)
    return output
