from llama_index.core import PromptTemplate

instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression."""

prompt_template = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}
    The last column of the dataframe contains vector embeddings values for the Text column.
    The embeddings are created using OpenAI API.

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)

context = """Purpose: The primary role of this agent is to assist users by providing accurate 
            information regardinga comprehensive collection of blog posts sourced from Medium.
            The answer has to be short and focused on answering the answer only. 
            The chat is focused specifically on articles published under the "Towards Data Science" publication.
            The information used will be based on the article text and it will be only the essence of most important information."""
