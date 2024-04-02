from pathlib import Path
import pandas as pd
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.tools import PythonAstREPLTool
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException

MAIN_DIR = Path(__file__).parents[1]
UPLOADED_FILES_DIR = MAIN_DIR / "uploaded_files"

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)

TEMPLATE = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`

<df>
{dhead}
</df>
you can modify the dataframe by invoking one of your tools
"""  # noqa: E501

class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")

class AgentInputs(BaseModel):
    input: str
    file_name: Optional[str] = None

# Assuming uploaded files are saved in 'uploaded_files' directory within MAIN_DIR
UPLOADED_FILES_DIR = MAIN_DIR / "uploaded_files"

class AgentInputs(BaseModel):
    input: str
    file_name: str  # New field to specify which file to work with

def load_dataframe(file_name: str) -> pd.DataFrame:
    """
    Load a DataFrame from the uploaded files directory based on the given file name.
    """
    file_path = UPLOADED_FILES_DIR / file_name
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix == '.xlsx':
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def save_dataframe(file_name: str, df: pd.DataFrame) -> None:
    """
    Load a DataFrame from the uploaded files directory based on the given file name.
    """
    print("Saving DF")
    print(df)
    file_path = UPLOADED_FILES_DIR / file_name
    if file_path.suffix == '.csv':
        df.to_csv(file_path,index=False)
        # return pd.read_csv(file_path)
    elif file_path.suffix in ('.xlsx','.xls'):
        df.to_excel(file_path,index=False)
        # return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def add_row(df, row_data):
    new_row = pd.Series(row_data)
    df = df.append(new_row, ignore_index=True)
    return df

def delete_row(df, condition):
    df = df.loc[~df.apply(condition, axis=1)]
    return df

def modify_row(df, condition, new_data):
    idx = df.loc[df.apply(condition, axis=1)].index
    if not idx.empty:
        for key, value in new_data.items():
            df.at[idx[0], key] = value
    return df

repl = PythonAstREPLTool(
    locals={"add_row": add_row, "delete_row": delete_row, "modify_row": modify_row},
    name="python_repl",
    description="Runs code and returns the output of the final line, allows modification of the dataframe",
    args_schema=PythonInputs,
)
tools = [repl]



async def execute_agent(agent_inputs: AgentInputs):
    """
    Execute the agent for a given input and file name.
    """
    # Load the DataFrame for the specified file
    df = load_dataframe(agent_inputs.file_name)
    
    # Update the prompt with the new DataFrame's head for contextual information
    updated_template = TEMPLATE.format(dhead=df.head().to_markdown())
    
    # Update REPL locals with the new DataFrame
    repl.locals["df"] = df
    
    # Continue with agent execution using the updated template and REPL
    # This part depends on how you execute your agent and pass inputs to it.
    # You might need to adapt the following lines according to your existing execution logic.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", updated_template),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", agent_inputs.input),
        ]
    )
    agent = OpenAIFunctionsAgent(
        llm=ChatOpenAI(temperature=.5, model="gpt-4"), prompt=prompt, tools=tools
    )
    # agent.prompt = prompt  # Assuming you can dynamically update the prompt of the agent
    agent_executor = AgentExecutor(
    agent=agent, tools=tools, max_iterations=5, early_stopping_method="generate"
    ) | (lambda x: x["output"])
    # Execute the agent as before, now with the dynamically loaded DataFrame
    # agent_executor.invoke
    result = agent_executor.invoke({"input": agent_inputs.input})
    save_dataframe(agent_inputs.file_name, repl.locals["df"])
    # repl.locals["df"].to_csv(agent_inputs.file_name+"temp.csv", index=False)
    return result