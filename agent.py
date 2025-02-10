import streamlit as st
import os
import warnings
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_retries=2)

prompt = PromptTemplate(
    input_variables=["task_description"],
    template="""
    You are a coding assistant. Based on the task description, generate Python code and execute it. Return the result or any error messages.
    Task Description: {task_description}
    Output the result in this format:
    ```result
    [result/error]
    ```
    """
)

python_tool = PythonREPLTool()

llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = initialize_agent(
    tools=[Tool(name="Python Executor", func=python_tool.run, description="Execute Python code.")],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def main():
    st.title("LangChain Custom Agent for Code Generation and Execution")
    st.markdown("### Enter a task description and see the generated Python code in action.")

    task_description = st.text_area("Task Description", placeholder="Describe the task you want solved using Python.")

    if st.button("Generate and Execute Code"):
        if task_description.strip():
            with st.spinner("Processing..."):
                try:
                    response = agent.run(task_description)
                    st.markdown("**Generated Code Execution Result:**")
                    st.text_area("Result", response, height=200, disabled=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a task description.")

if __name__ == "__main__":
    main()
