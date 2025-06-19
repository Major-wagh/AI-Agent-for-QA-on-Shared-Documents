import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from dotenv import load_dotenv

load_dotenv()

class QAAgent:
    def __init__(self, faiss_index_path="faiss_excel_index"):
        self.faiss_index_path = faiss_index_path
        self.vector_store = None
        self.llm = None
        self.vector_store = None
        self.retriever_tool = None
        self.agent_executor = None
        self._initialize_llm()
        self._load_vector_store()
        self._setup_agent()

    def _initialize_llm(self):
        """Initializes the Google Generative AI LLM."""
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set!")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)

    def _load_vector_store(self):
        """Loads the existing FAISS vector store."""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                task_type="semantic_similarity"
            )
            self.vector_store = FAISS.load_local(
                self.faiss_index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"FAISS vector store loaded from {self.faiss_index_path}")
            self.retriever_tool = create_retriever_tool(
                self.vector_store.as_retriever(),
                "excel_document_retriever",
                "Searches and returns information from Excel documents."
            )
        except Exception as e:
            print(f"Error loading FAISS vector store: {e}")
            self.vector_store = None
            self.retriever_tool = None

    def _setup_agent(self):
        """Sets up the LangGraph ReAct agent."""
        if not self.llm:
            raise ValueError("LLM not initialized. Cannot set up agent.")
        if not self.retriever_tool:
            raise ValueError("Retriever tool not initialized. Cannot set up agent.")

        tools = [self.retriever_tool]
        self.agent_executor = create_react_agent(self.llm, tools)

    def ask_question(self, query: str) -> str:
        """
        Answers a question using the LangGraph ReAct agent.
        """
        if not self.agent_executor:
            return "Error: Agent not set up. Cannot answer questions."

        try:
            # The agent expects a list of messages
            result = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=query)]}
            )
            # The result from the agent executor will be a dictionary,
            # we need to extract the 'output' or 'messages' content.
            # Depending on the agent's final step, it might be in 'output' or the last message's content.
            if "output" in result:
                return result["output"]
            elif "messages" in result and result["messages"]:
                # Get the content of the last message, which should be the answer
                return result["messages"][-1].content
            else:
                return "Could not retrieve an answer from the agent."
        except Exception as e:
            return f"An error occurred while processing your question: {e}"

if __name__ == "__main__":
    # Example usage:
    # Ensure you have GOOGLE_API_KEY set in your .env file
    # And the faiss_excel_index directory exists with your embeddings
    
    agent = QAAgent()
    if agent.vector_store:
        question = "What is the total sales for product A?"
        print(f"\nQuestion: {question}")
        answer = agent.ask_question(question)
        print(f"Answer: {answer}")

        question = "What is the forecast for Q3?"
        print(f"\nQuestion: {question}")
        answer = agent.ask_question(question)
        print(f"Answer: {answer}")
    else:
        print("Could not initialize QA Agent. Check your FAISS index path and GOOGLE_API_KEY.")
