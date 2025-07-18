import os
from typing import Dict, List, Any, Optional, Union, Callable
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatGoogleGenerativeAI

load_dotenv()

class LangChainIntegration:
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the LangChain LLM with Gemini"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize with fallback options
        model_names = [self.model_name, "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
        
        for model_name in model_names:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0.2,
                    top_p=0.95,
                    top_k=64,
                    max_output_tokens=2048
                )
                print(f"Successfully initialized LangChain with Gemini model: {model_name}")
                return llm
            except Exception as e:
                print(f"Failed to initialize LangChain with model {model_name}: {str(e)}")
                continue
        
        raise ValueError("Failed to initialize any Gemini model for LangChain")
    
    def create_chain(self, system_prompt: str) -> Callable:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
    
    def create_react_chain(self, tools: Dict[str, Callable], system_prompt: str) -> Callable:
        """
        Create a LangChain ReAct chain with the specified tools and system prompt
        
        Args:
            tools: Dictionary of tools available to the agent
            system_prompt: System prompt for the agent
            
        Returns:
            LangChain ReAct chain callable
        """
        from langchain.agents import AgentExecutor, create_react_agent
        
        # Convert tools to LangChain tool format
        from langchain.tools import Tool
        langchain_tools = []
        
        for name, func in tools.items():
            # Get function signature and docstring
            import inspect
            signature = str(inspect.signature(func))
            docstring = inspect.getdoc(func) or f"Tool for {name}"
            
            # Create LangChain tool
            langchain_tools.append(
                Tool(
                    name=name,
                    func=func,
                    description=f"{docstring}\nSignature: {name}{signature}"
                )
            )
        
        # Create ReAct agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        agent = create_react_agent(self.llm, langchain_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=langchain_tools, verbose=True)
        
        return agent_executor
    
    def create_memory_chain(self, system_prompt: str, memory_key: str = "chat_history") -> Callable:
        """
        Create a LangChain chain with memory
        
        Args:
            system_prompt: System prompt for the chain
            memory_key: Key for the memory buffer
            
        Returns:
            LangChain chain with memory
        """
        from langchain.memory import ConversationBufferMemory
        from langchain_core.runnables import RunnablePassthrough
        
        # Create memory
        memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])
        
        # Create chain
        chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: memory.load_memory_variables({})[memory_key]
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        def chain_with_memory(input_text):
            result = chain.invoke({"input": input_text})
            memory.save_context({"input": input_text}, {"output": result})
            return result
        
        return chain_with_memory
    
    def create_sequential_chain(self, chains: List[Callable]) -> Callable:
        """
        Create a sequential chain that runs multiple chains in sequence
        
        Args:
            chains: List of chains to run in sequence
            
        Returns:
            Sequential chain callable
        """
        def sequential_chain(input_text):
            result = input_text
            for chain in chains:
                result = chain(result)
            return result
        
        return sequential_chain
    
    def create_router_chain(self, 
                           router_prompt: str, 
                           destination_chains: Dict[str, Callable]) -> Callable:
        """
        Create a router chain that routes queries to appropriate destination chains
        
        Args:
            router_prompt: Prompt for the router to determine which chain to use
            destination_chains: Dictionary of destination chains
            
        Returns:
            Router chain callable
        """
        from langchain.chains.router import MultiPromptChain
        from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
        from langchain.prompts import PromptTemplate
        
        destinations = [{"name": name, "description": name} for name in destination_chains.keys()]
        
        router_template = router_prompt + "\n\n" + \
            "Options: {destinations}\n" + \
            "Query: {input}\n" + \
            "Destination: "
        
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input", "destinations"],
            output_parser=RouterOutputParser(),
        )
        
        router_chain = LLMRouterChain.from_llm(self.llm, router_prompt)
        
        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=list(destination_chains.values())[0],
            verbose=True,
        )
        
        return chain
