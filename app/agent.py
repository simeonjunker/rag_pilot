from app.llm import chat_model
from langchain.agents import create_agent
from app.tools import query_wikipedia_db, get_weather
from langgraph.checkpoint.memory import InMemorySaver


class ChatAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt="You are a helpful assistant that provides touristic information about the German state of Saxony.",
            tools=[query_wikipedia_db, get_weather],
            checkpointer=InMemorySaver(),
        )

    def verbose_query(self, question, thread_id="1"):
        """
        Queries the chat agent with a given question and thread ID.
        
        Args:
            question (str): The question to ask the chat agent.
            thread_id (str, optional): The thread ID for the conversation. Defaults to "1".
        """
        for step in self.agent.stream(
            {"messages": [("user", question)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
            
            
    async def stream_answer(self, question, thread_id="1"):
        config = {"configurable": {"thread_id": thread_id}}
        async for chunk, metadata in self.agent.astream(
            {"messages": [{"role": "user", "content": question}]},
            config,
            stream_mode="messages",
        ):
            
            if chunk.content:
                if chunk.type == "tool":
                    continue
                yield chunk.content
                
                
    async def stream_query(self, question, thread_id="1"):
        async for token in self.stream_answer(question, thread_id=thread_id):
            print(token, end="", flush=True)
