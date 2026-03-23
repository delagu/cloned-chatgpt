from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os

def get_chat_response(prompt,memory,api_key):
    model = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        api_key=api_key
    )
    chain = ConversationChain(llm=model,memory=memory)
    response = chain.invoke({"input":prompt})
    return response

if __name__ == '__main__':
    memory = ConversationBufferMemory(return_messages=True)
    response = get_chat_response("牛顿提出过哪些知名的定律？",memory,os.getenv("DASHSCOPE_API_KEY"))
    print(response["response"])