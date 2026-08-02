from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://192.168.50.78:8666",api_key="none")
response = llm.invoke("Test")
model_name = response.response_metadata.get('model_name')
if "gemma" in model_name:
    print(f"running gemma variant:{model_name}")
else:
    print(f"running qwen variant:{model_name}")
