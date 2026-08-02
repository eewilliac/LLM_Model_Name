from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://192.168.50.78:8666",api_key="none")
print(f"the name of the model is: {llm.model}")

response = llm.invoke("Test")
model_name = response.response_metadata.get('model_name')
if "gemma" in model_name:
    print("running gemma")
else:
    print("qwen variant")
