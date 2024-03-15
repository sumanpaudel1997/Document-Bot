from langchain_anthropic import ChatAnthropic
import os

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.2, max_tokens=1024)
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')


print(llm.invoke("how can langsmith help with testing?"))