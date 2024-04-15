from bs4 import BeautifulSoup
import requests

req = requests.get("https://bitskraft.com/career/")

res = req.content
soup = BeautifulSoup(res, "html.parser")
print(soup)

# urls = ["https://bitskraft.com/career/"]
# loader = AsyncHtmlLoader(urls)
# docs = loader.load()

# print(docs)