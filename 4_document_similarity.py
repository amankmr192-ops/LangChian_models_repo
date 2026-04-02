from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [

  "Virat Kohli is an Indian cricketer and former captain of the Indian national team.",
  "Sachin Tendulkar is a former Indian cricketer and one of the greatest batsmen in the history of cricket.",
  "MS Dhoni is a former Indian cricketer and captain of the Indian national team.",
  "Rohit Sharma is an Indian cricketer and the current captain of the Indian national team.",
  "Anil Kumble is a former Indian cricketer and one of the  greatest bowlers in the history of cricket."
]

query = "Who is the current captain of the Indian national cricket team?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index,score = sorted(enumerate(scores), key=lambda x: x[1])[-1]

print("Query:", query)
print(documents[index])
print("Similarity Score:", score)