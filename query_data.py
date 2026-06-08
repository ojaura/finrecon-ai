# ============================================================
# query.py
# Queries the ChromaDB vector store using Azure OpenAI embeddings
# and generates answers using the gpt-4o model
# ============================================================

import argparse
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Path to ChromaDB vector store
CHROMA_PATH = "chroma_db"

# Prompt template — instructs the model to answer only from the provided context
PROMPT_TEMPLATE = """
You are a financial reconciliation assistant for TD Bank.
Answer the question based only on the following context.
Be specific and include numbers where available.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

---

Question: {question}
"""

def main():
    # Parse the query from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Initialize Azure OpenAI embeddings 
    embedding_function = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # Load ChromaDB vector store
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search for the top 3 most relevant chunks
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # If no relevant results found, exit early
    if len(results) == 0 or results[0][1] < 0.2:
        print("Unable to find matching results.")
        return

    # Combine retrieved chunks into a single context block
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Format the prompt with context and question
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize Azure gpt-4o model for answer generation
    model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # Generate response
    response_text = model.invoke(prompt).content

    # Display the response and sources
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"\nResponse: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()