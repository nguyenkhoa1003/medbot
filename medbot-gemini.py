#
# Healthcare Chatbot with Langchain, Gemini, and MedlinePlus Data
#
# This script demonstrates how to build a Retrieval-Augmented Generation (RAG)
# chatbot using Langchain, the Google Gemini API, and health information
# from MedlinePlus. The chatbot can answer user questions about various
# health topics by searching through the MedlinePlus dataset.
#
# To run this script, you will need to install the following libraries:
#
#   pip install langchain langchain_community langchain_text_splitters langchain_openai langchain_chroma langchain-google-genai faiss-cpu beautifulsoup4 gradio python-dotenv requests
#
# You will also need a Google API key for the Gemini API.
#

import os
import requests
import gradio as gr
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# LOAD API key to environment
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
# It's recommended to set your Google API key as an environment variable
# for security purposes.
# Example: export GOOGLE_API_KEY="YOUR_API_KEY"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

# URL for the MedlinePlus health topics XML data
MEDLINE_PLUS_XML_URL = "https://medlineplus.gov/xml/mplus_topics_2025-07-11.xml"

# OR, you can store data locally
DATA_PATH = "dataset/mplus_topics_2025-07-11.xml"

    
# Read MedlinePlus dataset from local
def load_local_data(file_path):
    with open(DATA_PATH, 'r') as f:
        xml_content = f.read()

    soup = BeautifulSoup(xml_content, "xml")
    health_topics = soup.find_all("health-topic")

    documents = []
    for topic in health_topics:
        title = topic.get("title", "No Title")
        summary = topic.find("full-summary").text.strip()
        page_content = f"Title: {title}\nSummary: {summary}"
        documents.append(Document(page_content=page_content, metadata={"source": "MedlinePlus"}))
    return documents


# Read MedlinePlus dataset from internet
def load_medline_data(url):
    """
    Loads and parses the MedlinePlus XML data from the given URL.

    Args:
        url (str): The URL of the XML data.

    Returns:
        list: A list of Document objects, where each document represents a
              health topic with its title and summary.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, "xml")
        health_topics = soup.find_all("health-topic")

        documents = []
        for topic in health_topics:
            title = topic.get("title", "No Title")
            summary = topic.find("full-summary").text.strip()
            page_content = f"Title: {title}\nSummary: {summary}"
            documents.append(Document(page_content=page_content, metadata={"source": "MedlinePlus"}))
        return documents
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return []


# Store dataset as vector
def create_vector_store(documents):
    """
    Creates a FAISS vector store from a list of documents.

    Args:
        documents (list): A list of Document objects.

    Returns:
        FAISS: A FAISS vector store containing the document embeddings.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    print(f"Split into {len(split_docs)} text chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store


def create_conversational_chain():
    """
    Creates a question-answering chain with a custom prompt template.

    Returns:
        LLMChain: A Langchain question-answering chain.
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def main():
    """
    Main function to load data, create the chatbot, and launch the Gradio UI.
    """
    print("Loading MedlinePlus data...")
    documents = load_medline_data(MEDLINE_PLUS_XML_URL)
    #documents = load_local_data(DATA_PATH)

    if not documents:
        print("No documents were loaded. Exiting.")
        return

    print("Creating vector store...")
    vector_store = create_vector_store(documents)

    print("Creating conversational chain...")
    chain = create_conversational_chain()

    def chatbot_response(question):
        """
        Generates a response to a user's question.

        Args:
            question (str): The user's question.

        Returns:
            str: The chatbot's response.
        """
        docs = vector_store.similarity_search(question)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]

    # --- Gradio Interface ---
    print("Launching Gradio interface...")
    iface = gr.Interface(
        fn=chatbot_response,
        inputs=gr.Textbox(lines=2, placeholder="Ask a health-related question..."),
        outputs="text",
        title="MedPrompt",
        description="MSAI-630-M40: Group 2. This chatbot uses Langchain, Gemini, and MedlinePlus data to answer your health questions. And Gradio for the fastest way for UI",
        theme="soft",
        flagging_mode="never",
    )
    iface.launch()


if __name__ == "__main__":
    main()
