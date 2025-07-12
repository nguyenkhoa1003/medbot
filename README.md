# *MedPrompt: Healthcare Chatbot with Langchain, Gemini, Gradio and MedlinePlus Data*

medbot-gemini.py demonstrates how to build a Retrieval-Augmented Generation (RAG) chatbot using Langchain, the Google Gemini API, and health information from MedlinePlus. The chatbot can answer user questions about various health topics by searching through the MedlinePlus dataset.

Similarly, medbot-gpt.py is using openAI API for AI model.

To run this script, you will need to install the following libraries:

> pip install langchain langchain_community langchain_text_splitters langchain_openai langchain_chroma langchain-google-genai faiss-cpu beautifulsoup4 gradio python-dotenv requests

You will also need a Google API key for the Gemini API, and openAI API key for Chat GPT.
