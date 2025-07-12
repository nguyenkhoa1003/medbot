# MedPrompt: Healthcare Chatbot with Langchain, Gemini, Gradio and MedlinePlus Data

## Group 2: Residency Day 2: Project 2: Hackathon Projects
### The University of Cumberlands - MSAI-630-M40: Generative AI with Large Language Models

This is an LLM-based assistant that helps patients understand symptoms, diseases, medications, and basic first aid by answering queries from users/patients.

medbot-gemini.py demonstrates how to build a Retrieval-Augmented Generation (RAG) chatbot using Langchain, the Google Gemini API, and health information from MedlinePlus. The chatbot can answer user questions about various health topics by searching through the MedlinePlus dataset.

Similarly, medbot-gpt.py is using openAI API for AI model.

To run this script, you will need to install the following libraries:

> pip install langchain langchain_community langchain_text_splitters langchain_openai langchain_chroma langchain-google-genai faiss-cpu beautifulsoup4 gradio python-dotenv requests

You will also need a Google API key for using Gemini LLM, and openAI API key for ChatGPT LLM.

To get Google API key, go to [https://aistudio.google.com/](https://aistudio.google.com/app/apikey)

To get openAI API key, go to https://platform.openai.com/settings/organization/api-keys

After you have the key, put it into the *.env* file.
```
OPENAI_API_KEY = "Add your openAI API key here"
GOOGLE_API_KEY = "Add your gemini API key here"
```
