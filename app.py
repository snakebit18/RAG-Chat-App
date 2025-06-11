## RAG Q&A Conversation With PDF Including Chat History
"""
Conversational RAG Q&A Application with PDF Uploads and Chat History using Streamlit.
This application allows users to upload PDF files, which are then processed and indexed for retrieval-augmented generation (RAG) question answering. Users can interact with the content of the uploaded PDFs through a conversational interface that maintains chat history for context-aware responses.
Features:
- Upload and process multiple PDF files.
- Extract and split text from PDFs for embedding and retrieval.
- Use HuggingFace embeddings and Chroma vector store for document retrieval.
- Integrate with Groq LLM for generating responses.
- Reformulate user questions based on chat history for improved retrieval.
- Maintain per-session chat history for contextual conversations.
- Streamlit-based user interface for API key input, file upload, session management, and chat interaction.
Environment Variables:
- HF_TOKEN: HuggingFace API token for embeddings.
Inputs:
- Groq API key (entered by user).
- PDF files (uploaded by user).
- Session ID (for chat history management).
- User questions (entered via chat interface).
Outputs:
- Assistant responses based on retrieved PDF content and chat history.
- Display of chat history and session state.
Dependencies:
- streamlit
- langchain
- langchain_community
- langchain_core
- langchain_chroma
- langchain_groq
- langchain_huggingface
- langchain_text_splitters
- python-dotenv
Usage:
1. Enter your Groq API key.
2. Upload one or more PDF files.
3. Enter a session ID (optional, defaults to "default_session").
4. Ask questions about the content of the uploaded PDFs.
5. View assistant responses and chat history.
Note:
- The application requires valid API keys and internet access for embedding and LLM services.
- Uploaded PDFs are temporarily saved and processed for each session.
"""
import streamlit as st

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACE_API_KEY']=os.getenv("HUGGINGFACE_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## set up Streamlit 
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

## Input the Groq API Key
api_key=st.text_input("Enter your Groq API key:",type="password")

## Check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    ## chat interface

    session_id=st.text_input("Session ID",value="default_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    ## Process uploaded  PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()    

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Answer question

        # Answer question
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
### here history aware retriever is used to reformulate the user question based on the chat history instead of database retrieval


        def get_session_history(session:str)->BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session]=ChatMessageHistory()
            return st.session_state.store[session]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")










