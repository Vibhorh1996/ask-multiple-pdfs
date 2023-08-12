import streamlit as st
import os
import openai
import config # import config file
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    openai_api_key=os.getenv('OPENAI_API_KEY') #config.api_key
    print("API Key:", openai_api_key) #os.environ.get('OPENAI_API_KEY'))
    embeddings = OpenAIEmbeddings(openai_api_key = os.getenv('OPENAI_API_KEY'), engine=config.deployment_name) #config.api_key) # use config api key
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, deployment_name): # add deployment name as argument
    llm = ChatOpenAI()
    # llm.set_api_key(openai_api_key=os.getenv('OPENAI_API_KEY')) #config.api_key) # use config api key
    # llm.set_deployment_name(deployment_name) # use deployment name from config
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
# ... (imports and other functions)

def main():
    load_dotenv()
    print("Env loaded successfully:", os.getenv('OPENAI_API_KEY') is not None)

    # Retrieve the API key from the environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        st.error("ERROR: OPENAI_API_KEY environment variable not set.")
        st.stop()

    # Print the API key, API base URL, and deployment name for debugging purposes
    print("API Key:", api_key)
    print("API Base:", config.api_base)
    print("Deployment Name:", config.deployment_name)

    st.set_page_config(page_title="Data Chat", page_icon=':robot_face:')
    st.markdown("<h1 stype='text-align:center;'>Data Chat</h1>", unsafe_allow_html=True)
    st.markdown("<h2 stype='text-align:center;'>A Chatbot for conversing with your data</h2>", unsafe_allow_html=True)

    st.write(css, unsafe_allow_html=True)

    # set API parameters from config file
    openai.api_type = config.api_type
    openai.api_key = api_key
    openai.api_base = config.api_base
    openai.api_version = config.api_version

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Calculate the number of batches based on the 16-input limit
                batch_size = 16
                num_batches = (len(text_chunks) + batch_size - 1) // batch_size

                # Create conversation chain for each batch
                conversation_chains = []
                for i in range(num_batches):
                    batch_text_chunks = text_chunks[i * batch_size: (i + 1) * batch_size]
                    vectorstore = get_vectorstore(batch_text_chunks)
                    conversation_chain = get_conversation_chain(vectorstore, config.deployment_name)
                    conversation_chains.append(conversation_chain)

                # Combine all conversation chains
                combined_conversation_chain = ConversationBufferMemory.combine(*conversation_chains)
                st.session_state.conversation = combined_conversation_chain

if __name__ == '__main__':
    main()



# def main():
#     load_dotenv()
#     print("Env loaded successfully:", os.getenv('OPENAI_API_KEY') is not None)


#     # Retrieve the API key from the environment variable
#     api_key = os.getenv('OPENAI_API_KEY')
#     if api_key is None:
#         st.error("ERROR: OPENAI_API_KEY environment variable not set.")
#         st.stop()

#     # Print the API key, API base URL, and deployment name for debugging purposes
#     print("API Key:", api_key)
#     print("API Base:", config.api_base)
#     print("Deployment Name:", config.deployment_name)

#     st.set_page_config(page_title="Data Chat", page_icon=':robot_face:')
#     st.markdown("<h1 stype='text-align:center;'>Data Chat</h1>", unsafe_allow_html=True)
#     st.markdown("<h2 stype='text-align:center;'>A Chatbot for conversing with your data</h2>", unsafe_allow_html=True)

#     st.write(css, unsafe_allow_html=True)

#     # set API parameters from config file
#     openai.api_type = config.api_type
#     openai.api_key = api_key
#     openai.api_base = config.api_base
#     openai.api_version = config.api_version

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get pdf text
#                 raw_text = get_pdf_text(pdf_docs)

#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(vectorstore, config.deployment_name) # pass deployment name from config


if __name__ == '__main__':
    main()
