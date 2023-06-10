import streamlit as st
import os
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
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
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


def main():
    load_dotenv()
    st.set_page_config(page_title="Data Chat", page_icon=':robot_face:')
    st.markdown("<h1 stype='text-align:center;'>Data Chat</h1>", unsafe_allow_html=True)
    st.markdown("<h2 stype='text-align:center;'>A Chatbot for conversing with your data</h2>", unsafe_allow_html=True)
    
    st.write(css, unsafe_allow_html=True)

    # set API Key
    key = st.text_input('OpenAI API Key','',type='password')
    os.environ['OPENAPI_API_KEY'] = key
    os.environ['OPENAI_API_KEY'] = key

    # initialize session state variables
    # if "generated" not in st.session_state:
    #     st.session_state.generated=None

    # if "past" not in st.session_state:
    #     st.session_state.past= None

    # if "messages" not in st.session_state:
    #     st.session_state['messages']=[
    #         {"role":"DataChat","content":"You are a helpful bot."}
    #     ]

    # if "model_name" not in st.session_state:
    #     st.session_state.model_name=None

    # if "cost" not in st.session_state:
    #     st.session_state.cost= None

    # if "total_tokens" not in st.session_state:
    #     st.session_state.total_tokens=None

    # if "total_cost" not in st.session_state:
    #     st.session_state['total_cost']=0.0

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [{"role": "system", "content": "You will act as a PDF AI Assistant. You will be answering questions only related to the uploaded PDFs. If any question other than PDFs is asked, please reply: 'Not related to the PDFs. Can you ask another question?'"}]
    
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

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

        # model_name = st.sidebar.radio("Choose a model:",("GPT-3.5", "GPT-4"))
        # counter_placeholder = st.sidebar.empty()
        # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
        # #clear_button = st.sidebar.button("Clear Conversation", key="clear")

        # # map model names to OpenAI model IDs
        # if model_name == "GPT-3.5":
        #     model = "gpt-3.5-turbo"
        # else:
        #     model = "gpt-4"

        # # reset everything
        # if st.button("Clear Conversation"):
        #     st.session_state.generated = None
        #     st.session_state.past = None
        #     st.session_state['messages'] = [
        #         {"role": "system", "content": "You are a helpful assistant."}
        #     ]
        #     st.session_state.number_tokens = None
        #     st.session_state.model_name = None
        #     st.session_state.cost = None
        #     st.session_state.total_cost = 0.0
        #     st.session_state.total_tokens = None
        #     counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


if __name__ == '__main__':
    main()
    
    # # initialize session state variables
    # if 'generated' not in st.session_state:
    #     st.session_state['generated']=[]

    # if 'past' not in st.session_state:
    #     st.session_state['past']=[]

    # if 'messages' not in st.session_state:
    #     st.session_state['messages']=[
    #         {"role":"DataChat","content":"You are a helpful bot."}
    #     ]

    # if 'model_name' not in st.session_state:
    #     st.session_state['model_name']=[]

    # if 'cost' not in st.session_state:
    #     st.session_state['cost']=[]

    # if 'total_tokens' not in st.session_state:
    #     st.session_state['total_tokens']=[]

    # if 'total_cost' not in st.session_state:
    #     st.session_state['total_cost']=0.0

    
        
        # model_name = st.sidebar.radio("Choose a model:",("GPT-3.5", "GPT-4"))
        # counter_placeholder = st.sidebar.empty()
        # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
        # clear_button = st.sidebar.button("Clear Conversation", key="clear")

        # # map model names to OpenAI model IDs
        # if model_name == "GPT-3.5":
        #     model = "gpt-3.5-turbo"
        # else:
        #     model = "gpt-4"

        # # reset everything
        # if clear_button:
        #     st.session_state['generated'] = []
        #     st.session_state['past'] = []
        #     st.session_state['messages'] = [
        #         {"role": "system", "content": "You are a helpful assistant."}
        #     ]
        #     st.session_state['number_tokens'] = []
        #     st.session_state['model_name'] = []
        #     st.session_state['cost'] = []
        #     st.session_state['total_cost'] = 0.0
        #     st.session_state['total_tokens'] = []
        #     counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
