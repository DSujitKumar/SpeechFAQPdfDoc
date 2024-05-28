import base64

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import  PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import  FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from gtts import gTTS
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import os
def getPDFText(pdfFiles):
    text=""
    for pdf in pdfFiles:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def getTextChunks(raw_text):
    character_TextSplitter=CharacterTextSplitter(separator="\n",chunk_size=500,chunk_overlap=200,length_function=len)
    chunks=character_TextSplitter.split_text(raw_text)
    return chunks


def getVectorStore(text_chunks):
    embeddings=HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
    vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):

    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

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
            language = 'en'
            mytext = message.content
            myobj = gTTS(text=mytext, lang=language, slow=False)
            myobj.save("welcome.mp3")
            autoplay_audio("welcome.mp3")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs",page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    user_Audio = Button(label="Speak",button_type='success', width=400, height=100)
    user_Audio.js_on_event("button_click", CustomJS(code="""
            var recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;

            recognition.onresult = function (e) {
                var value = "";
                for (var i = e.resultIndex; i < e.results.length; ++i) {
                    if (e.results[i].isFinal) {
                        value += e.results[i][0].transcript;
                    }
                }
                if ( value != "") {
                    document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                }
            }
            recognition.start();
            """))
    result = streamlit_bokeh_events(
        user_Audio,
        events="GET_TEXT",
        key="listen",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0)

    if result:
        if "GET_TEXT" in result:
            print(result.get("GET_TEXT"))
            handle_userinput(result.get("GET_TEXT"))
    if user_question:
        print(user_question)
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your Documnets:")
        pdfFiles=st.file_uploader("Upload Your Files here",accept_multiple_files=True)
        if st.button("Process"):
            #get all the files
            with st.spinner("Processing..."):
                raw_text=getPDFText(pdfFiles)

            #get the Text chunks
            text_chunks=getTextChunks(raw_text)
            #st.write(text_chunks)

            #cretae Vector storage
            vector_store=getVectorStore(text_chunks)
            st.write("File has been procesed Successfully...")
            #create conversation
            st.session_state.conversation= get_conversation_chain(vector_store)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

