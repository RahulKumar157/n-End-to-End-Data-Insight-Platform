# ## RAG Q&A Conversation With PDF Including Chat History
# import streamlit as st
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_groq import ChatGroq
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# import os

# from dotenv import load_dotenv
# load_dotenv()

# os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
# embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ## set up Streamlit 
# st.title("Conversational RAG With PDF uplaods and chat history")
# st.write("Upload Pdf's and chat with their content")

# ## Input the Groq API Key
# api_key=st.text_input("Enter your Groq API key:",type="password")

# ## Check if groq api key is provided
# if api_key:
#     llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

#     ## chat interface

#     session_id=st.text_input("Session ID",value="default_session")
#     ## statefully manage chat history

#     if 'store' not in st.session_state:
#         st.session_state.store={}

#     uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
#     ## Process uploaded  PDF's
#     if uploaded_files:
#         documents=[]
#         for uploaded_file in uploaded_files:
#             temppdf=f"./temp.pdf"
#             with open(temppdf,"wb") as file:
#                 file.write(uploaded_file.getvalue())
#                 file_name=uploaded_file.name

#             loader=PyPDFLoader(temppdf)
#             docs=loader.load()
#             documents.extend(docs)

#     # Split and create embeddings for the documents
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
#         splits = text_splitter.split_documents(documents)
#         vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
#         retriever = vectorstore.as_retriever()    

#         contextualize_q_system_prompt=(
#             "Given a chat history and the latest user question"
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it if needed and otherwise return it as is."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", contextualize_q_system_prompt),
#                     MessagesPlaceholder("chat_history"),
#                     ("human", "{input}"),
#                 ]
#             )
        
#         history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

#         ## Answer question

#         # Answer question
#         system_prompt = (
#                 "You are an assistant for question-answering tasks. "
#                 "Use the following pieces of retrieved context to answer "
#                 "the question. If you don't know the answer, say that you "
#                 "don't know. Use three sentences maximum and keep the "
#                 "answer concise."
#                 "\n\n"
#                 "{context}"
#             )
#         qa_prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", system_prompt),
#                     MessagesPlaceholder("chat_history"),
#                     ("human", "{input}"),
#                 ]
#             )
        
#         question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
#         rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

#         def get_session_history(session:str)->BaseChatMessageHistory:
#             if session_id not in st.session_state.store:
#                 st.session_state.store[session_id]=ChatMessageHistory()
#             return st.session_state.store[session_id]
        
#         conversational_rag_chain=RunnableWithMessageHistory(
#             rag_chain,get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer"
#         )

#         user_input = st.text_input("Your question:")
#         if user_input:
#             session_history=get_session_history(session_id)
#             response = conversational_rag_chain.invoke(
#                 {"input": user_input},
#                 config={
#                     "configurable": {"session_id":session_id}
#                 },  # constructs a key "abc123" in `store`.
#             )
#             st.write(st.session_state.store)
#             st.write("Assistant:", response['answer'])
#             st.write("Chat History:", session_history.messages)
# else:
#     st.warning("Please enter the GRoq API Key")

## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
#####################################
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression  # or any model you trained
import plotly.express as px
#########################
# Simulate training a model (you can replace this with joblib.load)
def train_dummy_model():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    model = LogisticRegression()
    model.fit(X, y)
    return model

ml_model = train_dummy_model()


# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI setup
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

# API Key input
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Session and chat history management
    session_id = st.text_input("Session ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and embed
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        #  REPLACED Chroma with FAISS
        vectorstore = FAISS.from_documents(splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Build history-aware retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answering system
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User interaction
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the Groq API Key")
    

with st.expander(" Real-time Simulation & ML Prediction", expanded=True):
    st.subheader(" Simulate Data and Predict in Real Time")

    # Number of rows to simulate
    num_points = st.slider("Number of data points", min_value=10, max_value=100, value=20)

    start_button = st.button("Start Simulation")
if start_button:
    sim_data = []
    chart_placeholder = st.empty()

    for i in range(num_points):
        data_point = np.random.rand(1, 5)
        prediction = ml_model.predict(data_point)[0]

        sim_data.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "feature1": data_point[0][0],
            "feature2": data_point[0][1],
            "feature3": data_point[0][2],
            "feature4": data_point[0][3],
            "feature5": data_point[0][4],
            "prediction": prediction
        })

        time.sleep(0.5)

    # Move everything from here down into this block
    df = pd.DataFrame(sim_data)

    # Dropdown to select chart type
    chart_type = st.selectbox("📊 Choose Visualization Type", ["Line Chart", "Scatter", "Bar", "Area", "Heatmap"])

    if chart_type == "Line Chart":
        fig = px.line(df, x="timestamp", y="prediction", title="Line Chart - Predictions Over Time", markers=True)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter":
        fig = px.scatter(df, x="timestamp", y="prediction", color="prediction", title="Scatter Plot - Predictions")
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar":
        bar_data = df['prediction'].value_counts().reset_index()
        bar_data.columns = ['prediction', 'count']
        fig = px.bar(bar_data, x="prediction", y="count", title="Bar Chart - Prediction Counts")
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Area":
        df["index"] = df.index
        fig = px.area(df, x="index", y="prediction", title="Area Chart - Predictions Over Time")
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap":
        import seaborn as sns
        import matplotlib.pyplot as plt

        st.subheader("📍 Heatmap - Feature Correlation")
        sim_df = df[["feature1", "feature2", "feature3", "feature4", "feature5"]]
        fig, ax = plt.subplots()
        sns.heatmap(sim_df.corr(), annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

#         sim_data = []
#         chart_placeholder = st.empty()

#         for i in range(num_points):
#             # Simulate random data point
#             data_point = np.random.rand(1, 5)  # 5 features
#             prediction = ml_model.predict(data_point)[0]

#             # Add to simulation log
#             sim_data.append({
#                 "timestamp": time.strftime("%H:%M:%S"),
#                 "feature1": data_point[0][0],
#                 "feature2": data_point[0][1],
#                 "feature3": data_point[0][2],
#                 "feature4": data_point[0][3],
#                 "feature5": data_point[0][4],
#                 "prediction": prediction
#             })

#             # Convert to DataFrame and plot
#             # df = pd.DataFrame(sim_data)
#             # fig = px.scatter(df, x="timestamp", y="prediction", title="Live Predictions", color="prediction",
#             #                  labels={"timestamp": "Time", "prediction": "Prediction"})
#             # chart_placeholder.plotly_chart(fig, use_container_width=True)

#             # time.sleep(0.5)  # Simulate real-time delay
#             df = pd.DataFrame(sim_data)

# # Dropdown to select chart type
#  chart_type = st.selectbox("📊 Choose Visualization Type", ["Line Chart", "Scatter", "Bar", "Area", "Heatmap"])

# # Render the selected chart
# if chart_type == "Line Chart":
#     fig = px.line(df, x="timestamp", y="prediction", title="Line Chart - Predictions Over Time", markers=True)
#     chart_placeholder.plotly_chart(fig, use_container_width=True)

# elif chart_type == "Scatter":
#     fig = px.scatter(df, x="timestamp", y="prediction", color="prediction", title="Scatter Plot - Predictions")
#     chart_placeholder.plotly_chart(fig, use_container_width=True)

# elif chart_type == "Bar":
#     bar_data = df['prediction'].value_counts().reset_index()
#     bar_data.columns = ['prediction', 'count']
#     fig = px.bar(bar_data, x="prediction", y="count", title="Bar Chart - Prediction Counts")
#     chart_placeholder.plotly_chart(fig, use_container_width=True)

# elif chart_type == "Area":
#     df["index"] = df.index
#     fig = px.area(df, x="index", y="prediction", title="Area Chart - Predictions Over Time")
#     chart_placeholder.plotly_chart(fig, use_container_width=True)

# elif chart_type == "Heatmap":
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     st.subheader("📍 Heatmap - Feature Correlation")
#     sim_df = pd.DataFrame(df[["feature1", "feature2", "feature3", "feature4", "feature5"]])
#     fig, ax = plt.subplots()
#     sns.heatmap(sim_df.corr(), annot=True, cmap="YlGnBu", ax=ax)
#     st.pyplot(fig)


