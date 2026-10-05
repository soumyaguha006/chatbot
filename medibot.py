import os

from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")


@st.cache_resource
def load_database():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )


def setup_llm(repo_id, hf_token):
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        provider="featherless-ai",
        temperature=0.5,
        huggingfacehub_api_token=hf_token,
    )
    return ChatHuggingFace(llm=endpoint, temperature=0.5, max_tokens=512)


def main():
    st.title("Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_prompt = st.chat_input("Pass your prompt here")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        custom_prompt_template = """
            Use the pieces of information provided in the context to answer the user's question.
            If you don't know the answer, say that you don't know and do not make up an answer.
            Do not provide anything outside the given context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk.
        """

        try:
            vectorstore = load_database()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=setup_llm("Qwen/Qwen2.5-7B-Instruct", os.environ.get("HF_TOKEN")),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)},
            )

            response = qa_chain.invoke({"query": user_prompt})
            result = response["result"]
            source_documents = response.get("source_documents", [])
        #   result_to_show = result + "\nSource Docs:\n" + str(source_documents)
            result_to_show = result
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as exc:
            st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
