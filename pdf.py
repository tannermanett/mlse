import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # upload files
    pdf_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

    if pdf_files is not None:
        all_text = ""
        for pdf_file in pdf_files:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                all_text += page.extract_text()
            pdf_file.seek(0)  # Reset the file pointer to the beginning

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(all_text)

        if len(chunks) == 0:
            st.error("Waiting for upload")
            return

        # create embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")  # Replace with your actual OpenAI API key
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.text_input("Ask a question about your PDFs:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)

if __name__ == '__main__':
    main()
