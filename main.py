from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts.prompt import PromptTemplate

class RAG:
    def __init__(self,path_data,path_db,query):
        self.path_data = path_data
        self.presistant_directory = path_db
        self.query = query
    def load_documents(self):
        loader = PyPDFDirectoryLoader(self.path_data)
        self.documents = loader.load()

    def Text_Splitter(self):
        self.text = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        add_start_index = True,
        strip_whitespace = True,
        )
        self.chunks = self.text.split_documents(self.documents)
        print(f'from{len(self.documents)} to {len(self.chunks)}')

    def get_embeddings(self):
        embedder = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_chunks = [chunk.page_content for chunk in self.chunks]
        self.chunk_embeddings = embedder.embed_documents(self.text_chunks)
        print("converted to embeddings successfully")
        return embedder

    def embeddings_to_db(self):
        embedder = self.get_embeddings()
        if not os.path.exists(self.presistant_directory):
            os.mkdir(self.presistant_directory)

        self.db = FAISS.from_documents(self.chunks,embedder)
        self.db.save_local(self.presistant_directory)
        print("db saved sucessfully")

    def load_db(self):
        self.db = FAISS.load_local(self.presistant_directory, HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),allow_dangerous_deserialization=True)
        print("vector db loaded completely")

    def Search_documents(self):
        retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k":6}) 
        retrivered_doc = retriever.invoke(self.query)
        retrieved_doc = retriever.invoke(self.query)
        self.retrieved_doc_content = " ".join([do.page_content for doc in retrieved_doc])

    def Prompt(self):
        self.prompt_template = PromptTemplate(
        template="Hey! I found some information for you. \n\n{retrieved_doc}\n\nkindly read it carefully andanswer the following query?\n\n{self.query}"
        )

    def model(self):
        api_key = os.environ.get("GOOGLE_API_KEY",)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", verbose=True, temperature =0.1, google_api_key=api_key)
        prompt = self.prompt_template.format(
            retrieved_doc = self.retrieved_doc_content
            query = self.query
        )
        response = llm.invoke()


path_data = r"C:\Users\User\Downloads\RAG\datasets"
path_db = r"C:\Users\User\Downloads\RAG\DB"
query = "What actually is acne?"


obj = RAG(path_data,path_db,query)
#obj.load_documents()
#obj.Text_Splitter()
#obj.get_embeddings()
#obj.embeddings_to_db()
obj.load_db()
while True:
        query = input("please enter your query")
        obj.query = query
        obj.Search_documents()
