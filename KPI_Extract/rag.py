# !python3 -m pip install --upgrade pip
# !pip install -qU pypdf langchain_community PyPDF2 nltk pymupdf transformers pinecone-client chromadb bitsandbytes sentence-transformers peft streamlit
# !pip install --upgrade transformers bitsandbytes
# !pip install seqeval
# !pip install stanza
# !pip install spacy
# !python -m spacy download en_core_web_sm
# !pip install datasets
# !pip install openpyxl
# !pip install tf-keras
# !pip install --upgrade numpy scipy
# !pip install spacy
# !pip install qdrant_client

from langchain_community.document_loaders import PyPDFLoader
import PyPDF2
import fitz
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertModel
from langchain.embeddings.base import Embeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaModel, BitsAndBytesConfig, pipeline
from torch import cuda, bfloat16
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import langchain
import bertopic
from sklearn.feature_extraction.text import CountVectorizer
# from sentence_transformers import SentenceTransformer
import hdbscan
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import TextGeneration
import transformers
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma

from bertopic.representation import TextGeneration , KeyBERTInspired, MaximalMarginalRelevance
from utils import extract_text_with_page_numbers, hf_secret, mistral_secret
from mistralai import Mistral

import re
import pandas as pd
import ast
from transformers import AutoModel
# from mistralai.exceptions import MistralAPIException  # ✅ Add this
import time













# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






class RAG:

    def __init__(self, pdf_path, embed_model_id, model_name, tokenizer, temperature, question):
        self.pdf= pdf_path
        self.embed_model_id= embed_model_id
        self.model_name= model_name
        self.tokenizer= tokenizer
        self.temperature= temperature
        self.question= question
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


    def read_data(self):
        text_chunks, page_numbers = extract_text_with_page_numbers(self.pdf)
        self.text_chunks= text_chunks
        self.page_number= page_numbers
    
    # text_chunks= read_data()

    
    
    def embedding_function(self):

        embed_model = HuggingFaceEmbeddings(
            model_name= self.embed_model_id,
            model_kwargs={'device': self.device},
            # encode_kwargs={'device': device, 'batch_size': 32}
        )
        vectors = []
        for chunk in self.text_chunks:
            vectors.append(embed_model.embed_query(chunk))  

        # for idx, vector in enumerate(vectors):
        #     print(f"Vector {idx+1}: {vector}")

        self.vectors= vectors

    
    def push_to_chromadb(self):

        client = chromadb.PersistentClient()
        
        # collection = client.get_or_create_collection(name='nidhi_rag', metadata= {"embedding_dimension":1024})
        collection = client.get_or_create_collection(name='nidhi_project')


        existing_ids = collection.get()['ids']
    
        # Only delete if there are existing entries in the collection
        if existing_ids:
            collection.delete(ids=existing_ids)

        collection.add(
            documents= self.text_chunks,
            embeddings= self.vectors ,
            ids=[str(i) for i in range(len(self.text_chunks))],
            metadatas=[{"page_number": page} for page in self.page_number]
        )

        # Check data integrity
        print(f"Total documents in collection '{'nidhi_project'}': {collection.count()}")
        print(f"Collections available: {client.list_collections()}")

        model = SentenceTransformer(self.embed_model_id)  # Replace with your model
        query_text = "what is the Earnings per share for year 2023?"
        query_vector = model.encode([query_text])
    
        # Perform the query with the generated embedding
        results = collection.query(query_embeddings=query_vector, n_results=1)
        
        # results = collection.query(query_texts=[query_text], n_results=1)
        # for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        #     print(f"Page {meta['page_number']}: {doc}")

        return client.list_collections()

    def quantization_function(self):

        token = hf_secret()
        # model_name= "meta-llama/Llama-3.1-8B"
        # tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token= token)
        

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype= bfloat16
        )


        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            use_auth_token= token,
            quantization_config=bnb_config,  # Use 'int8' or 'int4' for quantization
            # device_map="auto"  # Automatically map the model to available devices
        )

        self.quantized_model = model.to(self.device)

        # return model

    
    def rag_setup(self):
        
        self.chroma_vectorstore = Chroma(
            client= chromadb.PersistentClient(),
            collection_name= 'nidhi_project',
            embedding_function=HuggingFaceEmbeddings(model_name=self.embed_model_id)
        )

        # Create a Hugging Face pipeline for text generation
        llama_pipeline = pipeline(
            "text-generation",
            model= self.quantized_model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,
            temperature =self.temperature
        )

        # Load LLaMA into LangChain LLM Wrapper
        self.llm = HuggingFacePipeline(pipeline=llama_pipeline)

    def rag_query(self):

        rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an AI assistant answering questions based on retrieved documents. 
            Follow these steps for accuracy:
            
            1. Read the provided context carefully.
            2. Identify relevant details that answer the question.
            3. Summarize the key points **without adding external knowledge**.
            4. If uncertain, say **"I don't have enough information in the provided context."**
        
            Context:
            {context}
        
            Question:
            {question}
        
            Answer:
            """)
        
        # Retrieve top documents from Chroma
        docs = self.chroma_vectorstore.similarity_search(self.question,k=4)
        print(len(docs))

        if not docs:
            return "No relevant information found."

        # Ensure retrieved context is structured properly
        context = "\n\n".join([f"- {doc.page_content}" for doc in docs])
        # context= docs

        # Prepare the prompt with retrieved context
        final_prompt = rag_prompt.format(context=context, question=self.question)

        # Generate the answer using LLaMA
        self.response = self.llm(final_prompt)

        answer_text = self.response
        if "Answer:" in answer_text:
            answer_text = answer_text.split("Answer:")[-1].strip()

        context= self.response
        if "Context:" in context:
            context= context.split("Context:")[-1].strip()
        
        # Print only the question and the answer
        print("Question:", self.question)
        print("Answer:", answer_text)
        # print("Context:", context)
    
        return self.response

    def run(self):
        self.read_data()
        self.embedding_function()
        self.push_to_chromadb()
        self.quantization_function()
        self.rag_setup()
        self.rag_query()










# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






'''

Topic Modelling

'''



class TopicModelling:

    def __init__(self, pdf, representation_model, embedding_model, hdbscan_model, min_df, max_df, ngram_range, stop_words, nr_topics, top_n_words):
        self.pdf_path= pdf
        self.representation_model=representation_model
        self.embedding_model= embedding_model
        self.hdbscan_model= hdbscan_model
        self.min_df= min_df
        self.max_df= max_df
        self.ngram_range= ngram_range
        self.stop_words= stop_words
        self.nr_topics= nr_topics
        self.top_n_words= top_n_words

    def extract_text(self):
        text_chunks, page_numbers= extract_text_with_page_numbers(self.pdf_path)
    
        self.documents = [doc.lower() for doc in text_chunks]

        
    
        
    def call_mistral_with_retry(self, messages, retries=3, wait_time=5, timeout_ms=20000):
        
        """
        Handles API calls with retries for rate limits and timeouts.
        """
        client = Mistral(api_key=mistral_secret()) 

        for attempt in range(retries):
            try:
                response = client.chat.complete(
                    model="mistral-small-latest",
                    messages=messages,
                    temperature=0.7,
                    timeout_ms=timeout_ms  # Set timeout
                )
                return response
            except Exception as e:
                if "Requests rate limit exceeded" in str(e):
                    wait = wait_time * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit hit. Retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    raise e  # Raise other errors immediately
        
        raise Exception("Exceeded retry attempts due to rate limit.")

    def topic_title(self, representative_docs):
        """
        Generates a title from representative documents.
        """
        prompt_template = f"""
        I am giving you the following text {representative_docs}, 
        can you assign a title in 3 words and give the output in the form of a Python string?
        Example output: 'Generated Topic Name'
        """

        messages = [
            {"role": "system", "content": "You are an AI assistant who answers financial questions."},
            {"role": "user", "content": prompt_template}
        ]

        response = self.call_mistral_with_retry(messages)
        raw_response = response.choices[0].message.content.strip()

        pattern = r"```(?:python)?\n\"([^\"]+)\"\n```"
        match = re.search(pattern, raw_response)

        if match:
            title = match.group(1)
            print(f"Extracted Title: {title}")
            return title
        else:
            return raw_response

    def generate_summary(self, representative_docs):
        """
        Generates a summary for the given representative documents.
        """
        messages = [
            {"role": "system", "content": "You are an AI assistant who generates summaries."},
            {"role": "user", "content": f"Create a meaningful text of 400 words using the following keywords\n\n{representative_docs}. Then, create a meaningful summary of 250 words."}
        ]

        response = self.call_mistral_with_retry(messages)
        raw_text = response.choices[0].message.content.strip()

        match = re.search(r"\*\*Summary:\*\*\n\n(.*)", raw_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            summary_start = raw_text.find("Summary:")
            if summary_start != -1:
                summary_text = raw_text[summary_start:].split("\n", 1)[1].strip()
                return summary_text
            else:
                return raw_text



    
    def fit_model(self):
        vectorizer_model = CountVectorizer(stop_words=self.stop_words, ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df)
        
        # Initialize BERTopic model
        self.bertopic_model = bertopic.BERTopic(
            language='multilingual',
            embedding_model=self.embedding_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=self.nr_topics,
            representation_model=self.representation_model,
            verbose=True,
            top_n_words=self.top_n_words
        )
        
        # Fit model on a subset of documents
        topics, probs = self.bertopic_model.fit_transform(self.documents)
        
        # Extract topic information
        self.topic_info = self.bertopic_model.get_topic_info()
        self.topic_info= self.topic_info[self.topic_info['Topic']>-1]
        
        # Extract representative documents for each topic
        # topic_representative_docs = {}
        topic_data = {'Title': [], 'Topic': [], 'Summary': []}
        
        for topic_id in self.topic_info['Topic'].unique():
            
            representative_doc = self.topic_info[self.topic_info['Topic'] == topic_id]['Representation'].values[0]
            title = self.topic_title(representative_doc)
            summary = self.generate_summary(representative_doc)
            
            topic_data['Title'].append(title)
            topic_data['Topic'].append(topic_id)
            topic_data['Summary'].append(summary)
        
        topic_data= pd.DataFrame(topic_data)
        
        
        self.topic_info.to_csv('topic_info.csv', index=False)
        topic_data.to_csv('topic_data.csv', index=False)
        self.bertopic_model.save("topic_model.pkl", serialization='pickle')
        
        print("Topic Information:")
        print(self.topic_info)
        
        return topics, probs


    
    def run(self):
        self.extract_text()
        self.fit_model()








# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# KEYWORDS= [
#             "Board-level ESG accountability",
#             "Sustainability risk integration into business model",
#             "ESG-linked executive compensation",
#             "Employee Benefits & Pension Accounting",
#             "Defined contribution vs. defined benefit plans",
#             "Share-based payments",
#             "Actuarial assumptions in pension valuation",
#             "GHG emissions reporting (Scope 1, 2, 3)",
#             "Pollution & hazardous material management",
#             "Water & resource consumption transparency",
#             "Biodiversity impact assessment",
#             "Circular economy & sustainable product lifecycle"
#         ]

# embedding_model = SentenceTransformer("all-mpnet-base-v2")


# system_prompt = """
# <s>[INST] <<SYS>>
# You are a helpful, respectful and honest assistant for labeling financial topics.
# <</SYS>>
# """

# example_prompt = """
# I have several topics document which has the yearly report publshed by Commerze Bank and they have published all the inofrmation, employee count, profit, earning per share, etc

# The topic is described by the following keywords: '"Water & resource consumption transparency, Pollution & hazardous material management, Biodiversity impact assessment'.

# Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

# [/INST] Environmental Disclosures (E1-E5)
# """

# main_prompt = f"""
# [INST]
# I have a topic that contains the following documents:
# {chunks}

# The topic is described by the following keywords: {KEYWORDS}.

# Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
# [/INST]
# """

# prompt = system_prompt + example_prompt + main_prompt

# generator = transformers.pipeline(
#     model= model, tokenizer= tokenizer,
#     task='text-generation',
#     temperature= temperature,
#     max_new_tokens=512)


# model = TextGeneration(generator, prompt= self.prompt)

# # representation_model = llama3
# representation_model_2 = KeyBERTInspired()
# representation_model_3 = MaximalMarginalRelevance(diversity=0.3)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
