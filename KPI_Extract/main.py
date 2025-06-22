import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.io as pio
from rag import  RAG, TopicModelling
from utils import extract_text_with_page_numbers, hf_secret, mistral_secret
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import bertopic
from bertopic.representation import MaximalMarginalRelevance
from bert_ft_02 import NERPipeline, Inferencing
import re
import plotly.express as px
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import time




# Initialize tokenizer
model_name_rag = "mistralai/Mistral-Small-24B-Instruct-2501"
tokenizer_rag = AutoTokenizer.from_pretrained(model_name_rag, use_auth_token=hf_secret)

# Dynamic hyperparameter setting based on document size
def set_bertopic_params(num_pages):
    if num_pages < 25:
        return {'min_df': 0.03, 'max_df': 0.94, 'nr_topics': 5, 'ngram_range': (1, 2)}
    elif 25 <= num_pages <= 200:
        return {'min_df': 0.07, 'max_df': 0.85, 'nr_topics': 10, 'ngram_range': (1, 3)}
    else:
        return {'min_df': 0.10, 'max_df': 0.85, 'nr_topics': 15, 'ngram_range': (1, 3)}

def main():
    st.set_page_config(page_title="KPI Extraction & Smart Analysis", layout="wide")
    st.title("📊 KPI Extraction & Smart Analysis Tool")

    uploaded_file = st.file_uploader("Upload a financial report (PDF)", type=["pdf"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            pdf_path = temp_pdf.name

        st.success("PDF uploaded successfully!")
        text_chunks, page_numbers = extract_text_with_page_numbers(pdf_path)
        num_pages = len(page_numbers)

        # # Ask the user to select a report type. Only one can be selected.
        # st.write("Please select the report type:")
        # report_type = st.radio("Select Report Type", options=["Financial Report", "ESG Report"])

        # # Determine the model name based on the selected report type
        # if report_type == "Financial Report":
        #     model_name = 'Nidhilakhan-17/finbert_ner_model'
        # else:
        #     model_name = 'Nidhilakhan-17/nl_kpi_ner_esg_bert'

        model_name = 'Nidhilakhan-17/nl_kpi_ner_bert'

        # Tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["🔑 KPI Extraction", "📚 Topic Modeling", "💬 Chat with PDF"])

        with tab1:
            st.subheader("🔑 KPI Extraction")
            if st.button("Extract KPIs"):
                with st.spinner("Extracting KPIs..."):
                    
                    kpi_pipeline = Inferencing(pdf_path= pdf_path, inf_model_name= model_name)
                    df = kpi_pipeline.run_inferencing()
                    visualization = kpi_pipeline.visualization(df)
                    
                    st.write("### Extracted KPI Table")
                    st.dataframe(df)
                    
                    # Display the KPI visualization graph
                    st.write("### KPI Visualization")
                    st.plotly_chart(visualization, use_container_width=True)



        with tab2:
            st.subheader("📚 Topic Modeling")
            bertopic_params = set_bertopic_params(num_pages)
        
            if st.button("Run Topic Modeling"):
                with st.spinner("Running Topic Modeling..."):
                    topic_model = TopicModelling(
                        pdf_path, representation_model=MaximalMarginalRelevance(diversity=0.3),
                        embedding_model=None,
                        hdbscan_model=None,
                        min_df=bertopic_params['min_df'], max_df=bertopic_params['max_df'],
                        ngram_range=bertopic_params['ngram_range'], stop_words='english',
                        nr_topics=bertopic_params['nr_topics'], top_n_words=10
                    )
                    topics = topic_model.run()
        
                    # Load topic information
                    st.write("### Topic Names & Summaries")
                    topic_info = pd.read_csv('topic_info.csv')
                    topic_data= pd.read_csv('topic_data.csv')
        
                    # Display topics with summaries
                    for _, row in topic_data.iterrows():
                        with st.expander(f"Topic {row['Topic']}: {row['Title']}"):
                            st.write(f"**Summary:** {row['Summary']}")
        
                    # Display topic bar chart
                    st.write("### Topic Distribution")
                    fig = topic_model.bertopic_model.visualize_barchart()
                    st.plotly_chart(fig)

        with tab3:
            st.subheader("💬 Chat with PDF")
            if st.button("Start Chat"):
                st.session_state['rag_pipeline'] = RAG(pdf_path, embed_model_id="all-mpnet-base-v2",
                                                      model_name= model_name_rag, tokenizer= tokenizer_rag,
                                                      temperature=0.7, question=" ")
                st.session_state['rag_pipeline'].run()
                st.success("Chatbot initialized!")

            if 'rag_pipeline' in st.session_state:
                user_query = st.text_input("Ask something about your report:")
                if st.button("Ask") and user_query:
                    with st.spinner("Fetching response..."):
                        st.session_state['rag_pipeline'].question = user_query
                        # st.session_state['rag_pipeline'].rag_query()
                        response = st.session_state['rag_pipeline'].rag_query()
                        answer_text = st.session_state['rag_pipeline'].response
                        if "Answer:" in answer_text:
                            answer_text = answer_text.split("Answer:")[-1].strip()
                        else:
                            answer_text= response
                
                        context= st.session_state['rag_pipeline'].response
                        if "Context:" in context:
                            context= context.split("Context:")[-1].strip()
                        
                        # # Print only the question and the answer
                        # print("Question:", self.question)
                        # print("Answer:", answer_text)
                        if response:
                            st.write(f"**Question:** {user_query}")
                            st.write(f"**Answer:** {answer_text}")
                            
                        else:
                            st.error("No answer received.")


if __name__ == "__main__":
    main()
