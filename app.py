import streamlit as st
import os
import uuid
import pandas as pd
import chromadb
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

# --- Page Title and Description ---
st.set_page_config(page_title="AI Cold Mail Generator", layout="wide")
st.title("📧 AI Cold Mail Generator")
st.markdown("Provide a job posting URL to generate a tailored cold email from AtliQ.")
st.divider()

# --- Get API key from Environment Variables ---
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ API key not found. Please set the GROQ_API_KEY environment variable.")
    st.stop()

# --- Initialize Vector Database on Startup ---
try:
    df = pd.read_csv("my_portfolio.csv")
    client = chromadb.PersistentClient('vectorstore')
    collection = client.get_or_create_collection(name="portfolio")
    
    if collection.count() != len(df):
        if collection.count() > 0:
            collection.delete(ids=collection.get()['ids'])
        
        for _, row in df.iterrows():
            collection.add(
                documents=row["Techstack"],
                metadatas={"links": row["Links"]},
                ids=[str(uuid.uuid4())]
            )
        st.success("✅ Portfolio vector database populated.")
    else:
        st.info("✅ Portfolio vector database already exists.")
except FileNotFoundError:
    st.error("❌ 'my_portfolio.csv' not found. Please ensure it is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ An error occurred during vector database initialization: {e}")
    st.stop()

# --- Input Field ---
job_url = st.text_input("Enter the Job Posting URL:")
generate_button = st.button("Generate Cold Mail")

# --- Main Logic ---
if generate_button:
    if not job_url:
        st.warning("Please enter a job URL to continue.")
        st.stop()

    with st.spinner("Processing... This may take a moment."):
        # --- 1. Initialize LLM ---
        try:
            llm = ChatGroq(
                temperature=0,
                groq_api_key=groq_api_key,
                model_name="llama3-70b-8192"
            )
            llm.invoke("Test LLM connection.")
        except Exception as e:
            st.error(f"❌ Error: Invalid Groq API key or model unavailable. Details: {e}")
            st.stop()

        # --- 2. Scrape and Extract Job Information ---
        try:
            loader = WebBaseLoader(job_url)
            page_data = loader.load().pop().page_content
        except Exception as e:
            st.error(f"❌ Error scraping URL. Please check the URL. Error: {e}")
            st.stop()
        
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE: {page_data}
            ### INSTRUCTION: Extract the job posting details and return them in JSON format with keys: `role`, `experience`, `skills` and `description`. Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        json_parser = JsonOutputParser()
        chain_extract = prompt_extract | llm | json_parser
        job = chain_extract.invoke(input={'page_data': page_data})
        
        # --- 3. Find Relevant Portfolio Links ---
        job_skills = job.get('skills', [])
        relevant_links = collection.query(
            query_texts=job_skills, 
            n_results=2
        ).get('metadatas', [])

        # --- 4. Generate Cold Email ---
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION: {job_description}
            ### INSTRUCTION: You are Mohan, a business development executive at AtliQ. Write a cold email to the client, describing AtliQ's capabilities in fulfilling their needs. Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | llm | StrOutputParser()
        email_content = chain_email.invoke({
            "job_description": str(job), 
            "link_list": relevant_links
        })

        st.success("🎉 Cold mail generated successfully!")
        st.markdown("---")
        st.subheader("Generated Cold Mail")
        st.code(email_content)
