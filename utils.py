import pandas as pd
import re
import json
from pypdf import PdfReader
from io import BytesIO
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from logging_config import logger

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    logger.critical("GROQ_API_KEY is not set. Please check your .env file.")
    raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
logger.info("Initialized Groq model.")

def get_pdf_text(pdf_doc):
    """Extract text from a PDF document."""
    text = ""
    try:
        logger.info("Starting PDF text extraction.")
        pdf_reader = PdfReader(BytesIO(pdf_doc))
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            logger.debug(f"Page {i+1} Text Extracted: {page_text[:500]}")
            text += page_text
        logger.info("PDF text extraction completed.")
    except Exception as e:
        logger.error(f"Error reading PDF: {e}", exc_info=True)
    return text


def extracted_data(pages_data):
    """Extract structured data from the PDF text using Groq API."""
    template = """
    Extract all the following values: invoice no., Description, Quantity, date,
    Unit price, Amount, Total, email, phone number, and address from this data: {pages}

    Expected output: remove any dollar symbols
    {{
        'Invoice no.': '2001321',
        'Description': 'HP Laptop',
        'Quantity': '1',
        'Date': '5/4/2023',
        'Unit price': '500.00',
        'Amount': '500.00',
        'Total': '500.00',
        'Email': 'sharathkumarraju@proton.me',
        'Phone number': '8888888888',
        'Address': 'Hyderabad, India'
    }}
    """
    prompt = template.format(pages=pages_data)
    try:
        logger.info("Sending data to Groq API for extraction.")
        response = llm.predict(text=prompt, temperature=0.1)
        logger.debug(f"Groq API Response: {response}")

        if not response:
            raise ValueError("No response from Groq API")

        json_match = re.search(r'{.*}', response, re.DOTALL)
        if json_match:
            json_data = json_match.group(0)
            return json_data
        else:
            raise ValueError("No valid JSON content found in Groq API response")
    except Exception as e:
        logger.error(f"Error during data extraction: {e}", exc_info=True)
        return None


def create_docs(user_pdf_list):
    """Process the list of uploaded PDFs and extract structured data."""
    df = pd.DataFrame(columns=['Invoice no.', 'Description', 'Quantity', 'Date', 
                               'Unit price', 'Amount', 'Total', 'Email', 'Phone number', 'Address'])

    for uploaded_file in user_pdf_list:
        logger.info(f"Processing file: {uploaded_file.name}")
        raw_data = get_pdf_text(uploaded_file.read())
        logger.debug(f"Extracted Raw Data: {raw_data[:500]}")

        llm_extracted_data = extracted_data(raw_data)

        if llm_extracted_data:
            try:
                cleaned_data = llm_extracted_data.replace("'", '"').replace("\n", "")
                data_dict = json.loads(cleaned_data)
                logger.debug(f"Parsed Data Dict: {data_dict}")
                df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error during parsing: {e}", exc_info=True)
        else:
            logger.warning("No data extracted from the LLM response.")
    
    logger.info("Data extraction process completed.")
    return df