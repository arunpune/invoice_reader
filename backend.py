import easyocr
import numpy as np
import cv2
import re
import json
import pandas as pd
from pypdf import PdfReader
from io import BytesIO
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from pdf2image import convert_from_bytes
from logging_config import logger

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    logger.critical("GROQ_API_KEY is not set. Please check your .env file.")
    raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
logger.info("Initialized Groq model.")

reader = easyocr.Reader(['en'])  # EasyOCR for OCR fallback


def get_pdf_text_with_ocr(pdf_doc):
    """Extract text from a PDF document, with OCR fallback for scanned PDFs."""
    text = ""
    try:
        logger.info("Starting PDF text extraction.")
        pdf_reader = PdfReader(BytesIO(pdf_doc))
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if text.strip():
            logger.info("PDF text extraction completed.")
            return text
    except Exception as e:
        logger.warning(f"Text extraction from PDF failed: {e}")

    # If text extraction fails, use OCR
    logger.info("Falling back to OCR for scanned PDF.")
    pdf_images = convert_from_bytes(pdf_doc)
    for img in pdf_images:
        ocr_result = reader.readtext(np.array(img))
        text += " ".join([item[1] for item in ocr_result])
    return text


def process_image(image_file):
    """Process an image file to extract text using OCR."""
    try:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Perform OCR
        result = reader.readtext(image)
        raw_text = " ".join([detection[1] for detection in result])
        return raw_text
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None


def extracted_data(raw_text):
    """Extract structured data from raw text using Groq API."""
    template = """
    Extract all the following values: invoice no., Description, Quantity, Date,
    Unit price, Amount, Total, email, phone number, and address from this data: {raw_text}

    If any field is missing, return 'N/A'. Remove any dollar symbols.

    Expected output:
    {{
        'Invoice no.': 'N/A',
        'Description': 'N/A',
        'Quantity': 'N/A',
        'Date': 'N/A',
        'Unit price': 'N/A',
        'Amount': 'N/A',
        'Total': 'N/A',
        'Email': 'N/A',
        'Phone number': 'N/A',
        'Address': 'N/A'
    }}
    """
    prompt = template.format(raw_text=raw_text)
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


def create_docs(user_file_list):
    """Process the list of uploaded files (PDFs or images) and extract structured data."""
    df = pd.DataFrame(columns=['Invoice no.', 'Description', 'Quantity', 'Date', 
                               'Unit price', 'Amount', 'Total', 'Email', 'Phone number', 'Address'])

    for uploaded_file in user_file_list:
        logger.info(f"Processing file: {uploaded_file.name}")
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            raw_data = get_pdf_text_with_ocr(uploaded_file.read())
        elif file_type in ["image/jpeg", "image/png"]:
            raw_data = process_image(uploaded_file)
        else:
            logger.warning(f"Unsupported file type: {uploaded_file.name} ({file_type})")
            continue

        if raw_data:
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
        else:
            logger.warning(f"No text extracted from file: {uploaded_file.name}")
    
    logger.info("Data extraction process completed.")
    return df