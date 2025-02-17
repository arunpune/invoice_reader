import streamlit as st
from dotenv import load_dotenv
from logging_config import logger
from utils import create_docs
from test import process_image
import pandas as pd

# Set the page configuration
st.set_page_config(page_title="Invoice Extraction Bot", layout="wide")

def add_custom_css():
    """Add custom CSS for styling, including a background image and center-aligned title."""
    st.markdown(
        """
        <style>
        

        /* Center-aligned title and subheader */
        .center-aligned-title {
            text-align: center;
            
            text-shadow: 2px 2px 4px #000000;
        }

        .center-aligned-subheader {
            text-align: center;
            
            text-shadow: 1px 1px 3px #000000;
        }

        /* File uploader and button styling */
        .stFileUploader, .stButton button {
            border-radius: 5px;
            padding: 10px;
            font-weight: bold;
        }

        .stFileUploader {
            background-color: rgba(255, 255, 255, 0.8);
            
        }

        /* Spinner styling */
        .css-1bk0rdt {
            
        }

        /* Dataframe table styling */
        .stDataFrame {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
        }

        /* Informational section styling */
        .info-section {
            background-color: rgba(white);
            
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.8);
        }

        .info-section h3 {
            color: #ffcc00;
        }

        .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stDownloadButton>button {
        background-color: #008CBA; /* Blue */
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: 0.3s;
    }
    .stDownloadButton>button:hover {
        background-color: #007bb5;
    }


        </style>
        """,
        unsafe_allow_html=True,
    )

    

    # Custom CSS for buttons



def main():
    load_dotenv()

     # Add custom CSS for styling
    add_custom_css()

    # App Title and Subtitle (center-aligned using CSS class)
    st.markdown('<h1 class="center-aligned-title">Invoice Extraction Bot</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="center-aligned-subheader">I can help you in extracting invoice data effortlessly.</h2>', unsafe_allow_html=True)


    uploaded_files = st.file_uploader(
        "Upload invoices here (PDFs or images allowed)", 
        accept_multiple_files=True
    )

    submit = st.button("Extract Data")

    if submit:
        if uploaded_files:
            with st.spinner("Processing..."):
                try:
                    # Process each file based on its type
                    extracted_data = []
                    for uploaded_file in uploaded_files:
                        file_type = uploaded_file.type

                        if file_type == "application/pdf":
                            logger.info(f"Processing PDF file: {uploaded_file.name}")
                            data = create_docs([uploaded_file])  # Expect a DataFrame
                        elif file_type in ["image/jpeg", "image/png"]:
                            logger.info(f"Processing image file: {uploaded_file.name}")
                            data_dict = process_image(uploaded_file)  # Expect a dict
                            if data_dict:
                                data = pd.DataFrame([data_dict])  # Convert dict to DataFrame
                            else:
                                data = None
                        else:
                            logger.warning(f"Unsupported file type: {uploaded_file.name} ({file_type})")
                            st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping...")
                            continue

                        if data is not None:
                            extracted_data.append(data)

                    # Combine and display extracted data
                    if extracted_data:
                        combined_data = pd.concat(extracted_data, ignore_index=True)
                        st.write(combined_data.head())

                        data_as_csv = combined_data.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download data as CSV",
                            data_as_csv,
                            "extracted_data.csv",
                            "text/csv",
                            key="download-csv",
                        )
                        st.success("Data extraction complete!")
                    else:
                        st.warning("No data extracted from the uploaded files.")

                except Exception as e:
                    logger.error(f"Error during data extraction: {str(e)}", exc_info=True)
                    st.error(f"Error during extraction: {str(e)}")
        else:
            st.warning("Please upload one or more files.")

    # Informational section
    st.markdown(
        """
        <div class="info-section">
            <h3>What is Invoice Data Extraction using LLM?</h3>
            <p>Invoice data extraction using a Large Language Model (LLM) involves leveraging advanced machine learning techniques to automatically identify, extract, and process key information from invoices. This includes details such as invoice numbers, dates, vendor names, amounts, and line-item data.</p>
            <p>By using LLMs, the process becomes highly accurate and efficient, reducing the manual effort involved in handling large volumes of invoices. It enables businesses to streamline their workflows and improve data accuracy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == '__main__':
    logger.info("Invoice Extraction Bot started.")
    main()
