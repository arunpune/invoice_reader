import pytesseract
import cv2
import numpy as np
import os
import re
import json
import streamlit as st
from langchain_groq import ChatGroq

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def process_image(image_file):
    try:
        # Load the image file
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display the uploaded image in Streamlit
        st.image(image, channels="BGR", caption="Uploaded Image")

        # Preprocess the image (convert to grayscale, increase contrast, and threshold)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_image = clahe.apply(gray_image)

        # Apply adaptive thresholding
        processed_image = cv2.adaptiveThreshold(
            processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Deskew the image if necessary
        coords = np.column_stack(np.where(processed_image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (height, width) = processed_image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(processed_image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)

        # Perform OCR
        custom_config = r'--oem 3 --psm 3'  # OEM 3 is default, PSM 3 for sparse text or block-based text
        raw_text = pytesseract.image_to_string(rotated_image, config=custom_config)

        # Show the extracted text in Streamlit for debugging
        st.text_area("Extracted Text", raw_text, height=200)

        # If no text was extracted, return None
        if not raw_text.strip():
            st.warning("No text extracted from the image.")
            return None

        # Define the LLM prompt
        template = """
        Extract the following details: invoice no., Description, Quantity, Date,
        Unit price, Amount, Total, email, phone number, and address from the text: {raw_text}

        Expected output (JSON format):
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
        prompt = template.format(raw_text=raw_text)

        # Log the formatted prompt (you can add print statements for debugging)
        print("Formatted Prompt:\n", prompt)

        # Ensure the LLM is initialized (using your existing LLM integration)
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment variables.")
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        # Get the response from the LLM
        response = llm.predict(text=prompt, temperature=0.1)

        # Log the LLM response for debugging
        print("LLM Response:\n", response)

        # Extract JSON from the LLM response
        json_match = re.search(r'{.*}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            st.warning("No valid JSON returned from LLM.")
            return None

    except Exception as e:
        print(f"Error processing image: {e}")
        st.error(f"Error processing image: {e}")
        return None
