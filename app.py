import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
from transformers import pipeline
from pdf2image import convert_from_bytes
from docx import Document
import imutils
from imutils.perspective import four_point_transform

st.title("OCR Document Categorizer")

uploaded_file = st.file_uploader(
    "Upload a document", type=["jpg","png","jpeg","pdf","docx"]
)

if uploaded_file is not None:
    text = ""

    # pdf
    if uploaded_file.type == "application/pdf":
        pages = convert_from_bytes(uploaded_file.read())
        for page in pages:
            image = np.array(page)
            custom_config = r'--psm 6'
            text += pytesseract.image_to_string(
                image, lang='eng+por+tha', config=custom_config
            ) + "\n"

    # document
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        text = "\n".join([p.text for p in doc.paragraphs])

    # image
    else:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        edged = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break

        if doc_cnt is not None:
            cropped = four_point_transform(image, doc_cnt.reshape(4,2))
        else:
            cropped = image

        st.image(cropped, caption="Cropped Document", use_container_width=True)

        custom_config = r'--psm 6'
        text = pytesseract.image_to_string(cropped, lang='eng+por+tha', config=custom_config)

 
    st.subheader("Extracted Text")
    st.text(text)

   
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["Invoice","Resume","Report","Letter","Email","Form","Contract","Manual","Other"]

    
    result = classifier(text[:2000], candidate_labels)  
    st.subheader("Categorized As")
    st.write(result['labels'][0])

    
