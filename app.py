import streamlit as st
import PyPDF2
import spacy
from io import BytesIO

# Load English NER model from spaCy
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_country_names(text):
    doc = nlp(text)
    countries = set()
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE is the label for geopolitical entities
            countries.add(ent.text)
    return countries

# Streamlit UI
st.title("PDF Text and NER")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.success("File successfully uploaded!")
    text = extract_text_from_pdf(BytesIO(uploaded_file.getvalue()))
    countries = extract_country_names(text)

    st.subheader("Extracted Text:")
    st.text(text)

    st.subheader("Countries Mentioned:")
    st.write(countries)


