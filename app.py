import spacy
import PyPDF2
from spacy_streamlit import visualize_ner
import streamlit as st
from io import BytesIO

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def main():
    # Streamlit UI
    st.title("PDF and NER Detection")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        text = extract_text_from_pdf(BytesIO(uploaded_file.getvalue()))

        # Load English NER model from spaCy
        nlp = spacy.load("en_core_web_sm")
        #doc = nlp("Sundar Pichai is the CEO of Google.")
        doc = nlp(text)
        visualize_ner(doc, labels=nlp.get_pipe("ner").labels)

if __name__ == "__main__":
    main()