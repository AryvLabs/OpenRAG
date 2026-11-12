from pypdf import PdfReader
import re

def load_pdf():
    print ("Loading the PDF File")
    readBuffer = PdfReader("corpus/corpus.pdf")
    finalText = ""
    for page in readBuffer.pages:
        finalText += page.extract_text() + "\n"
    processedText = re.split('\n\n\n',finalText)
    return [text for text in processedText if (text != "")]

def load_txt():
    print ("Loading the TXT File")
    with open('corpus/corpus.txt', 'r') as file:
        finalText = file.read()
        processedText = re.split('\n\n\n',finalText)
        return [text for text in processedText if (text != "")]

def load_corpus(input_filetype):
    if input_filetype.lower() == "pdf":
        buffer = load_pdf()
        return buffer
    elif input_filetype.lower() == "txt":
        buffer = load_txt()
        return buffer
    else:
        raise Exception ("Filetype not recognised.")