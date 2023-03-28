import os
import requests
from io import BytesIO
from PyPDF2 import PdfReader
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import streamlit as st
import numpy as np
import base64
import faiss



messages = [
    {"role": "system", "content": "You are SummarizeGPT, a large language model whose expertise is reading and summarizing scientific papers."}
]

class Chatbot():
    
    def parse_paper(self, pdf):
        # This function parses the PDF and returns a list of dictionaries with the text, 
        # font size, and x and y coordinates of each text element in the PDF
        print("Parsing paper")
        number_of_pages = len(pdf.pages)
        print(f"Total number of pages: {number_of_pages}")
        # This is the list that will contain all the text elements in the PDF and will be returned by the function
        paper_text = []
        
        for i in range(number_of_pages):
            # Iterate through each page in the PDF, and extract the text elements. pdf.pages is a list of Page objects.
            page = pdf.pages[i]
            # This is the list that will contain all the text elements in the current page
            page_text = []

            def visitor_body(text, cm, tm, fontDict, fontSize):
                # tm is a 6-element tuple of floats that represent a 2x3 matrix, which is the text matrix for the text.
                # The first two elements are the horizontal and vertical scaling factors, the third and fourth elements 
                # are the horizontal and vertical shear factors, and the fifth and sixth elements are the horizontal and vertical translation factors.
                
                # x and y are the coordinates of the text element
                x = tm[4]
                y = tm[5]
                
                # ignore header/footer, and empty text.
                # The y coordinate is used to filter out the header and footer of the paper
                # The length of the text is used to filter out empty text
                if (y > 50 and y < 720) and (len(text.strip()) > 1):
                    page_text.append({
                    # The fontsize is used to separate paragraphs into different elements in the paper_text list
                    'fontsize': fontSize,
                    # The text is stripped of whitespace and the \x03 character
                    'text': text.strip().replace('\x03', ''),
                    # The x and y coordinates are used to separate paragraphs into different elements in the paper_text list
                    'x': x,
                    'y': y
                    })

            # Extract the text elements from the page
            _ = page.extract_text(visitor_text=visitor_body)
            print(f'Page {i} text", {page_text}')

            
            blob_font_size = None
            blob_text = ''
            processed_text = []

            for t in page_text:
                if t['fontsize'] == blob_font_size:
                    blob_text += f" {t['text']}"
                    if len(blob_text) >= 2000:
                        processed_text.append({
                            'fontsize': blob_font_size,
                            'text': blob_text,
                            'page': i
                        })
                        blob_font_size = None
                        blob_text = ''
                else:
                    if blob_font_size is not None and len(blob_text) >= 1:
                        processed_text.append({
                            'fontsize': blob_font_size,
                            'text': blob_text,
                            'page': i
                        })
                    blob_font_size = t['fontsize']
                    blob_text = t['text']
                paper_text += processed_text
        print("Done parsing paper")
        print(paper_text)

        return paper_text

    def paper_df(self, pdf):
        print('Creating dataframe')
        filtered_pdf= []
        for row in pdf:
            # This will use the get method to safely access the 'text' key in the row dictionary, 
            # and if the key is not present, it will use an empty string as a default value. This 
            # should prevent a KeyError from occurring.
            if len(row.get('text', '')) < 30:
                continue
            filtered_pdf.append(row)
        print("Filtered paper_text", filtered_pdf)
        df = pd.DataFrame(filtered_pdf)
        print(df.shape)
        print(df.head)
        # remove elements with identical df[text] and df[page] values
        df = df.drop_duplicates(subset=['text', 'page'], keep='first')
        df['length'] = df['text'].apply(lambda x: len(x))
        print('Done creating dataframe')
        return df

    def calculate_embeddings(self, df):
        print('Calculating embeddings')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        embedding_model = "text-embedding-ada-002"
        # Get the embeddings for each text element in the dataframe
        embeddings = df.text.apply(lambda x: get_embedding(x, engine=embedding_model))
        embeddings = np.vstack(embeddings, dtype=np.float32)
        return embeddings    

    def search_embeddings(self, embeddings, df, query, n=3, pprint=True):
        
        # Step 1. Get an embedding for the question being asked to the PDF
        query_embedding = get_embedding(query, engine="text-embedding-ada-002")
        query_embedding = np.array(query_embedding, dtype=np.float32)
        # Step 2. Create a FAISS index and add the embeddings
        d = embeddings.shape[1]
        # Use the L2 distance metric
        index = faiss.IndexFlatL2(d)
        print("Embeddings shape:", embeddings.shape)
        print("Embeddings data type:", type(embeddings))
        index.add(embeddings)
        
        
        # Step 3. Search the index for the embedding of the question

        D, I = index.search(query_embedding.reshape(1,d), n)        
        
        # Step 4. Get the top n results from the dataframe
        results = df.iloc[I[0]]
        results['similarity'] = D[0]
        results = results.reset_index(drop=True)
        
        # Make a dictionary of the first n results with the page number as the key and the text as the value
        
        global sources 
        sources = []
        for i in range(n):
            # append the page number and the text as a dict to the sources list
            sources.append({'Page '+str(results.iloc[i]['page']): results.iloc[i]['text'][:150]+'...'})
        print(sources)
        return results.head(n)
    
    def create_prompt(self, embeddings, df, user_input):
        result = self.search_embeddings(embeddings, df, user_input, n=3)
        print(result)
        prompt = """
        You are Research Paper Guru
        The user is going to ask you a question about a research paper after uploading a PDF of the paper.
        You are a large language model whose expertise is reading and and providing answers to their queries, based on what you know about the subject as well as what you know about the text given to you.
            
            The user asks: """+ user_input + """
            
            And the information about the paper that is relevant to the question is: 
            
            1.""" + str(result.iloc[0]['text']) + """
            2.""" + str(result.iloc[1]['text']) + """
            3.""" + str(result.iloc[2]['text']) + """
            Knowing what you know about this answer, as well as being able to navigate this knowledge in conjuction with what is being said in the paper, provide an answer to the user. If the person asks you to summarize what is in the paper, do your best to provide a summary of the paper.
            The goal here is to keep the user happy and satisfied that you have given them the best answer to the question to the best of your knowledge. If necessary, you can also point them to outside resources for more information.:"""

        print('Done creating prompt')
        return prompt

    def gpt(self, prompt):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        print('got API key')
        messages.append({"role": "user", "content": prompt})
        r = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        answer = r['choices'][0]['message']['content']
        response = {'answer': answer, 'sources': sources}
        return response

    def reply(self, embeddings, user_input):
        print(user_input)
        prompt = self.create_prompt(embeddings, df, user_input)
        return self.gpt(prompt)

def process_pdf(file):
    print("Processing pdf")
    pdf = PdfReader(BytesIO(file))
    chatbot = Chatbot()
    paper_text = chatbot.parse_paper(pdf)
    global df
    df = chatbot.paper_df(paper_text)
    embeddings = chatbot.calculate_embeddings(df)
    print("Done processing pdf")
    return embeddings

def download_pdf(url):
    chatbot = Chatbot()
    r = requests.get(str(url))
    print(r.headers)
    pdf = PdfReader(BytesIO(r.content))
    paper_text = chatbot.parse_paper(pdf)
    global df
    df = chatbot.paper_df(paper_text)
    embeddings = chatbot.calculate_embeddings(df)
    print("Done processing pdf")
    return embeddings

def show_pdf(file_content):
    base64_pdf = base64.b64encode(file_content).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def main():
    st.title("Research Paper Guru")
    st.subheader("Upload PDF or Enter URL")
    embeddings = None
    pdf_option = st.selectbox("Choose an option:", ["Upload PDF", "Enter URL"])
    chatbot = Chatbot()

    if pdf_option == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            file_content = uploaded_file.read()
            embeddings = process_pdf(file_content)
            st.success("PDF uploaded and processed successfully!")
            show_pdf(file_content)

    elif pdf_option == "Enter URL":
        url = st.text_input("Enter the URL of the PDF:")
        if url:
            if st.button("Download and process PDF"):
                try:
                    r = requests.get(str(url))
                    content = r.content
                    embeddings = download_pdf(url)
                    st.success("PDF downloaded and processed successfully!")
                    show_pdf(content)
                except Exception as e:
                    st.error(f"An error occurred while processing the PDF: {e}")
    st.subheader("Ask a question about a research paper and get an answer with sources!")
    query = st.text_input("Enter your query:")
    if query:
        if st.button("Get answer"):
            if embeddings is not None:
                response = chatbot.reply(embeddings, query)
            else:
                st.warning("Please upload a PDF or enter a URL first.")
            st.write(response['answer'])
            st.write("Sources:")
            for source in response['sources']:
                st.write(source)

if __name__ == "__main__":
    main()
    
