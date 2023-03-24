




import os
import requests
from io import BytesIO
from PyPDF2 import PdfReader
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import pkg_resources
import streamlit as st
import numpy as np



messages = [
    {"role": "system", "content": "You are SummarizeGPT, a large language model whose expertise is reading and summarizing scientific papers."}
]

class Chatbot():
    
    def parse_paper(self, pdf):
        print("Parsing paper")
        number_of_pages = len(pdf.pages)
        print(f"Total number of pages: {number_of_pages}")
        paper_text = []
        for i in range(number_of_pages):
            page = pdf.pages[i]
            page_text = []

            def visitor_body(text, cm, tm, fontDict, fontSize):
                x = tm[4]
                y = tm[5]
                # ignore header/footer
                if (y > 50 and y < 720) and (len(text.strip()) > 1):
                    page_text.append({
                    'fontsize': fontSize,
                    'text': text.strip().replace('\x03', ''),
                    'x': x,
                    'y': y
                    })

            _ = page.extract_text(visitor_text=visitor_body)

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
        # print(paper_text)
        return paper_text

    def paper_df(self, pdf):
        print('Creating dataframe')
        filtered_pdf= []
        for row in pdf:
            if len(row['text']) < 30:
                continue
            filtered_pdf.append(row)
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
        # This is going to create embeddings for subsets of the PDF
        embeddings = df.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
        df["embeddings"] = embeddings
        print('Done calculating embeddings')
        print(pkg_resources.get_distribution("openai").version)
        return df
    
        

    def search_embeddings(self, df, query, n=3, pprint=True):
        
        # Step 1. Get an embedding for the question being asked to the PDF
        query_embedding = get_embedding(
            query,
            engine="text-embedding-ada-002"
        )
        # Step 2. Create a column in the dataframe that contains the cosine similarity (distance) between the query and the text in the dataframe
        df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
        # Step 3. Sort the dataframe by the similarity column
        results = df.sort_values("similarity", ascending=False, ignore_index=True)
        # make a dictionary of the the first three results with the page number as the key and the text as the value. The page number is a column in the dataframe.
        results = results.head(n)
        global sources 
        sources = []
        for i in range(n):
            # append the page number and the text as a dict to the sources list
            sources.append({'Page '+str(results.iloc[i]['page']): results.iloc[i]['text'][:150]+'...'})
        print(sources)
        return results.head(n)
    
    def create_prompt(self, df, user_input):
        result = self.search_embeddings(df, user_input, n=3)
        print(result)
        prompt = """You are a large language model whose expertise is reading and and providing answers about research papers. 
        You are given a query and a series of text embeddings from a paper in order of their cosine similarity to the query.
        You must take the given embeddings, as well as what you know from your model weights and knowledge of various fields of research to provide an answer to the query
        that lines up with what was provided in the text.
            
            Given the question: """+ user_input + """
            
            and the following embeddings as data: 
            
            1.""" + str(result.iloc[0]['text']) + """
            2.""" + str(result.iloc[1]['text']) + """
            3.""" + str(result.iloc[2]['text']) + """

            Return a detailed answer based on the paper. If the person asks you to summarize what is in the paper, do your best to provide a summary of the paper.:"""

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

    def reply(self, prompt):
        print(prompt)
        prompt = self.create_prompt(df, prompt)
        return self.gpt(prompt)

def process_pdf(file):
    print("Processing pdf")
    pdf = PdfReader(BytesIO(file))
    chatbot = Chatbot()
    paper_text = chatbot.parse_paper(pdf)
    global df
    df = chatbot.paper_df(paper_text)
    df = chatbot.calculate_embeddings(df)
    print("Done processing pdf")

def download_pdf(url):
    chatbot = Chatbot()
    r = requests.get(str(url))
    print(r.headers)
    pdf = PdfReader(BytesIO(r.content))
    paper_text = chatbot.parse_paper(pdf)
    global df
    df = chatbot.paper_df(paper_text)
    df = chatbot.calculate_embeddings(df)
    print("Done processing pdf")

def show_pdf(file_content):
    base64_pdf = base64.b64encode(file_content).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def main():
    st.title("Research Paper Guru")
    st.subheader("Upload PDF or Enter URL")
    
    pdf_option = st.selectbox("Choose an option:", ["Upload PDF", "Enter URL"])
    chatbot = Chatbot()

    if pdf_option == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            file_content = uploaded_file.read()
            process_pdf(uploaded_file.read())
            st.success("PDF uploaded and processed successfully!")
            show_pdf(file_content)

    elif pdf_option == "Enter URL":
        url = st.text_input("Enter the URL of the PDF:")
        if url:
            if st.button("Download and process PDF"):
                try:
                    r = requests.get(str(url))
                    content = r.content
                    download_pdf(url)
                    st.success("PDF downloaded and processed successfully!")
                    show_pdf(content)
                except Exception as e:
                    st.error(f"An error occurred while processing the PDF: {e}")

    query = st.text_input("Enter your query:")
    if query:
        if st.button("Get answer"):
            response = chatbot.reply(query)
            st.write(response['answer'])
            st.write("Sources:")
            for source in response['sources']:
                st.write(source)

if __name__ == "__main__":
    main()
    