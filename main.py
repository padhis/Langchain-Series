import os

from langchain.llms import OpenAI

import streamlit as st

os.environ['OPENAI_API_KEY'] = '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'

#streamlit framework
st.title('Langchain Demo with OPENAI API')

input_text = st.text_input('Search any topic of your choice')

#OPENAI LLMS
llm = OpenAI(temperature = 0.8)

if input_text:
    st.write(llm(input_text))