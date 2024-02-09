import streamlit as st
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

#One problem with SimpleSequentialChain it will show you the last output
#To show the entire information use SequentialChain
from langchain.chains import SequentialChain

#Save the info in memory
from langchain.memory import ConversationBufferMemory

#Specify the openai key
os.environ['OPENAI_API_KEY'] = ''

#streamlit framework
st.title('Celebriti Search')
input_text = st.text_input('Enter a celebrity name')

llm = OpenAI(temperature = 0.8)

#prompt template
first_input_text = PromptTemplate(
    input_variable = ['name'],
    template = 'Tell me about {name}'
)

#Memory
person_memory = ConversationBufferMemory(input_key = 'name', memory_key = 'chat_history')
dob_memory = ConversationBufferMemory(input_key = 'person', memory_key = 'chat_history')
descr_memory = ConversationBufferMemory(input_key = 'dob', memory_key = 'description_history')

#chaim has to be provided for prompt template
chain1 = LLMChain(llm, prompt = first_input_text, verbose = True, output_key = 'person', memory = person_memory)

#second prompt template
second_prompt_template = PromptTemplate(
    input_variable = ['person'],
    template = 'When was {person} born'
)

#WRT to second prompt template we build another chain
chain2 = LLMChain(llm, prompt = second_prompt_template, verbose = True, output_key = 'dob', memory = dob_memory)

#third prompt template
third_prompt_template = PromptTemplate(
    input_variables = ['dob'],
    template = 'Mention five major events happend around {dob}'
)

#Create chain wrt to third prompt template
chain3 = LLMChain(llm, prompt = third_prompt_template, verbose = True, output_key = 'description', memory = descr_memory)

#Combine the chain
parent_chain = SequentialChain(chain = [chain1, chain2, chain3], input_variable = ['name'], 
                               output_variables = ['person', 'dob', 'description'], verbose = True)


if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info('descr_memory.buffer')