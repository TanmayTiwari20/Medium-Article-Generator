import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ["OPENAI_API_KEY"] = apikey

st.title("Medium Article Writer")
topic = st.text_input("Enter a topic:")

# A prompt with defined template
title_template = PromptTemplate(
    input_variables=["topic"],
    template="Give me a medium article title on {topic}",
)

article_template = PromptTemplate(
    input_variables=["title_template"],
    template="Give me a medium article for the title : {title_template}",
)
# OpenAI llm instance
# Temperature (0 = precision, 1 = creativity)
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
article_chain = LLMChain(llm=llm, prompt=article_template, verbose=True)

overall_chain = SimpleSequentialChain(chains=[title_chain, article_chain], verbose=True)

if topic:
    # Simple Chain
    response = overall_chain.run(topic)
    st.write(response)
