from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain 
from langchain_core.prompts import PromptTemplate

import streamlit as st
import os

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY'] # comes from streamlit cloud from secrets

gpt_mini_model = ChatOpenAI(model_name="gpt-4o-mini")

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash-latest")

few_shot_template = """Give {num} question on the topic of {topic} in {lang}
Follow the below example and make topic bold and mention only once:
Topic: Solar System

Q: Which planet is known as the "Red Planet"?

A) Venus
B) Mars
C) Jupiter
D) Saturn

Correct Answer: B

explanation: ...
"""

few_shot_prompt = PromptTemplate(template = few_shot_template, input_variables=['num','topic', 'lang'])




# Frond End Code

st.header("Questions Generator 📚❓✍️")
st.subheader("Generates Multiple Choice questions with answers and explanation")

model = st.selectbox(
    "Select AI Model",
    ("Gemini", "GPT"),
)

topic = st.text_input(
    label="Topic",
    value="Tamil",
    max_chars=100,
    type="default",
    placeholder="Type a topic to generate questions",
    disabled=False
)

number = st.number_input("Number of Questions", min_value = 1, max_value = 20, value = 1, step = 1)

lang = st.text_input(
    label="Language",
    value="English",
    max_chars=15,
    type="default",
    disabled=False
)

if st.button("Generate"):

    if model == "GPT":
        # Create LLM chain using the prompt template and model
        question_gen_chain = few_shot_prompt | gpt_mini_model
    elif model == "Gemini":
        question_gen_chain = few_shot_prompt | gemini_model
    
    mcq = question_gen_chain.invoke({
      "num" : number,
      "topic" : topic,
      "lang" : lang
    })
    st.write(mcq.content)
