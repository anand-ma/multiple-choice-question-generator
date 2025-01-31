from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain import LLMChain

import streamlit as st
import os

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

gpt_mini_model = ChatOpenAI(model_name="gpt-4o-mini")

few_shot_template = """Give {num} question on the topic of {topic} in {lang}
Follow the below example:
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

# Create LLM chain using the prompt template and model
question_gen_chain = few_shot_prompt | gpt_mini_model


# Frond End Code

st.header("Questions Generator 📚❓✍️")
st.subheader("Generates Multiple Choice questions with answers and explanation")


topic = st.text_input(
    label="Topic",
    value="Tamil",
    max_chars=15,
    type="default",
    placeholder="Type a topic to generate questions",
    disabled=False
)

number = st.number_input("Number of Questions", min_value = 1, max_value = 10, value = 1, step = 1)

lang = st.text_input(
    label="Language",
    value="English",
    max_chars=15,
    type="default",
    disabled=False
)



if st.button("Generate"):
    mcq = question_gen_chain.invoke({
      "num" : number,
      "topic" : topic,
      "lang" : lang
    })
    st.write(mcq.content)
