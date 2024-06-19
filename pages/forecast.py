import streamlit as st
import google.generativeai as genai
from pathlib import Path
import hashlib
import os
from dotenv import load_dotenv, dotenv_values

load_dotenv('.env')

ss = st.session_state
key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key= key ) 

df = ss.df
df_new = ss.df_new
stock_name = ss.stock_name


generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

system_prompt = f" looking at the historic stock data for company {stock_name}, you are a consultant and you need to state the risks ,benefits and future potential involved from investing in this stock, you should not look for any latest news and answer should be based on only the stock data. In the end add a disclaimer . Here is your data {df.to_string()}"

prompt_parts =[system_prompt]

st.set_page_config(page_title='Stonks', page_icon ='📈')
st.title('Stock Market Analysis and Investing advise 📊')
submit_btn = st.button('Generate Text')

if submit_btn:
    output = model.generate_content(prompt_parts)
    ss.gen_op = output.text

    
if 'gen_op' in ss:
    st.write(ss.gen_op)

