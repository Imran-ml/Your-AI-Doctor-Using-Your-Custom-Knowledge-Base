import streamlit as st
from ai_assistant import ai_doctor_chat

# Display title
st.markdown("<h1 style='text-align: center;'>Your AI Doctor Using Your Custom Knowledge Base &#129302;</h1>", unsafe_allow_html=True)

# Create layout with two columns
left_column, right_column = st.columns([1, 3])

# Display image in the left column
left_column.image("ai_doctor_img.jpg", width=200, use_column_width="auto")

# Create a text input box for the OpenAI key
openai_key = right_column.text_input('Enter your OpenAI Key', type='password')

query = right_column.text_input('Enter your query', type='default')
submit = right_column.button('Submit')
if submit:
    if query and openai_key:
        try:
            with st.spinner('Processing your query...'):
                response = ai_doctor_chat(openai_key, query)
            right_column.write(response)
        except Exception as e:
            right_column.error(f'An error occurred: {e}', icon='🚨')
    else:
        right_column.error('Please enter your OpenAI key and Query both!', icon="🚨")

st.markdown("---")
st.write("Connect with me:")

kaggle, linkedin, google_scholar, youtube, github = st.columns(5)

kaggle.markdown("[Kaggle](https://www.kaggle.com/muhammadimran112233)")
linkedin.markdown("[LinkedIn](https://www.linkedin.com/in/muhammad-imran-zaman)")
google_scholar.markdown("[Google Scholar](https://scholar.google.com/citations?user=ulVFpy8AAAAJ&hl=en)")
youtube.markdown("[YouTube](https://www.youtube.com/@consolioo)")
github.markdown("[GitHub](https://github.com/Imran-ml)")
