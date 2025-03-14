import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document 
from youtube_transcript_api import YouTubeTranscriptApi


## Streamlit App
st.set_page_config(page_title="LangChain: Summarize a Video from YouTube", page_icon="🤖", layout="wide")
st.title("🤖 Summarize a Video from YouTube")
st.subheader("Give Me The Video's URL")


## Get the Groq API Key and the URL field to summarize
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")

url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Te voy a pasar un contenido, debes identificar el idioma del contenido (pero no lo menciones en tu respuesta) y crea un resumen del siguiente contenido en 300 palabras:
<Contenido>
{text}
</Contenido>
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


if st.button("Summarize"):
    if not groq_api_key:
        st.error("Please enter your Groq API Key")
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        try:
            ## Gemma Model using groq api
            llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
            with st.spinner("Summarizing..."):
                # Loading the website or youtube video data
                if "youtube.com" in url:
                    video_id = url.split("v=")[-1]
                    transcript = YouTubeTranscriptApi.get_transcript(video_id=video_id,languages=['es', 'it', 'fr', 'de', 'en', 'pl'])
                    text = " ".join([entry['text'] for entry in transcript])
                    docs = [Document(page_content=text)]

                ##   Chain For Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.invoke(docs)

                st.video(url)
                st.success(docs)
                st.success(output_summary['output_text'])
        except Exception as e:
            st.exception(f"Exeption: {e}")