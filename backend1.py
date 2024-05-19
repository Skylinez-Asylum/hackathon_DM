from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from groq import Groq
def response_query(user_query):
    loader1=PyPDFLoader("disaster_management_in_india.pdf")
    docs1=loader1.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs1)
    cohere_api_key = "leTByPB6J9FNbFIup99z08dhPaFwiquAlRqScvJv"
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=cohere_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    groq_api_key="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"
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    template = """Use the following pieces of context to answer the question at the end.
            Say that you don't know when asked a question you don't know, donot make up an answer. Be precise and concise in your answer.
            dont say based on the given context. you are a disaster management helper
            you need to answer the user queries and help them and comfort them
            if the situation demands
            {context}

            Question: {input}

            Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)
    from langchain.chains.combine_documents import create_stuff_documents_chain
    document_chain=create_stuff_documents_chain(chat,prompt)
    retriever=vectorstore.as_retriever()
    from langchain.chains import create_retrieval_chain
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    #user_query="Who is the HOD of CS department"

    res=retrieval_chain.invoke({"input":user_query})

    return res['answer']


def response_selector(user_query):
    client = Groq(
             api_key='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',
)
    chat_completion = client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": f'''classify the following message '{user_query}' into category 0 or 1 or 2 or 3 or 4
                          if the messgae  is about current weather then category 0
                          if the message is  about situations like floods,wildfire,etc then category 1 
                          if the message is about daily news or local news about a place like kollam then category 2
                          if the message is about flood prediction for the year then category 3
                          if the message is about nearest relief camps or something like that then category 4
                          if the message is about email or phone number or contact details then category 5 
                          if the message is about  anything else category 6
                          return the category number only''',
        }
         ],
        model="llama3-8b-8192",
     )
    response_message = chat_completion.choices[0].message.content
    return response_message    

def palce_finder(user_query):
    matches = re.findall(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b|\b[A-Z][a-z]+\b",user_query)
    return matches    


import os
def response_tavily(user_query):
    os.environ["TAVILY_API_KEY"] = "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"
    from langchain_community.retrievers import TavilySearchAPIRetriever

    retriever = TavilySearchAPIRetriever(k=3)

    groq_api_key="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"
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.
            dont say according to context. reply like a human
    Context: {context}

    Question: {question}"""
    )
    chain = (
        RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
        | prompt
        | chat
        | StrOutputParser()
    )
    return chain.invoke({"question": user_query})


def response_from_news(user_query):
    urls = [
    "https://www.manoramaonline.com/district-news/thiruvananthapuram.html",
    "https://www.manoramaonline.com/district-news/kollam.html",
    "https://www.manoramaonline.com/district-news/pathanamthitta.html",
    "https://www.manoramaonline.com/district-news/alappuzha.html",
    "https://www.manoramaonline.com/district-news/kottayam.html",
    "https://www.manoramaonline.com/district-news/idukki.html",
    "https://www.manoramaonline.com/district-news/ernakulam.html",
    "https://www.manoramaonline.com/district-news/thrissur.html",
    "https://www.manoramaonline.com/district-news/palakkad.html",
    "https://www.manoramaonline.com/district-news/kozhikode.html",
    "https://www.manoramaonline.com/district-news/wayanad.html",
    "https://www.manoramaonline.com/district-news/malappuram.html",
    "https://www.manoramaonline.com/district-news/kasargod.html",
    "https://www.manoramaonline.com/district-news/kannur.html"
]
    loader1=WebBaseLoader(urls)
    docs1=loader1.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs1)
    cohere_api_key = "leTByPB6J9FNbFIup99z08dhPaFwiquAlRqScvJv"
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=cohere_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    groq_api_key="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"
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    template = """Use the following pieces of context to answer the question at the end.
            Say that you don't know when asked a question you don't know, donot make up an answer. Be precise and concise in your answer.
            dont say based on the given context. you are a chatbot and u have all the data about all the districts in kerela
            ans all queries from user especiallly about disasters and weather and dont say here are some news. just answer your query like a human
            {context}

            Question: {input}

            Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)
    from langchain.chains.combine_documents import create_stuff_documents_chain
    document_chain=create_stuff_documents_chain(chat,prompt)
    retriever=vectorstore.as_retriever()
    from langchain.chains import create_retrieval_chain
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    #user_query="Who is the HOD of CS department"

    res=retrieval_chain.invoke({"input":user_query})

    return res['answer']

from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # Radius of earth in kilometers
    return distance

def nearest_relief_camp(latitude, longitude):
    """
    Find the nearest relief camp to the given coordinates
    """

    relief_camps=[(8, 76, 'Govt UPS Thekkumbhagom'),
 (8.2878007, 77.1894504, 'GVHSS AHSS, Nediyavila'),
 (8.7811, 76.9451, 'G U P S, Mankadu'),
 (8.7846295, 76.9497227, 'GHSS, Kummil'),
 (8.79, 76.68, 'Kalakkodu UPS'),
 (8.811043, 76.761009, 'Parippally Panchayath Community Hall'),
 (8.822932, 76.917989, 'Government Town L P School'),
 (8.828568, 76.856117, 'Marhaba Auditorium'),
 (8.835007, 76.925773, 'Government U P School'),
 (8.842605, 76.718137, 'Govt. High School Uliyanadu'),
 (8.864393, 76.689962, 'Mylakkad UPS'),
 (8.8920661, 76.6881336, 'GLPS MUTTAKAV'),
 (8.8929181, 76.6496564, 'G.L.P.S. Mukhathala'),
 (8.894647, 76.5779009, 'Govt Model Boys H.S.S'),
 (8.897099, 76.838839, 'Elamadu UP School'),
 (8.8978958, 76.6386724, 'Meenakshi Vilasam Govt. LP School'),
 (8.8998342, 76.6461488, 'G.L.P.S Cheriyela'),
 (8.9036598, 76.7125236, 'GHSS PALLIMON'),
 (8.903864, 76.935231, 'Govt H.S.S Karukone'),
 (8.9131575, 76.6559579, 'N.S.S.U.P.S Mukhathala'),
 (8.919289, 76.920847, 'Vadamon UPS, Anchal'),
 (8.9237512, 76.7100434, 'Nallila Govt UPS'),
 (8.926096, 76.91857, 'Challirickal Auditorium'),
 (8.93, 76.94, 'Delta Dental School'),
 (8.9395468, 76.6992644, 'RSMHS Pazhangalam'),
 (8.9429, 76.598, 'W. L. P. S Idavattom'),
 (8.945829, 76.588814, 'Prakkulam Govt LP School'),
 (8.9460505, 76.6614969, 'St Vincents School'),
 (8.9556653, 76.5340876, 'Govt LPS Neendakara'),
 (8.9631436, 76.8349081, 'Government LP School Valakom'),
 (8.9635, 76.6211, 'GHSS Panayil'),
 (8.965906, 77.069593, 'GLPS THENMALA'),
 (8.972462, 76.923842, 'Community Hall'),
 (8.98271, 76.924639, 'Govt L.P.S, Karavaloor'),
 (8.984338, 76.764857, 'Navodaya'),
 (8.986495, 76.912429, 'Govt L.P.S, Venchempu'),
 (8.990974, 77.015799, 'GLPS URUKUNNU'),
 (8.9931, 76.5335, 'B.J.MemmorialGovt.College'),
 (8.9942876, 76.5866871, 'Government Higher Secondary School Vettikkavala'),
 (8.998003, 76.755074, 'E V H S Neduvathoor'),
 (8.9981, 76.62, 'Govt. LPS Munroethuruthu'),
 (9, 76, 'Karunagappally Municipality'),
 (9.000353, 76.774148, 'Boys Higher Secondary School'),
 (9.005851, 76.778095, 'Marthomaschool'),
 (9.00744, 76.538405, 'Valiyam Central School'),
 (9.011, 76.601, 'Kanatharkunnam L.P.S'),
 (9.012431, 76.534675, 'Chittoor UPS'),
 (9.0153, 76.5651, 'SMVLPS, Padinjattakkara'),
 (9.018016, 76.724501, 'G W L P S, Thevalappuram'),
 (9.0231352, 76.8522763, 'A.P.P.M V H S School'),
 (9.0252, 76.688, 'G W LP S Vanivila'),
 (9.031726, 76.550183, 'SVPMHS Vadakkumthala'),
 (9.0388, 76.683, 'Glps Cherupoika'),
 (9.051125, 76.55601, 'GOVT SNTTC'),
 (9.053236, 76.701168, 'GWLPS Pangode'),
 (9.054782, 76.699512, 'SNGHSS Pangode'),
 (9.0551, 76.6453, 'St. Marys Hostel For Women, MTMM Nursing Hostel'),
 (9.0572332, 76.8522618, 'IGMVHSS, Manjakkala'),
 (9.074002, 76.588668, 'Vengara Govt LPS'),
 (9.0769917, 76.5380012, 'Alhana'),
 (9.0772165, 76.7965, 'Model LPS Pattazhy'),
 (9.0827171, 76.5240171, 'Chitumoola Masjid'),
 (9.0833, 76.6113, 'Govt LPS Eravichira'),
 (9.0877385, 76.857, 'St.Stephens HSS & HS'),
 (9.0990778, 76.4921058, 'YMCA Building'),
 (9.1, 76.79, 'G U P S Earathuvadakku'),
 (9.110479, 76.594978, 'Amrutha UPS'),
 (9.116, 76.4751, 'GFHSS Kuzhithura'),
 (9.1636591, 76.6221816, 'GHSS Sooranad'),
 (85935, 770515, 'TCNM Auditorium'),
 (9000466, 76.988622, 'GLPS EDAMON')]



    nearest_camp = None
    min_distance = float('inf')

    for camp in relief_camps:
        camp_lat, camp_lon, _ = camp
        distance = haversine(latitude, longitude, camp_lat, camp_lon)
        if distance < min_distance:
            min_distance = distance
            nearest_camp = camp

    return nearest_camp


