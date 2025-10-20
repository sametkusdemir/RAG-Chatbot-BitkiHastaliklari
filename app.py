import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Arayüz Yapılandırması ---
st.set_page_config(
    page_title="Bitki Hastalıkları RAG Chatbotu (Gemini)",
    layout="wide"
)
st.title("🌿 Akbank GenAI Projesi: Bitki Hastalıkları Bilgi Asistanı")
st.markdown("RAG (Retrieval Augmented Generation) mimarisi ile desteklenen bu chatbot, tarımsal veri tabanından bilgi çekerek sorularınızı yanıtlar.")

# --- RAG Bileşenlerinin Yüklenmesi (Önbelleğe Alınmıştır) ---

@st.cache_resource
def setup_rag_system():
    """
    RAG sisteminin tüm bileşenlerini (DB, LLM, Zincir) yükler ve önbelleğe alır.
    """
    # 1. API Anahtarının Kontrolü
    if "GEMINI_API_KEY" not in os.environ:
        st.error("Lütfen GEMINI_API_KEY ortam değişkenini ayarlayın ve uygulamayı yeniden başlatın.")
        return None, None
    
    # 2. Vektör Veri Tabanını Yükleme
    PERSIST_DIR = "./chroma_db"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        # Chroma DB'nin daha önce oluşturulmuş olması gerekir
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Vektör Veri Tabanı yüklenirken hata oluştu. 'prepare_db.py' dosyasını çalıştırdığınızdan emin olun. Hata: {e}")
        return None, None

    # 3. LLM ve Prompt
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    prompt_template = """
    Sen tarımsal hastalıklar konusunda uzman bir yapay zeka asistanısın. Görevin,
    kullanıcının sorusunu SADECE aşağıda verilen bağlam (context) bilgisine göre
    detaylı ve bilgilendirici bir şekilde yanıtlamaktır. Eğer verilen bağlamda
    sorunun cevabı YOKSA, "Üzgünüm, bu sorunun cevabı elimdeki bitki hastalıkları
    bilgi tabanında bulunmamaktadır." diye cevap vermelisin. Cevaplarını Türkçe ver.

    --- BAĞLAM ---
    {context}
    ---

    Kullanıcının Sorusu: {question}
    Cevap:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 4. LangChain RAG Zinciri
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# RAG sistemini başlat
rag_chain, retriever = setup_rag_system()

# --- Chat Arayüzü Mantığı ---

if rag_chain:
    # Mesaj geçmişini başlatma (Streamlit Session State)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmiş mesajları görüntüleme
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcı girişi
    if prompt_input := st.chat_input("Bitki hastalığınız hakkında bilgi alın..."):
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Asistan cevabını al ve ekle
        with st.chat_message("assistant"):
            with st.spinner("Bilgi tabanı taranıyor ve cevap üretiliyor..."):
                # RAG zincirini çalıştırma
                response = rag_chain.invoke(prompt_input)
                st.markdown(response)

        # Cevabı session state'e kaydet
        st.session_state.messages.append({"role": "assistant", "content": response})

        # *Opsiyonel: Hangi kaynakları çektiğini gösteren bir Sidebar ekleme*
        with st.sidebar:
            st.header("Çekilen Kaynaklar (Retrieval)")
            docs = retriever.invoke(prompt_input)
            for i, doc in enumerate(docs):
                st.subheader(f"Kaynak {i+1}")
                st.caption(f"Puan: {doc.metadata.get('_score', 'N/A')}")
                st.text(doc.page_content[:250] + "...") # İlk 250 karakteri göster
