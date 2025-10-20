import os
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence
from langchain.schema.output_parser import StrOutputParser

# 1. Ön Koşul Kontrolü
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("Lütfen GEMINI_API_KEY ortam değişkenini ayarlayın.")

# 2. Vektör Veri Tabanını Yükleme
# Önceki adımda kaydettiğimiz Chroma DB'yi yüklüyoruz.
PERSIST_DIR = "./chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

print(f"Vektör veritabanı {PERSIST_DIR} konumundan yükleniyor...")
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

# 3. Retriever (Bilgi Çekici) Nesnesi
# Veritabanında en alakalı 3 dokümanı çekecek bir retriever oluşturuyoruz.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Retriever (Bilgi Çekici) hazırlandı.")

# 4. LLM (Gemini Modeli)
# Metin üretimi için Gemini Pro modelini kullanıyoruz.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1  # Yaratıcılık/Hallüsinasyon riskini azaltmak için düşük değer
)

# 5. Prompt (İstek Şablonu) Oluşturma
# RAG mimarisinin kalbi: LLM'e cevabı SADECE çekilen bağlama (context) göre vermesini söylüyoruz.
prompt_template = """
Sen tarımsal hastalıklar konusunda uzman bir yapay zeka asistanısın. Görevin,
kullanıcının sorusunu SADECE aşağıda verilen bağlam (context) bilgisine göre
detaylı ve bilgilendirici bir şekilde yanıtlamaktır. Eğer verilen bağlamda
sorunun cevabı YOKSA, "Üzgünüm, bu sorunun cevabı elimdeki bitki hastalıkları
bilgi tabanında bulunmamaktadır." diye cevap vermelisin.

--- BAĞLAM ---
{context}
---

Kullanıcının Sorusu: {question}
Cevap:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
print("Chat Prompt Şablonu hazırlandı.")

# 6. LangChain RAG Zincirini Oluşturma
# Zincir, (Retriever -> Prompt Hazırlama -> LLM) sırasını takip eder.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Test Sorgusu
def run_rag_query(query):
    """RAG zincirini çalıştırır ve cevabı yazdırır."""
    print(f"\n[Kullanıcı]: {query}")
    print("-----------------------------------------------------------------")
    # Zinciri çalıştır
    result = rag_chain.invoke(query)
    print(f"[Asistan]: {result}")
    return result

# --- ÇALIŞTIRMA ÖRNEKLERİ ---

# Veri setinde olması beklenen bir soru
run_rag_query("Domates bitkisinde oluşan bakteriyel leke hastalığının belirtileri ve tedavisi nedir?")

# Veri setinde muhtemelen olmayan bir genel bilgi sorusu (Cevap 'Üzgünüm' olmalı)
run_rag_query("Trakya bölgesinde buğday ekimi ne zaman yapılmalıdır?")
