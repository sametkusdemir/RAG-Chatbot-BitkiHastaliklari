import os
from google import genai
from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. API Anahtarının Kontrolü
# Projenizi çalıştırabilmek için GEMINI_API_KEY ortam değişkeni ayarlanmalıdır.
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("Lütfen GEMINI_API_KEY ortam değişkenini ayarlayın.")

# 2. Veri Setini Yükleme (Hugging Face)
# ipranavks/agriculturaldatasetnew veri seti yükleniyor.
# Sadece RAG için gerekli olan 'Crop', 'Disease' ve 'Description' sütunlarını alıyoruz.
print("Veri seti yükleniyor...")
dataset = load_dataset("ipranavks/agriculturaldatasetnew", split="train")
df = dataset.to_pandas()

# 3. Metinsel Veriyi Hazırlama ve Yükleme (LangChain Loader)
# Her bir satırı (bitki hastalığı) RAG dokümanı olarak hazırlıyoruz.
# Source/Metadata bilgisi, chatbot'un hangi bilgiyi nereden çektiğini görmesi için önemlidir.
df['content'] = df.apply(lambda row: f"Bitki: {row['Crop']}, Hastalık Adı: {row['Disease']}. Açıklama: {row['Description']}", axis=1)

# Veri setini LangChain'in anlayacağı 'Document' formatına dönüştürüyoruz.
loader = DataFrameLoader(df, page_content_column="content")
documents = loader.load()
print(f"Toplam yüklü doküman sayısı: {len(documents)}")

# 4. Metin Parçalama (Text Splitting / Chunking)
# RAG performansını artırmak için büyük metinleri küçük parçalara ayırıyoruz.
# Ayırıcı (splitter), en ideal parçalama için farklı karakterleri deneyecektir.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"Parçalama sonrası oluşan 'chunk' sayısı: {len(chunks)}")

# 5. Embedding Modelini Tanımlama
# Gemini'nin embedding modeli kullanılıyor.
print("Embedding modeli yükleniyor...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 6. Vektör Veri Tabanını Oluşturma ve Kaydetme (ChromaDB)
# Vektörleştirme yapılıyor ve ChromaDB'ye kaydediliyor.
# Kalıcı bir disk depolama alanı belirtiyoruz (persist_directory).
PERSIST_DIR = "./chroma_db"
print(f"Vektör veritabanı {PERSIST_DIR} konumuna oluşturuluyor ve kaydediliyor...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR  # Veritabanını kaydet
)

print("\n🎉 Veri tabanı hazırlığı tamamlandı! Artık RAG sorguları için hazır.")
