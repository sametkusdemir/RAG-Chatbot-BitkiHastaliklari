# RAG-Chatbot-BitkiHastaliklari
#  🌿 Bitki Hastalıkları Bilgi Sistemi: RAG Destekli Chatbot

## 🎯 Projenin Amacı

[cite_start]Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, Retrieval Augmented Generation (RAG) temelli bir bilgi sistemidir[cite: 2].

**Temel Amaç:** Kullanıcıların tarım ve bitki hastalıkları hakkındaki metin tabanlı sorularını doğru, bağlam ile zenginleştirilmiş ve hızlı bir şekilde yanıtlayan bir chatbot sunmaktır. Model, önceden hazırlanmış tarımsal veri setinden (bilgi tabanı) bilgi çekerek, Büyük Dil Modeli'nin (LLM) halüsinasyon yapma riskini en aza indirir ve güvenilir cevaplar sağlar.

## 💾 Veri Seti Hakkında Bilgi

Projenin bilgi tabanını (Knowledge Base) oluşturmak için bitki hastalıkları ve tarımsal konuları içeren bir metin veri seti kullanılmıştır.

* **Veri Seti Kaynağı:** [ipranavks/agriculturaldatasetnew](https://huggingface.co/datasets/ipranavks/agriculturaldatasetnew) (veya kullanılan diğer kaynaklar)
* **Veri Tipi:** Bitki türleri, hastalık adları, semptomlar, önleme ve tedavi yöntemleri gibi tarımsal konuları içeren yapılandırılmış **metin verisi**dir.
* **Hazırlık Metodolojisi:** Veri setindeki ham metinler, RAG mimarisine uygun hale getirilmek için parçalara ($chunks$) ayrılmış ve vektör temsilleri ($embeddings$) oluşturularak Vektör Veri Tabanı'na kaydedilmiştir.

## ⚙️ Kullanılan Yöntemler ve Çözüm Mimarisi

Bu projede, güvenilir ve bağlama dayalı yanıtlar üretmek için RAG mimarisi kullanılmıştır.

* [cite_start]**RAG Pipeline Framework:** LangChain / Haystack (Seçiminizi buraya yazın) 
* **Generation Model (LLM):** Gemini API / OpenAI API / vb. (Seçiminizi buraya yazın) [cite_start][cite: 42]
* **Embedding Model:** Google / Cohere / vb. (Seçiminizi buraya yazın) [cite_start][cite: 43]
* **Vektör Veri Tabanı:** Chroma / FAISS / Pinecone / vb. (Seçiminizi buraya yazın) [cite_start][cite: 43]
* **Web Arayüzü:** Streamlit / Gradio / Flask (Seçiminizi buraya yazın)

## 📈 Elde Edilen Sonuçlar (Özet)

* Geliştirilen RAG sistemi, %X doğruluk/uygunluk oranıyla (ileride bir metrikle doldurulacak) bitki hastalıkları hakkında bağlama uygun cevaplar üretebilmektedir.
* (Örnek bir başarı: "Sistem, özellikle 'domates yaprak lekesi' gibi spesifik hastalıkların tedavi yöntemleri hakkında doğru ve güncel bilgileri hızlıca çekebilmektedir.")

## 🔗 Uygulama Linki (Deployment)

Projenin çalışan web arayüzüne aşağıdaki linkten erişilebilir:

[cite_start]**Web Linki:** `[Deploy Linkiniz Buraya Gelecek - Adım 5 Sonrası Doldurulacak]` [cite: 13]

---
