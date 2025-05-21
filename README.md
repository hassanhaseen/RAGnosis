# 🩺 RAGnosis – Clinical Q&A over MIMIC-IV with Google AI

<img src="https://i.ibb.co/kV53YCm0/Untitled-design.png" alt="RAGnosis Logo" width=150 align="right" />

Welcome to **RAGnosis** — a Retrieval-Augmented Generation (RAG) based medical assistant built on top of the **MIMIC-IV clinical dataset**, enhanced with **Google Gemini** and **FAISS-powered vector search**. Designed for clinical research and educational purposes, this app answers complex medical queries using real-world ICU records 🧠📊

---

## ✨ Features

✅ Ask natural language **clinical questions**  
✅ RAG pipeline powered by **Google Gemini + FAISS**  
✅ Semantic search using **GoogleGenerativeAIEmbeddings**  
✅ Automatic document chunking with synonym expansion  
✅ Beautiful dark-themed **Streamlit UI** with progress feedback  
✅ Fully deployable on **Streamlit Cloud** (API-key secure)  
✅ Designed for medical researchers & AI enthusiasts

---

## 🚀 Live Demo

🔗 [Launch RAGnosis on Streamlit](https://ragnosis-direct.streamlit.app/)

---

## 📖 Medium Blog

🔗 [Read How I Built RAGnosis](https://medium.com/@hassanhaseen/ragnosis-ai-powered-clinical-query-system-using-rag-3fae8ff00000)

---

## 📸 Overall Preview

| RAGnosis - Streamlit UI |
|-------------------------|
| ![App Preview](https://i.ibb.co/m56ZfXMD/image.png)|

## Working

| RAGnosis - Streamlit Working |
|-------------------------|
| ![App Working](https://i.ibb.co/v49rBtV2/image.png)|

---

## 🧠 How It Works

RAGnosis uses a hybrid system that:
1. **Embeds medical records** (from MIMIC-IV `Finished/` JSONs) using Google’s `text-embedding-004` model.
2. Stores them in a local **FAISS** vector index.
3. At query time, retrieves top-k relevant chunks and passes them to **Google Gemini Flash (gemini-2.0-flash)** for final answer generation.

📁 **Dataset**  
Unzipped MIMIC-IV-Extension JSONs placed inside:  
```bash
./mimic-iv-ext-direct-1.0.0/Finished/
```

📚 **Preprocessing**  
Includes regex-based medical abbreviation expansion (`HTN` → `hypertension`, `CAD` → `coronary artery disease`, etc.)

---

## 🖼️ App Highlights

- 💬 Natural-language Q&A over real medical data  
- 📎 Source docs displayed in expandable panels  
- 🔄 Cached data loading, chunking, vector storage  
- 🧩 Dynamic vectorstore naming based on embedding model  
- 🧠 Built-in synonym expansion to improve relevance  
- 🧪 Styled error handling and sidebar insights

---

## ⚙️ Tech Stack

- 🧠 LangChain (vector retrieval + RAG chaining)  
- 🏥 MIMIC-IV Clinical Dataset  
- 🔍 FAISS (local vector search)  
- ✨ GoogleGenerativeAI (Gemini & Embeddings)  
- 🎨 Streamlit (frontend)  
- 🧪 Python + dotenv + regex + JSON

---

## 👨‍⚕️ Ethical Note

> This project is for **educational & research purposes only**.  
> Do not use the system for real medical diagnosis or decision-making.

---

## 🤝 Built By

**Team CodeRunners**  
- [Hassan Haseen](https://github.com/hassanhaseen)
- [Sameen Muzaffar](https://github.com/SameenRajpoot) 
> Always open to collaboration or feedback 🧠✨

---

## 📝 License

This project is licensed under the [MIT License](LICENSE)

---

## ⭐ Give It a Star!

If you found **RAGnosis** insightful, please ⭐ the repo — your support helps us build more AI+Health innovations! 🩺⚡

---

## 🧠 TL;DR  
- RAG pipeline over MIMIC-IV using Google AI + LangChain  
- Streamlit UI with chunking, embeddings, and Gemini  
- Secure API key handling via Streamlit Secrets  
- Powered by **real medical data** & **generative retrieval**
