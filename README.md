# 🧠 Fantasy Football AI Platform (FastAPI + ML + LangChain)

> **End-to-end AI system** combining a production-ready FastAPI backend, ONNX-based machine learning models, and a LangChain-powered toolkit for intelligent data interaction — inspired by the *“Hands-On APIs for AI and Data Science”* book by Ryan Day (O’Reilly, 2025).

---

## 🚀 Overview

This project is a **modular AI application** designed to demonstrate modern **MLOps**, **API engineering**, and **AI integration** practices using Python.  
It provides a full-stack implementation for managing **fantasy football data**, training and serving ML models for **player acquisition prediction**, and exposing functionality through a **FastAPI** interface and **LangChain** toolkit.

The project demonstrates:
- **API-first architecture** for data science applications  
- **Model lifecycle management**: training → conversion to ONNX → deployment  
- **Database-driven API** using SQLAlchemy + SQLite  
- **LangChain integration** for intelligent querying and reasoning over the API  
- **Containerization & reproducibility** via requirements files and modular code design  

---

## 🏗️ Project Structure

```
├── LICENSE
├── README.md                ← You are here
├── api/                     ← FastAPI backend (CRUD + models + routes)
│   ├── main.py              ← API entrypoint
│   ├── models.py            ← SQLAlchemy ORM models
│   ├── schemas.py           ← Pydantic schemas for request/response validation
│   ├── database.py          ← SQLite database connection
│   ├── crud.py              ← CRUD operations for fantasy player data
│   ├── requirements.txt     ← API dependencies
│   ├── test_main.py         ← API endpoint tests
│   └── test_crud.py         ← CRUD layer unit tests
│
├── model-training/          ← ML model training and evaluation
│   ├── player_acquisition_model.ipynb ← Jupyter notebook for model development
│   ├── acquisition_model_10.onnx      ← Exported ONNX model (v1)
│   ├── acquisition_model_50.onnx      ← Exported ONNX model (v2)
│   ├── acquisition_model_90.onnx      ← Exported ONNX model (v3)
│   ├── main.py              ← Inference pipeline (FastAPI model endpoint)
│   ├── player_training_data_full.csv  ← Dataset for model training
│   └── schemas.py           ← Data validation schema for model input/output
│
├── langchain/               ← LangChain agent and graph reasoning toolkit
│   ├── swc_toolkit.py       ← Custom LangChain tool for API access
│   ├── langgraph_notebook.ipynb       ← LLM integration notebook
│   ├── langgraph_notebook_with_toolkit.ipynb ← Extended demo with toolkit
│   ├── requirements.txt     ← LangChain dependencies
│   └── README.md            ← Submodule documentation
└──
```

---

## ⚙️ Features

✅ **FastAPI Backend**
- Full CRUD API for managing fantasy football players and acquisitions  
- OpenAPI documentation (`/docs`) auto-generated  
- Integrated validation with Pydantic  
- SQLAlchemy ORM and SQLite database  

✅ **Machine Learning Module**
- Trains multiple models for player acquisition prediction  
- Models exported to **ONNX** for high-performance inference  
- Inference served via FastAPI and Docker-ready  

✅ **LangChain Toolkit**
- Connects LLMs (e.g., GPT-4 or Claude) to interact with the API  
- Provides intelligent querying and analytics over sports data  
- Uses **LangGraph** for structured reasoning and chain management  

✅ **Testing & CI**
- Unit tests for CRUD and API layers  
- Logs for training and inference  
- Structured modular design for extensibility  

---

## 🧩 Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Language** | Python 3.12 |
| **Web Framework** | FastAPI |
| **Database** | SQLite + SQLAlchemy |
| **ML Frameworks** | scikit-learn, ONNX, NumPy, pandas |
| **MLOps Tools** | Docker (optional), GitHub Actions (for CI/CD) |
| **AI Integration** | LangChain, LangGraph |
| **Testing** | pytest, unittest |
| **Documentation** | OpenAPI / Swagger UI |

---

## 🧪 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/phroo02/Fantasy-Football-AI-Platform.git
cd Fantasy-Football-AI-Platform
```

### 2️⃣ Set Up the Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
```

Install dependencies:

```bash
pip install -r api/requirements.txt
pip install -r model-training/requirments.txt
pip install -r langchain/requirements.txt
```

### 3️⃣ Run the FastAPI Server

```bash
cd api
uvicorn main:app --reload
```

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.

### 4️⃣ Run the ML Model (Standalone)

```bash
cd model-training
python main.py
```

### 5️⃣ Explore LangChain Integration (Optional)

Open `langchain/langgraph_notebook_with_toolkit.ipynb` in Jupyter to interact with the API using LangChain tools.

---

## 📊 Example Use Cases

- Predict which players will be acquired based on historical stats and team performance  
- Serve predictions in real time via FastAPI endpoints  
- Query the API with natural language using a LangChain agent  
- Extend the system with additional ML models (e.g., injury prediction, fantasy ranking)

---

## 🧠 Learning Outcomes

This project demonstrates:
- Full **data→model→API→LLM** workflow  
- **MLOps principles**: modularity, testing, containerization, reproducibility  
- Integration of **AI APIs and generative tools** in production  
- API documentation and deployment best practices  

It aligns with the final portfolio project of the book  
> “**Hands-On APIs for AI and Data Science: Python Development with FastAPI**” – Ryan Day, O’Reilly (2025)

---

## 📈 Future Improvements
- [ ] Add Docker Compose for API + LangChain orchestration  
- [ ] Integrate model version tracking via MLflow  
- [ ] Migrate to PostgreSQL for scalability  
- [ ] Add unit tests for LangChain tool interactions  
- [ ] Deploy on AWS Lightsail or Azure App Service  

---

## 📜 License
This project is licensed under the terms of the **MIT License**.  
See [LICENSE](./LICENSE) for details.

---

> *“APIs are the modern fabric of AI systems. Build them smart, scalable, and human-friendly.”*  
> — Ryan Day, *Hands-On APIs for AI and Data Science (O’Reilly, 2025)*
