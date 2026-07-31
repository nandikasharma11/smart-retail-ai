# Smart Retail & Customer Intelligence Platform

Welcome to the **Smart Retail & Customer Intelligence Platform**, an AI-powered end-to-end analytics and customer assistance solution designed to modernize retail environments. The platform integrates real-time computer vision, natural language processing, and an interactive dashboard to track visitor logs, customer sentiments, and FAQ inquiries.

---

## 🚀 Key Features

* **Interactive Analytics Dashboard**: A modern web interface displaying real-time metrics, visitor histories, customer sentiment distributions, and chatbot usage counters.
* **Computer Vision (CV) Services**:
  * **Face Recognition**: Detects and recognizes registered customers (e.g., matching with a local face database) using OpenCV and statistical face embeddings.
  * **Product Classification**: Automatically classifies products from images using a deep learning classifier built in PyTorch (trained on FashionMNIST-like categories).
* **Natural Language Processing (NLP) Services**:
  * **Sentiment Analysis**: Classifies customer feedback text into positive, neutral, or negative sentiments using a scikit-learn classification pipeline.
  * **Retrieval-Based FAQ Chatbot**: Provides answers to customer questions using text preprocessing (TF-IDF vectorizer) and a rule-fallback classification model.
* **Docker Support**: Containerized configuration for easy and consistent deployment across development, staging, and production environments.

---

## 📂 Project Directory Structure

```text
smart-retail-ai/
├── Dockerfile                  # Containerization setup for the platform
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── app/                        # Application source code
│   ├── main.py                 # FastAPI application entry point
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── models/                 # Pre-trained models and local database files
│   │   ├── chatbot_model.pkl   # Serialized chatbot model
│   │   ├── face_db.pkl         # Local face embeddings database
│   │   ├── product_classifier.pt # PyTorch product classification model
│   │   ├── sentiment_model.pkl # Trained sentiment classifier
│   │   └── vectorizer.pkl      # Saved TF-IDF vectorizer
│   ├── routers/                # FastAPI routing layers
│   │   ├── chatbot.py          # Chatbot interactive endpoint
│   │   ├── nlp.py              # NLP sentiment analysis endpoint
│   │   └── vision.py           # Face recognition & product classification endpoints
│   ├── services/               # Platform business logic services
│   │   ├── chatbot_service.py  # FAQ chatbot matching engine
│   │   ├── cv_service.py       # High-level vision processing service
│   │   ├── cv_utils.py         # OpenCV face detection and PyTorch helper utilities
│   │   ├── face_recognition_module.py # Local face recognition implementation
│   │   ├── nlp_service.py      # Core NLP and sentiment analysis service
│   │   └── pipeline.py         # Unified ML pipeline runner
│   └── static/                 # Frontend assets
│       └── index.html          # Interactive Dashboard UI
├── data/                       # Datasets, reference assets, and visitor logs
│   ├── FashionMNIST/           # Cache of FashionMNIST data (if training)
│   ├── customer_visits.csv     # Real-time log of customer visits
│   ├── intents.json            # Base intents for the FAQ chatbot
│   ├── lena.jpg                # Sample image for CV validation
│   └── reviews.csv             # Customer reviews dataset
├── notebooks/                  # Development and training notebooks
│   ├── 01_cv_utils_and_basics.ipynb
│   ├── 02_image_classifier_training.ipynb
│   ├── 03_face_recognition_setup.ipynb
│   ├── 04_text_preprocessing_and_sentiment.ipynb
│   ├── 05_chatbot_training.ipynb
│   └── 06_unified_pipeline_test.ipynb
└── tests/                      # Automated test suite
    └── test_endpoints.py       # Endpoint unit and integration tests
```

---

## 🛠️ Tech Stack

* **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
* **Computer Vision**: [OpenCV](https://opencv.org/), [PyTorch](https://pytorch.org/)
* **Natural Language Processing**: [Scikit-Learn](https://scikit-learn.org/), [NLTK](https://www.nltk.org/)
* **Frontend**: HTML5, Vanilla CSS, JavaScript
* **Database & Storage**: CSV logging, Pickled databases (`.pkl`), PyTorch weights (`.pt`)
* **Testing**: [pytest](https://docs.pytest.org/)

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10+
* pip

### 1. Clone the Repository
```bash
git clone https://github.com/nandikasharma11/smart-retail-ai.git
cd smart-retail-ai
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Application

### Running Locally
Start the FastAPI server using Uvicorn:
```bash
uvicorn app.main:app --reload
```
Once started, access:
* **Interactive Dashboard (Web UI)**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Running with Docker
Build and run the container:
```bash
docker build -t smart-retail-ai .
docker run -p 8000:8000 smart-retail-ai
```

---

## 🧪 Testing

To run the automated tests, execute `pytest` with the `PYTHONPATH` environment variable set:

```bash
PYTHONPATH=. pytest
```
