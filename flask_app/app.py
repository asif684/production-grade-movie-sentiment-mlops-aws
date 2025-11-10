from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
import numpy as np
import warnings

# -------------------------------------------------------------------------------------
# Suppress warnings for clean logs
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------
# Text Preprocessing Functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in str(text).split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# -------------------------------------------------------------------------------------
# MLflow & DagsHub setup
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "mohammedasifameenbaig684")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if not DAGSHUB_TOKEN and not os.getenv("CI"):
    raise EnvironmentError(
        "❌ DAGSHUB_TOKEN not found. Please set it as an environment variable or GitHub Secret."
    )

os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/mohammedasifameenbaig684/production-grade-movie-sentiment-mlops-aws.mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
if DAGSHUB_TOKEN:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Skip OAuth if CI or token missing
if not os.getenv("CI"):
    try:
        dagshub.init(
            repo_owner="mohammedasifameenbaig684",
            repo_name="production-grade-movie-sentiment-mlops",
            mlflow=True,
            oauth=False
        )
        print("✅ DagsHub initialized successfully.")
    except Exception as e:
        print(f"⚠️ DagsHub initialization skipped due to error: {e}")
else:
    print("Skipping DagsHub initialization (CI environment detected).")

# -------------------------------------------------------------------------------------
# Flask app setup
app = Flask(__name__)

# Prometheus metrics
registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    "app_request_count", "Total requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions per class", ["prediction"], registry=registry
)

# -------------------------------------------------------------------------------------
# Load model and vectorizer
model_name = "my_model"

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if latest_versions:
        return latest_versions[0].version

    versions = client.search_model_versions(f"name='{model_name}'")
    if versions:
        latest = max(int(v.version) for v in versions)
        return str(latest)

    raise ValueError(f"No registered versions found for model '{model_name}'")

try:
    if os.getenv("CI"):  # CI-safe dummy model
        print("🧪 CI detected — using DummyModel for tests.")
        class DummyModel:
            def predict(self, X):
                return ["Positive"]
        model = DummyModel()
    else:
        model_version = get_latest_model_version(model_name)
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"✅ Loading MLflow model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"❌ Error loading MLflow model: {e}")
    model = None

# Load vectorizer safely
vectorizer_path = "models/vectorizer.pkl"
if os.path.exists(vectorizer_path):
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
else:
    print("⚠️ Vectorizer file not found — using CountVectorizer fallback.")
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()

# -------------------------------------------------------------------------------------
# Flask Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form.get("text", "")
    text = normalize_text(text)
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    if model is None:
        prediction = "Model not loaded"
    else:
        try:
            result = model.predict(features_df)
            prediction = result[0]
        except Exception as e:
            prediction = f"Prediction error: {e}"

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics"""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
