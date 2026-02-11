# Book Recommender System

A collaborative filtering-based book recommendation engine built with Python and Streamlit. Given a book title, the system suggests 5 similar books using a K-Nearest Neighbors model trained on user rating patterns.

## How It Works

1. **Data Ingestion** -- Downloads the dataset from Kaggle and extracts the raw CSV files.
2. **Data Validation** -- Cleans the data by filtering out inactive users (fewer than 200 ratings) and unpopular books (fewer than 50 ratings), then removes duplicates.
3. **Data Transformation** -- Builds a pivot table (books vs. users) from the cleaned ratings and fills missing values with zeros.
4. **Model Training** -- Converts the pivot table to a sparse matrix and fits a K-Nearest Neighbors model using brute-force distance computation.
5. **Recommendation** -- For a selected book, the trained KNN model finds the 5 most similar books based on rating patterns and displays them with cover images.

## Tech Stack

- **Python 3.13**
- **scikit-learn** -- KNN model (NearestNeighbors)
- **pandas / NumPy** -- Data manipulation
- **SciPy** -- Sparse matrix representation
- **Streamlit** -- Web interface
- **Docker** -- Containerized deployment

## Project Structure

```
book-recommendation-system/
├── app.py                          # Streamlit web app
├── main.py                         # Script to run the training pipeline
├── config/
│   └── config.yaml                 # All paths and settings
├── books_recommender/
│   ├── components/
│   │   ├── data_ingestion.py       # Download and extract dataset
│   │   ├── data_validation.py      # Clean and filter data
│   │   ├── data_transformation.py  # Build pivot table
│   │   └── model_trainer.py        # Train KNN model
│   ├── config/
│   │   └── configuration.py        # Reads config.yaml into typed objects
│   ├── entity/
│   │   └── config_entity.py        # Named tuple definitions for configs
│   ├── pipeline/
│   │   └── training_pipeline.py    # Orchestrates all components
│   ├── exception/
│   │   └── exception_handler.py    # Custom exception with file/line info
│   ├── logger/
│   │   └── log.py                  # Timestamped file logging
│   ├── utils/
│   │   └── util.py                 # YAML reader utility
│   └── constant/
│       └── __init__.py             # Project-level constants
├── notebooks/
│   └── research.ipynb              # Exploratory analysis
├── templates/
│   └── book_names.pkl              # Serialized book names for the dropdown
├── Dockerfile
├── requirements.txt
└── setup.py
```

## Setup and Installation

### Local

```bash
git clone https://github.com/HELLRAISER3/book-recommendation-system.git
cd book-recommendation-system
pip install -r requirements.txt
```

### Run the Training Pipeline

```bash
python main.py
```

This downloads the dataset, processes it, and trains the model. Artifacts are saved under `artifacts/`.

### Launch the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Select a book from the dropdown and click "Show Recommendation" to see 5 similar books.

### Docker

```bash
docker build -t book-recommender .
docker run -p 8501:8501 book-recommender
```

## Dataset

[Book Recommendation Dataset](https://www.kaggle.com/datasets/ra4u12/bookrecommendation) from Kaggle, containing book metadata and user ratings.
