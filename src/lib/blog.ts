export interface BlogPost {
  id: string;
  title: string;
  description: string;
  content: string;
  date: string;
  readTime: string;
  category: string;
  image: string;
  featured: boolean;
  author: {
    name: string;
    avatar: string;
    bio: string;
  };
  tags: string[];
  seo: {
    metaTitle?: string;
    metaDescription?: string;
    keywords?: string[];
  };
}

export const blogPosts: BlogPost[] = [
  {
    id: "python-data-analysis-best-practices",
    title: "Python Data Analysis: Best Practices for Clean Code",
    description:
      "Explore essential techniques for writing maintainable and efficient Python code for data analysis projects.",
    date: "2024-01-15",
    readTime: "8 min read",
    category: "Python",
    image:
      "https://images.unsplash.com/photo-1526379095098-d400fd0bf935?w=1200&q=80",
    featured: true,
    author: {
      name: "Alex Johnson",
      avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=alex",
      bio: "Senior Data Scientist with 8+ years of experience in Python development and machine learning.",
    },
    tags: ["Python", "Data Analysis", "Best Practices", "Clean Code"],
    seo: {
      metaTitle: "Python Data Analysis Best Practices - Clean Code Guide",
      metaDescription:
        "Learn essential techniques for writing maintainable Python code for data analysis. Improve code quality and efficiency.",
      keywords: [
        "python",
        "data analysis",
        "clean code",
        "best practices",
        "pandas",
      ],
    },
    content: `
# Python Data Analysis: Best Practices for Clean Code

Writing clean, maintainable code is crucial for any data analysis project. In this post, we'll explore essential techniques that will make your Python data analysis code more readable, efficient, and professional.

## 1. Structure Your Project Properly

A well-organized project structure is the foundation of clean code:

\`\`\`
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── tests/
├── requirements.txt
└── README.md
\`\`\`

## 2. Use Meaningful Variable Names

Instead of:
\`\`\`python
df = pd.read_csv('data.csv')
x = df['column1']
y = df['column2']
\`\`\`

Use:
\`\`\`python
sales_data = pd.read_csv('sales_data.csv')
revenue = sales_data['monthly_revenue']
customer_count = sales_data['customer_count']
\`\`\`

## 3. Write Modular Functions

Break down complex operations into smaller, reusable functions:

\`\`\`python
def clean_sales_data(df):
    \"\"\"Clean and preprocess sales data.\"\"\"
    df = df.dropna()
    df['date'] = pd.to_datetime(df['date'])
    df['revenue'] = df['revenue'].astype(float)
    return df

def calculate_monthly_metrics(df):
    \"\"\"Calculate monthly sales metrics.\"\"\"
    monthly_data = df.groupby(df['date'].dt.to_period('M')).agg({
        'revenue': 'sum',
        'customer_count': 'nunique'
    })
    return monthly_data
\`\`\`

## 4. Document Your Code

Use docstrings and comments effectively:

\`\`\`python
def analyze_customer_segments(df, segment_column='customer_type'):
    \"\"\"
    Analyze customer segments and their performance metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Customer data with transaction information
    segment_column : str, default 'customer_type'
        Column name containing customer segment information
    
    Returns:
    --------
    pandas.DataFrame
        Aggregated metrics by customer segment
    \"\"\"
    # Group by customer segment and calculate key metrics
    segment_analysis = df.groupby(segment_column).agg({
        'revenue': ['sum', 'mean'],
        'transaction_count': 'sum',
        'customer_id': 'nunique'
    })
    
    return segment_analysis
\`\`\`

## 5. Handle Errors Gracefully

\`\`\`python
def load_and_validate_data(file_path, required_columns):
    \"\"\"Load data and validate required columns exist.\"\"\"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f\"Data file not found: {file_path}\")
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f\"Missing required columns: {missing_columns}\")
    
    return df
\`\`\`

## 6. Use Configuration Files

Store configuration in separate files:

\`\`\`python
# config.py
DATA_PATH = 'data/raw/'
OUTPUT_PATH = 'data/processed/'
REQUIRED_COLUMNS = ['date', 'customer_id', 'revenue']
DATE_FORMAT = '%Y-%m-%d'
\`\`\`

## Conclusion

Following these best practices will make your data analysis code more maintainable, readable, and professional. Remember that clean code is not just about following rules—it's about making your work accessible to others (including your future self).

Start implementing these practices in your next project, and you'll see immediate improvements in code quality and development efficiency.
    `,
  },
  {
    id: "pandas-performance-optimization",
    title: "Optimizing Pandas Performance for Large Datasets",
    description:
      "Learn advanced techniques to speed up your pandas operations and handle large datasets efficiently.",
    date: "2024-01-10",
    readTime: "12 min read",
    category: "Data Analysis",
    image:
      "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&q=80",
    featured: false,
    author: {
      name: "Sarah Chen",
      avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=sarah",
      bio: "Data Engineer specializing in big data processing and performance optimization.",
    },
    tags: ["Pandas", "Performance", "Optimization", "Big Data"],
    seo: {
      metaTitle: "Pandas Performance Optimization for Large Datasets",
      metaDescription:
        "Master advanced pandas techniques to handle large datasets efficiently. Learn memory optimization and performance tips.",
      keywords: [
        "pandas",
        "performance",
        "optimization",
        "large datasets",
        "memory",
      ],
    },
    content: `
# Optimizing Pandas Performance for Large Datasets

Working with large datasets in pandas can be challenging. This comprehensive guide covers advanced techniques to optimize performance and memory usage when dealing with big data.

## Understanding Memory Usage

Before optimizing, it's crucial to understand how pandas uses memory:

\`\`\`python
import pandas as pd
import numpy as np

# Check memory usage
df = pd.read_csv('large_dataset.csv')
print(df.info(memory_usage='deep'))
print(f\"Total memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\")

# Memory usage by column
for col in df.columns:
    print(f\"{col}: {df[col].memory_usage(deep=True) / 1024**2:.2f} MB\")
\`\`\`

## 1. Data Type Optimization

### Choose Appropriate Data Types

\`\`\`python
# Before optimization
df = pd.read_csv('large_dataset.csv')
print(\"Before optimization:\")
print(df.dtypes)
print(f\"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\")

# Optimize data types
optimizations = {
    'category_col': 'category',
    'small_int_col': 'int8',
    'medium_int_col': 'int16',
    'boolean_col': 'bool',
    'float_col': 'float32'
}

for col, dtype in optimizations.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)

print(\"\\nAfter optimization:\")
print(df.dtypes)
print(f\"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\")
\`\`\`

### Automatic Type Inference

\`\`\`python
def optimize_dtypes(df):
    \"\"\"Automatically optimize DataFrame dtypes.\"\"\"
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        
        else:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
    
    return df
\`\`\`

## 2. Chunking for Large Files

\`\`\`python
def process_large_csv(file_path, chunk_size=10000, operation='sum'):
    \"\"\"Process large CSV files in chunks.\"\"\"
    results = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Optimize chunk dtypes
        chunk = optimize_dtypes(chunk)
        
        # Process each chunk
        if operation == 'sum':
            processed_chunk = chunk.groupby('category')['value'].sum()
        elif operation == 'mean':
            processed_chunk = chunk.groupby('category')['value'].mean()
        
        results.append(processed_chunk)
    
    # Combine results
    if operation == 'sum':
        final_result = pd.concat(results).groupby(level=0).sum()
    elif operation == 'mean':
        final_result = pd.concat(results).groupby(level=0).mean()
    
    return final_result

# Usage
result = process_large_csv('huge_dataset.csv', chunk_size=50000)
\`\`\`

## 3. Vectorized Operations

### Avoid Loops

\`\`\`python
# Slow: Using loops
result = []
for index, row in df.iterrows():
    result.append(row['value'] * 2 + row['bonus'])

# Fast: Vectorized operation
result = df['value'] * 2 + df['bonus']

# Even better: Using eval for complex expressions
result = df.eval('value * 2 + bonus')
\`\`\`

### Use Built-in Methods

\`\`\`python
# Slow
df['is_weekend'] = df['date'].apply(lambda x: x.weekday() >= 5)

# Fast
df['is_weekend'] = df['date'].dt.weekday >= 5
\`\`\`

## 4. Advanced Performance Techniques

### Using Categorical Data

\`\`\`python
# Convert string columns to categorical
df['status'] = df['status'].astype('category')
df['region'] = df['region'].astype('category')

# This can reduce memory usage by 50-90% for string columns
\`\`\`

### Parallel Processing with Dask

\`\`\`python
import dask.dataframe as dd

# Read large CSV with Dask
ddf = dd.read_csv('huge_dataset.csv')

# Perform operations
result = ddf.groupby('category')['value'].sum().compute()
\`\`\`

## 5. Memory Management

### Monitor Memory Usage

\`\`\`python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f\"Memory usage: {get_memory_usage():.2f} MB\")
\`\`\`

### Clean Up Unused DataFrames

\`\`\`python
# Delete unused dataframes
del df_temp

# Force garbage collection
import gc
gc.collect()
\`\`\`

## Conclusion

Optimizing pandas performance requires a combination of techniques:

1. **Memory optimization** through appropriate data types
2. **Chunking** for datasets larger than memory
3. **Vectorized operations** instead of loops
4. **Efficient indexing** and filtering
5. **Parallel processing** for CPU-intensive tasks
6. **Out-of-core processing** with tools like Dask

Implement these techniques progressively, measuring performance improvements at each step. Remember that the best optimization strategy depends on your specific use case and data characteristics.

### Performance Benchmarking

Always measure your optimizations:

\`\`\`python
import time

start_time = time.time()
# Your pandas operation here
end_time = time.time()

print(f\"Operation took {end_time - start_time:.2f} seconds\")
\`\`\`

With these techniques, you can handle datasets 10-100x larger than your available RAM efficiently.
    `,
  },
  {
    id: "machine-learning-model-deployment",
    title: "From Jupyter to Production: ML Model Deployment",
    description:
      "A comprehensive guide to deploying machine learning models from development to production environments.",
    date: "2024-01-05",
    readTime: "15 min read",
    category: "Machine Learning",
    image:
      "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1200&q=80",
    featured: false,
    author: {
      name: "Michael Rodriguez",
      avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=michael",
      bio: "MLOps Engineer with expertise in model deployment and production systems.",
    },
    tags: ["Machine Learning", "Deployment", "MLOps", "Production"],
    seo: {
      metaTitle: "ML Model Deployment Guide - From Jupyter to Production",
      metaDescription:
        "Complete guide to deploying machine learning models in production. Learn MLOps best practices and deployment strategies.",
      keywords: [
        "machine learning",
        "model deployment",
        "mlops",
        "production",
        "jupyter",
      ],
    },
    content: `
# From Jupyter to Production: ML Model Deployment

Deploying machine learning models from development to production is one of the most critical challenges in the ML lifecycle. This comprehensive guide covers the entire journey from Jupyter notebooks to scalable production systems.

## The ML Deployment Challenge

Moving from a Jupyter notebook to production involves several key challenges:

- **Environment consistency** across development and production
- **Model versioning** and reproducibility
- **Scalability** and performance requirements
- **Monitoring** and maintenance
- **Security** and compliance
- **Data drift** and model degradation

## 1. Preparing Your Model for Deployment

### Model Serialization

\`\`\`python
import joblib
import pickle
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Train a sample model with preprocessing
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a pipeline with preprocessing
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Save the complete pipeline
joblib.dump(pipeline, 'model_pipeline.pkl')

# Alternative: using MLflow for model tracking
with mlflow.start_run():
    mlflow.sklearn.log_model(pipeline, \"model\")
    mlflow.log_param(\"n_estimators\", 100)
    mlflow.log_metric(\"accuracy\", pipeline.score(X_test, y_test))
\`\`\`

### Model Validation Pipeline

\`\`\`python
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class ModelValidator:
    def __init__(self, model_path, test_data_path):
        self.model = joblib.load(model_path)
        self.test_data = pd.read_csv(test_data_path)
    
    def validate_model(self):
        \"\"\"Validate model performance on test data.\"\"\"
        X_test = self.test_data.drop('target', axis=1)
        y_test = self.test_data['target']
        
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        
        print(f\"Model Accuracy: {accuracy:.4f}\")
        print(\"\\nClassification Report:\")
        print(classification_report(y_test, predictions))
        print(\"\\nConfusion Matrix:\")
        print(confusion_matrix(y_test, predictions))
        
        # Additional validation checks
        self._check_prediction_distribution(predictions)
        self._check_probability_calibration(probabilities, y_test)
        
        return accuracy > 0.8  # Minimum acceptable accuracy
    
    def _check_prediction_distribution(self, predictions):
        \"\"\"Check if prediction distribution is reasonable.\"\"\"
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique, counts / len(predictions)))
        print(f\"\\nPrediction Distribution: {distribution}\")
    
    def _check_probability_calibration(self, probabilities, y_true):
        \"\"\"Check probability calibration.\"\"\"
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, probabilities[:, 1], n_bins=10
        )
        
        print(\"\\nCalibration check completed\")
    
    def validate_input_schema(self, input_data):
        \"\"\"Validate input data schema.\"\"\"
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = self.model.feature_names_in_
        else:
            # For pipelines, get from the last step
            expected_features = self.model.steps[-1][1].feature_names_in_
        
        input_features = input_data.columns.tolist()
        
        missing_features = set(expected_features) - set(input_features)
        extra_features = set(input_features) - set(expected_features)
        
        if missing_features:
            raise ValueError(f\"Missing features: {missing_features}\")
        
        if extra_features:
            print(f\"Warning: Extra features will be ignored: {extra_features}\")
        
        return True
\`\`\`

## 2. Creating a Model API

### FastAPI Implementation (Recommended)

\`\`\`python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=\"ML Model API\", version=\"1.0.0\")

# Load model at startup
try:
    model = joblib.load('model_pipeline.pkl')
    logger.info(\"Model loaded successfully\")
except Exception as e:
    logger.error(f\"Failed to load model: {e}\")
    model = None

class PredictionRequest(BaseModel):
    features: List[List[float]]
    
class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]
    timestamp: str
    model_version: str

class ModelAPI:
    def __init__(self, model):
        self.model = model
        self.prediction_count = 0
        self.start_time = datetime.now()
    
    def predict(self, features: List[List[float]]) -> Dict[str, Any]:
        \"\"\"Make predictions.\"\"\"
        try:
            # Convert to DataFrame
            df = pd.DataFrame(features)
            
            # Make predictions
            predictions = self.model.predict(df)
            probabilities = self.model.predict_proba(df)
            
            self.prediction_count += 1
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'timestamp': datetime.now().isoformat(),
                'model_version': '1.0.0'
            }
        
        except Exception as e:
            logger.error(f\"Prediction error: {e}\")
            raise HTTPException(status_code=400, detail=str(e))

api = ModelAPI(model) if model else None

@app.post(\"/predict\", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not api:
        raise HTTPException(status_code=503, detail=\"Model not available\")
    
    result = api.predict(request.features)
    return PredictionResponse(**result)

@app.get(\"/health\")
async def health():
    return {
        'status': 'healthy' if model else 'unhealthy',
        'model_loaded': model is not None,
        'predictions_made': api.prediction_count if api else 0,
        'uptime_seconds': (datetime.now() - api.start_time).total_seconds() if api else 0
    }

@app.get(\"/metrics\")
async def metrics():
    if not api:
        return {'error': 'Model not available'}
    
    return {
        'total_predictions': api.prediction_count,
        'uptime_seconds': (datetime.now() - api.start_time).total_seconds(),
        'model_version': '1.0.0'
    }
\`\`\`

## 3. Containerization with Docker

### Dockerfile

\`\`\`dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]
\`\`\`

## 4. Monitoring and Logging

### Model Performance Monitoring

\`\`\`python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Metrics
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')

class MonitoredModelAPI(ModelAPI):
    def predict(self, features):
        with PREDICTION_LATENCY.time():
            result = super().predict(features)
            PREDICTION_COUNT.inc()
            return result
    
    def update_accuracy(self, accuracy):
        MODEL_ACCURACY.set(accuracy)
\`\`\`

## 5. CI/CD Pipeline

### GitHub Actions Workflow

\`\`\`yaml
name: ML Model Deployment

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: pytest tests/
    - name: Validate model
      run: python validate_model.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v2
    - name: Build and push Docker image
      run: |
        docker build -t ml-model:latest .
        docker push your-registry/ml-model:latest
    - name: Deploy to production
      run: |
        # Your deployment commands here
\`\`\`

## Conclusion

Successful ML model deployment requires careful planning and implementation of several key components:

1. **Model preparation** and validation with comprehensive testing
2. **API development** with proper error handling and monitoring
3. **Containerization** for consistent environments
4. **Monitoring and logging** for production insights
5. **CI/CD pipelines** for automated deployment
6. **Scaling and versioning** strategies
7. **Security** considerations and access control
8. **Performance monitoring** and alerting

### Best Practices Summary

- **Version everything**: Code, data, models, and configurations
- **Test thoroughly**: Unit tests, integration tests, and model validation
- **Monitor continuously**: Performance, accuracy, and system health
- **Plan for failure**: Graceful degradation and rollback strategies
- **Document extensively**: APIs, deployment procedures, and troubleshooting guides

Start with a simple deployment and gradually add complexity as your requirements grow. Remember that deployment is not a one-time activity—it's an ongoing process that requires continuous monitoring and improvement.

### Next Steps

1. Implement A/B testing for model comparison
2. Set up automated retraining pipelines
3. Add feature stores for consistent data access
4. Implement model explainability tools
5. Set up comprehensive alerting and incident response

With these practices in place, you'll have a robust, scalable ML deployment that can handle production workloads reliably.
    `,
  },
  {
    id: "data-visualization-storytelling",
    title: "Data Visualization: Telling Stories with Numbers",
    description:
      "Master the art of creating compelling data visualizations that communicate insights effectively.",
    date: "2023-12-28",
    readTime: "10 min read",
    category: "Data Visualization",
    image:
      "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&q=80",
    featured: false,
    author: {
      name: "Emma Thompson",
      avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=emma",
      bio: "Data Visualization Specialist and UX Designer with a passion for storytelling through data.",
    },
    tags: ["Data Visualization", "Storytelling", "Charts", "Design"],
    seo: {
      metaTitle: "Data Visualization Storytelling - Effective Chart Design",
      metaDescription:
        "Learn to create compelling data visualizations that tell stories. Master chart design principles and storytelling techniques.",
      keywords: [
        "data visualization",
        "storytelling",
        "charts",
        "design",
        "matplotlib",
        "seaborn",
      ],
    },
    content: `
# Data Visualization: Telling Stories with Numbers

Data visualization is more than just creating charts—it's about telling compelling stories that drive decision-making. This guide explores the principles and techniques for creating visualizations that communicate insights effectively and engage your audience.

## The Power of Visual Storytelling

Humans process visual information 60,000 times faster than text. A well-designed visualization can:

- **Reveal patterns** hidden in raw data
- **Simplify complex** information
- **Engage audiences** emotionally
- **Drive action** through clear insights
- **Make data memorable** and impactful
- **Bridge communication gaps** between technical and non-technical stakeholders

## 1. Choosing the Right Chart Type

### The Chart Selection Framework

\`\`\`python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Chart type decision matrix
chart_guide = {
    'comparison': ['bar', 'column', 'radar'],
    'composition': ['pie', 'donut', 'stacked_bar', 'treemap'],
    'distribution': ['histogram', 'box', 'violin', 'density'],
    'relationship': ['scatter', 'bubble', 'heatmap', 'correlation'],
    'trend': ['line', 'area', 'slope'],
    'geographic': ['choropleth', 'bubble_map', 'flow_map']
}
\`\`\`

### Comparison Charts

\`\`\`python
# Sample data
data = {
    'Product': ['A', 'B', 'C', 'D', 'E'],
    'Sales': [23000, 45000, 56000, 78000, 32000],
    'Profit': [12000, 19000, 24000, 35000, 15000],
    'Market_Share': [15, 28, 35, 48, 20]
}
df = pd.DataFrame(data)

# Enhanced bar chart with multiple metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Sales comparison
sns.barplot(data=df, x='Product', y='Sales', palette='viridis', ax=ax1)
ax1.set_title('Sales by Product', fontsize=16, fontweight='bold')
ax1.set_ylabel('Sales ($)', fontsize=12)

# Add value labels on bars
for i, v in enumerate(df['Sales']):
    ax1.text(i, v + 1000, f'\${v:,}', ha='center', va='bottom', fontweight='bold')

# Profit margin analysis
df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100
sns.barplot(data=df, x='Product', y='Profit_Margin', palette='RdYlGn', ax=ax2)
ax2.set_title('Profit Margin by Product', fontsize=16, fontweight='bold')
ax2.set_ylabel('Profit Margin (%)', fontsize=12)

# Add percentage labels
for i, v in enumerate(df['Profit_Margin']):
    ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
\`\`\`

### Interactive Visualizations with Plotly

\`\`\`python
# Interactive dashboard-style visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Sales Performance', 'Market Share', 'Profit Analysis', 'Growth Trends'),
    specs=[[{\"secondary_y\": True}, {\"type\": \"pie\"}],
           [{\"colspan\": 2}, None]]
)

# Sales and profit bars
fig.add_trace(
    go.Bar(x=df['Product'], y=df['Sales'], name='Sales', marker_color='lightblue'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df['Product'], y=df['Profit'], mode='lines+markers', 
               name='Profit', line=dict(color='red', width=3)),
    row=1, col=1, secondary_y=True
)

# Market share pie chart
fig.add_trace(
    go.Pie(labels=df['Product'], values=df['Market_Share'], name='Market Share'),
    row=1, col=2
)

# Combined trend analysis
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
for product in df['Product']:
    trend_data = np.random.normal(df[df['Product']==product]['Sales'].iloc[0], 5000, 6)
    fig.add_trace(
        go.Scatter(x=months, y=trend_data, mode='lines+markers', name=f'{product} Trend'),
        row=2, col=1
    )

fig.update_layout(height=800, showlegend=True, title_text=\"Sales Dashboard\")
fig.show()
\`\`\`

## 2. Design Principles

### Color Psychology and Accessibility

\`\`\`python
# Colorblind-friendly palette
colorblind_palette = {
    'primary': '#1f77b4',      # Blue - trust, stability
    'success': '#2ca02c',      # Green - growth, positive
    'warning': '#ff7f0e',      # Orange - attention, caution
    'danger': '#d62728',       # Red - urgency, negative
    'neutral': '#7f7f7f',      # Gray - neutral information
    'accent': '#9467bd'        # Purple - creativity, premium
}

# Create an accessible color-coded chart
categories = ['Excellent', 'Good', 'Average', 'Poor', 'Critical']
values = [25, 35, 20, 15, 5]
color_map = [colorblind_palette['success'], colorblind_palette['primary'], 
            colorblind_palette['neutral'], colorblind_palette['warning'], 
            colorblind_palette['danger']]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(categories, values, color=color_map, edgecolor='black', linewidth=1.2)

# Enhanced styling
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
            f'{value}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add pattern for accessibility
    if value < 20:
        bar.set_hatch('///')

ax.set_title('Customer Satisfaction Ratings', fontsize=18, fontweight='bold', pad=20)
ax.set_ylabel('Percentage (%)', fontsize=14)
ax.set_ylim(0, max(values) * 1.2)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
\`\`\`

## 3. Advanced Visualization Techniques

### Storytelling with Annotations

\`\`\`python
# Time series with story annotations
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=365, freq='D')
base_trend = np.linspace(100, 150, 365)
seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)
noise = np.random.normal(0, 5, 365)
values = base_trend + seasonal + noise

# Add some events
event_dates = ['2023-03-15', '2023-07-04', '2023-11-24']
event_labels = ['Product Launch', 'Summer Campaign', 'Black Friday']
event_impacts = [15, 25, 40]

for i, (date, impact) in enumerate(zip(event_dates, event_impacts)):
    event_idx = (pd.to_datetime(date) - dates[0]).days
    values[event_idx:event_idx+7] += impact

df_ts = pd.DataFrame({'date': dates, 'value': values})

# Create the story-driven visualization
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(df_ts['date'], df_ts['value'], linewidth=2, color='#2E86AB')

# Add event annotations
for date, label, impact in zip(event_dates, event_labels, event_impacts):
    event_date = pd.to_datetime(date)
    event_idx = (event_date - dates[0]).days
    event_value = values[event_idx]
    
    ax.annotate(label, 
                xy=(event_date, event_value),
                xytext=(event_date, event_value + 30),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.set_title('Sales Performance: A Year of Growth and Key Milestones', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales Value', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

### Multi-dimensional Analysis

\`\`\`python
# Bubble chart for multi-dimensional insights
np.random.seed(42)
n_companies = 20

company_data = pd.DataFrame({
    'revenue': np.random.lognormal(10, 1, n_companies),
    'profit_margin': np.random.normal(15, 5, n_companies),
    'employees': np.random.randint(50, 5000, n_companies),
    'industry': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Retail'], n_companies)
})

# Create bubble chart
fig, ax = plt.subplots(figsize=(12, 8))

industry_colors = {'Tech': '#FF6B6B', 'Finance': '#4ECDC4', 
                  'Healthcare': '#45B7D1', 'Retail': '#96CEB4'}

for industry in company_data['industry'].unique():
    industry_data = company_data[company_data['industry'] == industry]
    ax.scatter(industry_data['revenue'], industry_data['profit_margin'],
              s=industry_data['employees']/10, alpha=0.6,
              c=industry_colors[industry], label=industry,
              edgecolors='black', linewidth=1)

ax.set_xlabel('Revenue (Millions)', fontsize=12)
ax.set_ylabel('Profit Margin (%)', fontsize=12)
ax.set_title('Company Performance Analysis\\n(Bubble size = Number of Employees)', 
            fontsize=14, fontweight='bold')
ax.legend(title='Industry', title_fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## 4. Dashboard Design Principles

### Creating Effective Dashboards

\`\`\`python
# Dashboard layout principles
dashboard_principles = {
    'hierarchy': 'Most important metrics at the top-left',
    'grouping': 'Related metrics should be visually grouped',
    'white_space': 'Use white space to avoid clutter',
    'consistency': 'Consistent colors, fonts, and styling',
    'interactivity': 'Allow users to drill down into details',
    'mobile_friendly': 'Ensure responsiveness across devices'
}

# Example dashboard structure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# KPI cards (top row)
kpis = [('Revenue', '$2.4M', '+12%'), ('Users', '45.2K', '+8%'), 
        ('Conversion', '3.2%', '+0.5%'), ('Churn', '2.1%', '-0.3%')]

for i, (title, value, change) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.text(0.5, 0.7, value, ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.4, title, ha='center', va='center', fontsize=14)
    ax.text(0.5, 0.2, change, ha='center', va='center', fontsize=12, 
           color='green' if '+' in change else 'red')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='gray'))

# Main chart (middle)
ax_main = fig.add_subplot(gs[1, :3])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
revenue_trend = [2.1, 2.3, 2.2, 2.5, 2.4, 2.4]
ax_main.plot(months, revenue_trend, marker='o', linewidth=3, markersize=8)
ax_main.set_title('Revenue Trend', fontsize=14, fontweight='bold')
ax_main.grid(True, alpha=0.3)

# Side chart
ax_side = fig.add_subplot(gs[1, 3])
sources = ['Organic', 'Paid', 'Social', 'Email']
values = [40, 30, 20, 10]
ax_side.pie(values, labels=sources, autopct='%1.1f%%')
ax_side.set_title('Traffic Sources', fontsize=14, fontweight='bold')

# Bottom charts
ax_bottom1 = fig.add_subplot(gs[2, :2])
ax_bottom2 = fig.add_subplot(gs[2, 2:])

# User engagement
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
engagement = [85, 92, 88, 95, 90, 75, 70]
ax_bottom1.bar(days, engagement, color='lightblue')
ax_bottom1.set_title('Daily User Engagement', fontsize=14, fontweight='bold')
ax_bottom1.set_ylabel('Engagement Score')

# Geographic distribution
regions = ['North', 'South', 'East', 'West']
users_by_region = [12000, 8500, 15000, 9500]
ax_bottom2.barh(regions, users_by_region, color='lightgreen')
ax_bottom2.set_title('Users by Region', fontsize=14, fontweight='bold')
ax_bottom2.set_xlabel('Number of Users')

plt.suptitle('Executive Dashboard - Q2 2024', fontsize=18, fontweight='bold')
plt.show()
\`\`\`

## 5. Best Practices and Common Pitfalls

### Visualization Checklist

\`\`\`python
visualization_checklist = {
    'clarity': [
        'Is the main message immediately clear?',
        'Are axes properly labeled?',
        'Is the chart type appropriate for the data?'
    ],
    'accuracy': [
        'Do the visual proportions match the data?',
        'Are scales consistent and not misleading?',
        'Is the data source clearly indicated?'
    ],
    'aesthetics': [
        'Is the color scheme accessible?',
        'Is there sufficient contrast?',
        'Is the layout clean and uncluttered?'
    ],
    'context': [
        'Is there enough context for interpretation?',
        'Are comparisons meaningful?',
        'Is the time frame clearly indicated?'
    ]
}
\`\`\`

### Common Mistakes to Avoid

1. **Misleading scales**: Always start bar charts at zero
2. **Too many colors**: Limit to 5-7 colors maximum
3. **3D effects**: They distort perception and add no value
4. **Pie charts with too many slices**: Use bar charts instead
5. **Missing context**: Always provide baselines and benchmarks

## Conclusion

Effective data visualization combines technical skills with design principles and storytelling techniques. Remember:

1. **Start with the story** you want to tell
2. **Choose the right chart** for your data and message
3. **Design for your audience** and context
4. **Iterate and refine** based on feedback
5. **Always prioritize clarity** over complexity
6. **Make it accessible** to all users
7. **Provide context** and actionable insights

### The Visualization Process

1. **Understand your audience** and their needs
2. **Define the key message** you want to communicate
3. **Choose appropriate chart types** for your data
4. **Design with accessibility** in mind
5. **Test and iterate** based on feedback
6. **Document your decisions** for future reference

Great visualizations don't just show data—they reveal insights, inspire action, and drive better decision-making. Practice these principles, and your visualizations will become powerful tools for communication and influence.

### Tools and Resources

- **Python**: Matplotlib, Seaborn, Plotly, Bokeh
- **R**: ggplot2, plotly, shiny
- **Business Intelligence**: Tableau, Power BI, Looker
- **Web**: D3.js, Chart.js, Observable
- **Design**: Adobe Illustrator, Figma, Canva

Invest time in learning these tools and understanding design principles. The combination of technical skills and design thinking will set your visualizations apart and make your data stories truly compelling.
    `,
  },
  {
    id: "sql-advanced-techniques",
    title: "Advanced SQL Techniques for Data Analysts",
    description:
      "Dive deep into advanced SQL concepts including window functions, CTEs, and query optimization.",
    date: "2023-12-20",
    readTime: "14 min read",
    category: "SQL",
    image:
      "https://images.unsplash.com/photo-1544383835-bda2bc66a55d?w=1200&q=80",
    featured: false,
    author: {
      name: "David Kim",
      avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=david",
      bio: "Database Architect and SQL expert with 10+ years of experience in data analytics and optimization.",
    },
    tags: ["SQL", "Database", "Analytics", "Optimization"],
    seo: {
      metaTitle:
        "Advanced SQL Techniques - Window Functions, CTEs & Optimization",
      metaDescription:
        "Master advanced SQL concepts for data analysis. Learn window functions, CTEs, query optimization, and performance tuning.",
      keywords: [
        "sql",
        "window functions",
        "cte",
        "query optimization",
        "database",
        "analytics",
      ],
    },
    content: `
# Advanced SQL Techniques for Data Analysts

SQL is the backbone of data analysis, but mastering its advanced features can dramatically improve your analytical capabilities. This comprehensive guide covers sophisticated SQL techniques that will elevate your data analysis skills and make you a more effective analyst.

## Window Functions: The Game Changer

Window functions perform calculations across a set of table rows related to the current row, without collapsing the result set like GROUP BY would. They're essential for advanced analytics.

### Basic Window Function Syntax

\`\`\`sql
SELECT 
    column1,
    column2,
    WINDOW_FUNCTION() OVER (
        [PARTITION BY column]     -- Divide data into groups
        [ORDER BY column]         -- Define order within partitions
        [ROWS/RANGE frame_specification]  -- Define window frame
    ) AS window_result
FROM table_name;
\`\`\`

### Ranking Functions

\`\`\`sql
-- Sample sales data
WITH sales_data AS (
    SELECT 'John' as salesperson, 'Q1' as quarter, 15000 as sales_amount, 'North' as region
    UNION ALL SELECT 'Jane', 'Q1', 18000, 'South'
    UNION ALL SELECT 'Bob', 'Q1', 12000, 'East'
    UNION ALL SELECT 'Alice', 'Q1', 20000, 'West'
    UNION ALL SELECT 'John', 'Q2', 16000, 'North'
    UNION ALL SELECT 'Jane', 'Q2', 19000, 'South'
    UNION ALL SELECT 'Bob', 'Q2', 14000, 'East'
    UNION ALL SELECT 'Alice', 'Q2', 22000, 'West'
)

SELECT 
    salesperson,
    quarter,
    region,
    sales_amount,
    -- Rank within each quarter (gaps in ranking)
    RANK() OVER (PARTITION BY quarter ORDER BY sales_amount DESC) as rank_in_quarter,
    -- Dense rank (no gaps in ranking)
    DENSE_RANK() OVER (PARTITION BY quarter ORDER BY sales_amount DESC) as dense_rank,
    -- Row number (unique sequential number)
    ROW_NUMBER() OVER (PARTITION BY quarter ORDER BY sales_amount DESC) as row_num,
    -- Percentile rank (0 to 1)
    PERCENT_RANK() OVER (PARTITION BY quarter ORDER BY sales_amount) as percentile_rank,
    -- Ntile for quartiles
    NTILE(4) OVER (PARTITION BY quarter ORDER BY sales_amount) as quartile
FROM sales_data
ORDER BY quarter, sales_amount DESC;
\`\`\`

### Analytical Functions

\`\`\`sql
-- Advanced analytical functions
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(sales_amount) as monthly_revenue
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
)

SELECT 
    month,
    monthly_revenue,
    -- Running total
    SUM(monthly_revenue) OVER (ORDER BY month) as running_total,
    -- Moving average (3-month)
    AVG(monthly_revenue) OVER (
        ORDER BY month 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as moving_avg_3m,
    -- Previous month comparison
    LAG(monthly_revenue, 1) OVER (ORDER BY month) as prev_month,
    -- Next month (for forecasting validation)
    LEAD(monthly_revenue, 1) OVER (ORDER BY month) as next_month,
    -- Month-over-month growth
    (monthly_revenue - LAG(monthly_revenue, 1) OVER (ORDER BY month)) / 
    LAG(monthly_revenue, 1) OVER (ORDER BY month) * 100 as mom_growth_pct,
    -- Year-over-year comparison
    LAG(monthly_revenue, 12) OVER (ORDER BY month) as same_month_last_year,
    -- First and last values in the dataset
    FIRST_VALUE(monthly_revenue) OVER (ORDER BY month) as first_month_sales,
    LAST_VALUE(monthly_revenue) OVER (
        ORDER BY month 
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as last_month_sales
FROM monthly_sales
ORDER BY month;
\`\`\`

## Common Table Expressions (CTEs)

CTEs make complex queries more readable and maintainable by breaking them into logical steps.

### Recursive CTEs

\`\`\`sql
-- Organizational hierarchy analysis
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: top-level managers
    SELECT 
        employee_id,
        name,
        manager_id,
        title,
        1 as level,
        CAST(name AS VARCHAR(1000)) as hierarchy_path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees with managers
    SELECT 
        e.employee_id,
        e.name,
        e.manager_id,
        e.title,
        eh.level + 1,
        eh.hierarchy_path || ' -> ' || e.name
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)

SELECT 
    level,
    COUNT(*) as employees_at_level,
    STRING_AGG(name, ', ') as employee_names
FROM employee_hierarchy
GROUP BY level
ORDER BY level;
\`\`\`

### Complex Multi-CTE Analysis

\`\`\`sql
-- Comprehensive customer analysis
WITH customer_metrics AS (
    -- Calculate customer lifetime metrics
    SELECT 
        customer_id,
        COUNT(*) as total_orders,
        SUM(order_amount) as total_spent,
        AVG(order_amount) as avg_order_value,
        MAX(order_date) as last_order_date,
        MIN(order_date) as first_order_date,
        COUNT(DISTINCT DATE_TRUNC('month', order_date)) as active_months
    FROM orders
    GROUP BY customer_id
),
customer_segments AS (
    -- Segment customers based on RFM analysis
    SELECT 
        customer_id,
        total_orders,
        total_spent,
        avg_order_value,
        active_months,
        -- Recency (days since last order)
        CURRENT_DATE - last_order_date as days_since_last_order,
        -- Frequency score
        CASE 
            WHEN total_orders >= 10 THEN 'High'
            WHEN total_orders >= 5 THEN 'Medium'
            ELSE 'Low'
        END as frequency_score,
        -- Monetary score
        CASE 
            WHEN total_spent >= 1000 THEN 'High'
            WHEN total_spent >= 500 THEN 'Medium'
            ELSE 'Low'
        END as monetary_score,
        -- Customer segment
        CASE 
            WHEN total_spent >= 1000 AND total_orders >= 5 THEN 'VIP'
            WHEN total_spent >= 500 OR total_orders >= 3 THEN 'Regular'
            WHEN CURRENT_DATE - last_order_date <= 90 THEN 'New'
            ELSE 'At Risk'
        END as customer_segment,
        -- Activity status
        CASE 
            WHEN CURRENT_DATE - last_order_date <= 30 THEN 'Active'
            WHEN CURRENT_DATE - last_order_date <= 90 THEN 'At Risk'
            ELSE 'Inactive'
        END as activity_status
    FROM customer_metrics
),
segment_analysis AS (
    -- Analyze segments
    SELECT 
        customer_segment,
        activity_status,
        COUNT(*) as customer_count,
        AVG(total_spent) as avg_total_spent,
        AVG(avg_order_value) as avg_order_value,
        AVG(total_orders) as avg_total_orders,
        AVG(active_months) as avg_active_months
    FROM customer_segments
    GROUP BY customer_segment, activity_status
)

SELECT 
    customer_segment,
    activity_status,
    customer_count,
    ROUND(avg_total_spent, 2) as avg_total_spent,
    ROUND(avg_order_value, 2) as avg_order_value,
    ROUND(avg_total_orders, 1) as avg_total_orders,
    ROUND(avg_active_months, 1) as avg_active_months,
    -- Calculate segment value
    ROUND(customer_count * avg_total_spent, 2) as segment_total_value,
    -- Percentage of total customers
    ROUND(100.0 * customer_count / SUM(customer_count) OVER (), 2) as pct_of_customers
FROM segment_analysis
ORDER BY segment_total_value DESC;
\`\`\`

## Advanced Analytical Patterns

### Cohort Analysis

\`\`\`sql
-- Customer cohort retention analysis
WITH customer_cohorts AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', MIN(order_date)) as cohort_month
    FROM orders
    GROUP BY customer_id
),
cohort_data AS (
    SELECT 
        c.cohort_month,
        DATE_TRUNC('month', o.order_date) as order_month,
        COUNT(DISTINCT o.customer_id) as customers
    FROM customer_cohorts c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.cohort_month, DATE_TRUNC('month', o.order_date)
),
cohort_sizes AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT customer_id) as cohort_size
    FROM customer_cohorts
    GROUP BY cohort_month
)

SELECT 
    cd.cohort_month,
    cd.order_month,
    EXTRACT(MONTH FROM AGE(cd.order_month, cd.cohort_month)) as period_number,
    cd.customers,
    cs.cohort_size,
    ROUND(100.0 * cd.customers / cs.cohort_size, 2) as retention_rate
FROM cohort_data cd
JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
ORDER BY cd.cohort_month, cd.order_month;
\`\`\`

### Time Series Analysis

\`\`\`sql
-- Advanced time series analysis with seasonality
WITH daily_sales AS (
    SELECT 
        DATE(order_date) as sale_date,
        SUM(order_amount) as daily_revenue,
        COUNT(*) as daily_orders
    FROM orders
    GROUP BY DATE(order_date)
),
sales_with_trends AS (
    SELECT 
        sale_date,
        daily_revenue,
        daily_orders,
        -- Day of week analysis
        EXTRACT(DOW FROM sale_date) as day_of_week,
        TO_CHAR(sale_date, 'Day') as day_name,
        -- Seasonal patterns
        EXTRACT(MONTH FROM sale_date) as month_num,
        EXTRACT(QUARTER FROM sale_date) as quarter_num,
        -- Moving averages
        AVG(daily_revenue) OVER (
            ORDER BY sale_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as ma_7_day,
        AVG(daily_revenue) OVER (
            ORDER BY sale_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as ma_30_day,
        -- Volatility measures
        STDDEV(daily_revenue) OVER (
            ORDER BY sale_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as volatility_30_day,
        -- Trend detection
        daily_revenue - LAG(daily_revenue, 7) OVER (ORDER BY sale_date) as wow_change,
        daily_revenue - LAG(daily_revenue, 30) OVER (ORDER BY sale_date) as mom_change
    FROM daily_sales
)

SELECT 
    sale_date,
    daily_revenue,
    day_name,
    ROUND(ma_7_day, 2) as seven_day_avg,
    ROUND(ma_30_day, 2) as thirty_day_avg,
    ROUND(volatility_30_day, 2) as volatility,
    ROUND(wow_change, 2) as week_over_week_change,
    ROUND(mom_change, 2) as month_over_month_change,
    -- Anomaly detection (simple version)
    CASE 
        WHEN ABS(daily_revenue - ma_30_day) > 2 * volatility_30_day 
        THEN 'Anomaly'
        ELSE 'Normal'
    END as anomaly_flag
FROM sales_with_trends
WHERE sale_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY sale_date DESC;
\`\`\`

## Query Optimization Techniques

### Index Strategy

\`\`\`sql
-- Comprehensive indexing strategy

-- Single column indexes for frequent WHERE clauses
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(status);

-- Composite indexes for common query patterns
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
CREATE INDEX idx_orders_date_amount ON orders(order_date, order_amount);
CREATE INDEX idx_products_category_price ON products(category, price);

-- Covering indexes (include frequently selected columns)
CREATE INDEX idx_orders_analysis ON orders(customer_id, order_date) 
INCLUDE (order_amount, status, product_id);

-- Partial indexes for specific conditions
CREATE INDEX idx_active_customers ON customers(customer_id, last_order_date) 
WHERE status = 'Active';

CREATE INDEX idx_high_value_orders ON orders(customer_id, order_date)
WHERE order_amount > 1000;

-- Functional indexes for computed columns
CREATE INDEX idx_orders_month ON orders(DATE_TRUNC('month', order_date));
CREATE INDEX idx_customers_full_name ON customers(LOWER(first_name || ' ' || last_name));
\`\`\`

### Query Performance Optimization

\`\`\`sql
-- Performance optimization techniques

-- Use EXISTS instead of IN for better performance
-- Slow version
SELECT * FROM customers 
WHERE customer_id IN (
    SELECT customer_id FROM orders WHERE order_amount > 1000
);

-- Fast version
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id 
    AND o.order_amount > 1000
);

-- Use LIMIT with ORDER BY efficiently
-- Add appropriate indexes for ORDER BY columns
SELECT customer_id, order_date, order_amount
FROM orders
ORDER BY order_date DESC, order_id DESC
LIMIT 100;

-- Optimize GROUP BY queries
-- Ensure GROUP BY columns are indexed
SELECT 
    customer_id,
    COUNT(*) as order_count,
    SUM(order_amount) as total_spent
FROM orders
WHERE order_date >= '2024-01-01'
GROUP BY customer_id
HAVING COUNT(*) > 5;

-- Use window functions instead of self-joins
-- Slow: self-join approach
SELECT 
    o1.customer_id,
    o1.order_date,
    o1.order_amount,
    o2.prev_order_amount
FROM orders o1
LEFT JOIN orders o2 ON o1.customer_id = o2.customer_id 
    AND o2.order_date = (
        SELECT MAX(order_date) 
        FROM orders o3 
        WHERE o3.customer_id = o1.customer_id 
        AND o3.order_date < o1.order_date
    );

-- Fast: window function approach
SELECT 
    customer_id,
    order_date,
    order_amount,
    LAG(order_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
    ) as prev_order_amount
FROM orders;
\`\`\`

## Advanced Data Quality and Validation

\`\`\`sql
-- Comprehensive data quality checks
WITH data_quality_checks AS (
    SELECT 
        'orders' as table_name,
        'completeness' as check_type,
        'customer_id' as column_name,
        COUNT(*) as total_rows,
        COUNT(customer_id) as non_null_rows,
        ROUND(100.0 * COUNT(customer_id) / COUNT(*), 2) as completeness_pct
    FROM orders
    
    UNION ALL
    
    SELECT 
        'orders',
        'validity',
        'order_amount',
        COUNT(*),
        COUNT(CASE WHEN order_amount > 0 THEN 1 END),
        ROUND(100.0 * COUNT(CASE WHEN order_amount > 0 THEN 1 END) / COUNT(*), 2)
    FROM orders
    
    UNION ALL
    
    SELECT 
        'orders',
        'consistency',
        'order_date',
        COUNT(*),
        COUNT(CASE WHEN order_date <= CURRENT_DATE THEN 1 END),
        ROUND(100.0 * COUNT(CASE WHEN order_date <= CURRENT_DATE THEN 1 END) / COUNT(*), 2)
    FROM orders
)

SELECT 
    table_name,
    check_type,
    column_name,
    total_rows,
    non_null_rows,
    completeness_pct,
    CASE 
        WHEN completeness_pct >= 95 THEN 'PASS'
        WHEN completeness_pct >= 90 THEN 'WARNING'
        ELSE 'FAIL'
    END as quality_status
FROM data_quality_checks
ORDER BY table_name, check_type, column_name;
\`\`\`

## Conclusion

Mastering these advanced SQL techniques will significantly enhance your data analysis capabilities:

### Key Takeaways

1. **Window Functions** are essential for advanced analytics and time series analysis
2. **CTEs** make complex queries readable and maintainable
3. **Proper indexing** is crucial for query performance
4. **Query optimization** techniques can dramatically improve execution time
5. **Data quality checks** should be built into your analytical workflows

### Best Practices

- **Start simple** and gradually add complexity
- **Always consider performance** when writing queries
- **Document your queries** with clear comments
- **Test with realistic data volumes** before deploying to production
- **Monitor query performance** and optimize as needed
- **Use version control** for your SQL scripts

### Next Steps

1. Practice these techniques with your own datasets
2. Learn about query execution plans and how to read them
3. Explore database-specific features (PostgreSQL, SQL Server, etc.)
4. Study advanced topics like query parallelization and partitioning
5. Consider learning about modern analytical databases (Snowflake, BigQuery, etc.)

Remember that the best SQL technique is the one that solves your specific problem efficiently and maintainably. Always consider readability and performance together, and don't hesitate to refactor queries as requirements evolve.

With these advanced techniques in your toolkit, you'll be able to tackle complex analytical challenges and extract deeper insights from your data.
    `,
  },
  {
    id: "python-automation-workflows",
    title: "Automating Data Workflows with Python",
    description:
      "Build robust automation pipelines to streamline your data processing and analysis workflows.",
    date: "2023-12-15",
    readTime: "11 min read",
    category: "Automation",
    image:
      "https://images.unsplash.com/photo-1518186285589-2f7649de83e0?w=1200&q=80",
    featured: false,
    author: {
      name: "Lisa Wang",
      avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=lisa",
      bio: "Automation Engineer specializing in data pipelines and workflow optimization.",
    },
    tags: ["Python", "Automation", "Workflows", "Data Pipeline"],
    seo: {
      metaTitle: "Python Data Workflow Automation - Build Efficient Pipelines",
      metaDescription:
        "Learn to automate data workflows with Python. Build robust pipelines for data processing, analysis, and reporting.",
      keywords: [
        "python",
        "automation",
        "data pipeline",
        "workflow",
        "etl",
        "scheduling",
      ],
    },
    content: `
# Automating Data Workflows with Python

Data automation is essential for scaling analytics operations and ensuring consistent, reliable data processing. This comprehensive guide covers building robust automation pipelines that can handle everything from data extraction to report generation.

## Why Automate Data Workflows?

Automation provides several key benefits:

- **Consistency**: Eliminates human error and ensures reproducible results
- **Efficiency**: Processes data faster than manual operations
- **Scalability**: Handles increasing data volumes without proportional resource increases
- **Reliability**: Runs on schedule without human intervention
- **Monitoring**: Provides visibility into data pipeline health

## 1. Building a Basic ETL Pipeline

### Pipeline Architecture

\`\`\`python
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional

class DataPipeline:
    \"\"\"Base class for data pipeline operations.\"\"\"
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        \"\"\"Configure logging for the pipeline.\"\"\"
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = Path(self.config.get('log_dir', 'logs')) / f\"{datetime.now().strftime('%Y%m%d')}_pipeline.log\"
        log_file.parent.mkdir(exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
            
        return logger
    
    def extract(self) -> pd.DataFrame:
        \"\"\"Extract data from source.\"\"\"
        raise NotImplementedError
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Transform the extracted data.\"\"\"
        raise NotImplementedError
    
    def load(self, data: pd.DataFrame) -> None:
        \"\"\"Load transformed data to destination.\"\"\"
        raise NotImplementedError
    
    def run(self) -> None:
        \"\"\"Execute the complete ETL pipeline.\"\"\"
        try:
            self.logger.info(\"Starting ETL pipeline\")
            
            # Extract
            self.logger.info(\"Extracting data\")
            raw_data = self.extract()
            self.logger.info(f\"Extracted {len(raw_data)} records\")
            
            # Transform
            self.logger.info(\"Transforming data\")
            transformed_data = self.transform(raw_data)
            self.logger.info(f\"Transformed to {len(transformed_data)} records\")
            
            # Load
            self.logger.info(\"Loading data\")
            self.load(transformed_data)
            self.logger.info(\"ETL pipeline completed successfully\")
            
        except Exception as e:
            self.logger.error(f\"Pipeline failed: {str(e)}\")
            raise
\`\`\`

## 2. Scheduling and Orchestration

### Using APScheduler for Job Scheduling

\`\`\`python
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import atexit

class PipelineScheduler:
    \"\"\"Scheduler for automated pipeline execution.\"\"\"
    
    def __init__(self):
        self.scheduler = BlockingScheduler()
        self.jobs = {}
        
        # Graceful shutdown
        atexit.register(lambda: self.scheduler.shutdown())
    
    def add_daily_job(self, func, hour: int = 6, minute: int = 0, **kwargs):
        \"\"\"Add a job that runs daily at specified time.\"\"\"
        job = self.scheduler.add_job(
            func,
            CronTrigger(hour=hour, minute=minute),
            **kwargs
        )
        return job
    
    def start(self):
        \"\"\"Start the scheduler.\"\"\"
        print(\"Starting pipeline scheduler...\")
        self.scheduler.start()
\`\`\`

## Conclusion

Automating data workflows with Python provides the foundation for scalable, reliable data operations. Start with simple ETL pipelines and gradually add complexity as your needs grow.

Remember that good automation requires careful planning, robust error handling, and comprehensive monitoring. Invest time in building these foundations, and your automated workflows will serve you well as your data operations scale.
    `,
  },
];

// Utility functions for blog management
export function getAllPosts(): BlogPost[] {
  return blogPosts.sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
  );
}

export function getFeaturedPosts(): BlogPost[] {
  return blogPosts.filter((post) => post.featured);
}

export function getPostById(id: string): BlogPost | undefined {
  return blogPosts.find((post) => post.id === id);
}

export function getPostsByCategory(category: string): BlogPost[] {
  return blogPosts.filter(
    (post) => post.category.toLowerCase() === category.toLowerCase(),
  );
}

export function getPostsByTag(tag: string): BlogPost[] {
  return blogPosts.filter((post) =>
    post.tags.some((postTag) => postTag.toLowerCase() === tag.toLowerCase()),
  );
}

export function searchPosts(query: string): BlogPost[] {
  const lowercaseQuery = query.toLowerCase();
  return blogPosts.filter(
    (post) =>
      post.title.toLowerCase().includes(lowercaseQuery) ||
      post.description.toLowerCase().includes(lowercaseQuery) ||
      post.content.toLowerCase().includes(lowercaseQuery) ||
      post.tags.some((tag) => tag.toLowerCase().includes(lowercaseQuery)),
  );
}

export function getRelatedPosts(
  currentPost: BlogPost,
  limit: number = 3,
): BlogPost[] {
  const relatedPosts = blogPosts
    .filter((post) => post.id !== currentPost.id)
    .map((post) => {
      let score = 0;

      // Same category gets higher score
      if (post.category === currentPost.category) {
        score += 3;
      }

      // Shared tags get points
      const sharedTags = post.tags.filter((tag) =>
        currentPost.tags.includes(tag),
      );
      score += sharedTags.length * 2;

      return { post, score };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((item) => item.post);

  return relatedPosts;
}

export function getCategories(): string[] {
  const categories = [...new Set(blogPosts.map((post) => post.category))];
  return categories.sort();
}

export function getTags(): string[] {
  const allTags = blogPosts.flatMap((post) => post.tags);
  const uniqueTags = [...new Set(allTags)];
  return uniqueTags.sort();
}

export function getPostStats() {
  return {
    totalPosts: blogPosts.length,
    featuredPosts: blogPosts.filter((post) => post.featured).length,
    categories: getCategories().length,
    tags: getTags().length,
    averageReadTime: Math.round(
      blogPosts.reduce((sum, post) => {
        const minutes = parseInt(post.readTime.split(" ")[0]);
        return sum + minutes;
      }, 0) / blogPosts.length,
    ),
  };
}
