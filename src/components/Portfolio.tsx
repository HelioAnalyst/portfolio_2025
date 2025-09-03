import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Button } from "./ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./ui/accordion";
import { ExternalLink, Github, Code, HelpCircle } from "lucide-react";
import Testimonials from "./Testimonials";

interface Project {
  id: string;
  title: string;
  description: string;
  technologies: string[];
  image: string;
  githubUrl?: string;
  liveUrl?: string;
}

interface Example {
  id: string;
  title: string;
  description: string;
  code: string;
  language: string;
}

interface FAQ {
  id: string;
  question: string;
  answer: string;
}

const projects: Project[] = [
  {
    id: "1",
    title: "Shopify API Integration System",
    description:
      "Automated API integration system for Shopify that reduced manual product update time by 15+ hours per month and improved synchronization speed by 60%.",
    technologies: ["Python", "Shopify API", "BC365", "SQL", "API Integration"],
    image:
      "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=400&q=80",
    githubUrl: "https://github.com/helio/shopify-integration",
    liveUrl: "https://shopify-demo.helio.com",
  },
  {
    id: "2",
    title: "Web Scraping & Data Extraction Tool",
    description:
      "Selenium-based web scraping tool that extracts data from 500+ competitor websites to support strategic product adjustments and market analysis.",
    technologies: ["Python", "Selenium", "BeautifulSoup", "Pandas", "SQLite"],
    image:
      "https://images.unsplash.com/photo-1518186285589-2f7649de83e0?w=400&q=80",
    githubUrl: "https://github.com/helio/web-scraper",
    liveUrl: "https://scraper-demo.helio.com",
  },
  {
    id: "3",
    title: "Order Processing Automation",
    description:
      "Automated order processing system that increased accuracy and boosted monthly orders by 300+ through intelligent API integrations and error handling.",
    technologies: ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"],
    image:
      "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=400&q=80",
    githubUrl: "https://github.com/helio/order-automation",
    liveUrl: "https://orders-demo.helio.com",
  },
  {
    id: "4",
    title: "Data Analysis & Visualization Suite",
    description:
      "Comprehensive Python toolkit for data cleaning, statistical analysis, and visualization using pandas, numpy, and matplotlib for business insights.",
    technologies: [
      "Python",
      "Pandas",
      "NumPy",
      "Matplotlib",
      "Seaborn",
      "Jupyter",
    ],
    image:
      "https://images.unsplash.com/photo-1526379095098-d400fd0bf935?w=400&q=80",
    githubUrl: "https://github.com/helio/data-analysis",
    liveUrl: "https://analysis-demo.helio.com",
  },
  {
    id: "5",
    title: "API Documentation & Testing Framework",
    description:
      "Internal API documentation system and testing framework that enabled smooth onboarding of new developers and reduced integration errors by 20%.",
    technologies: ["Python", "FastAPI", "Swagger", "Pytest", "Docker"],
    image:
      "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=400&q=80",
    githubUrl: "https://github.com/helio/api-docs",
    liveUrl: "https://api-docs-demo.helio.com",
  },
  {
    id: "6",
    title: "Digital Inventory Management System",
    description:
      "Digital inventory tracking system that reduced food waste by 15% through automated monitoring and predictive analytics for restaurant operations.",
    technologies: ["Python", "SQLite", "Pandas", "Tkinter", "Matplotlib"],
    image:
      "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?w=400&q=80",
    githubUrl: "https://github.com/helio/inventory-system",
    liveUrl: "https://inventory-demo.helio.com",
  },
];

const examples: Example[] = [
  {
    id: "1",
    title: "Pandas Data Analysis & Visualization",
    description:
      "Comprehensive data analysis using pandas with statistical insights and advanced visualizations.",
    language: "python",
    code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

class DataAnalyzer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.setup_analysis()
    
    def setup_analysis(self):
        """Initial data setup and cleaning"""
        # Handle missing values
        self.df = self.df.dropna(subset=['revenue', 'customer_id'])
        
        # Convert date columns
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Create derived features
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['revenue_per_customer'] = self.df['revenue'] / self.df['customers']
    
    def generate_insights(self):
        """Generate comprehensive business insights"""
        insights = {}
        
        # Revenue analysis
        insights['total_revenue'] = self.df['revenue'].sum()
        insights['avg_monthly_revenue'] = self.df.groupby('month')['revenue'].mean()
        insights['revenue_growth'] = self.calculate_growth_rate('revenue')
        
        # Customer analysis
        insights['customer_retention'] = self.calculate_retention_rate()
        insights['top_customers'] = self.df.nlargest(10, 'revenue_per_customer')
        
        # Statistical analysis
        insights['revenue_correlation'] = self.df[['revenue', 'customers', 'marketing_spend']].corr()
        
        return insights
    
    def calculate_growth_rate(self, column: str) -> float:
        """Calculate month-over-month growth rate"""
        monthly_data = self.df.groupby('month')[column].sum()
        growth_rates = monthly_data.pct_change().dropna()
        return growth_rates.mean() * 100
    
    def calculate_retention_rate(self) -> float:
        """Calculate customer retention rate"""
        monthly_customers = self.df.groupby('month')['customer_id'].nunique()
        retention_rates = []
        
        for i in range(1, len(monthly_customers)):
            current_customers = set(self.df[self.df['month'] == monthly_customers.index[i]]['customer_id'])
            previous_customers = set(self.df[self.df['month'] == monthly_customers.index[i-1]]['customer_id'])
            
            retained = len(current_customers.intersection(previous_customers))
            retention_rate = retained / len(previous_customers) if previous_customers else 0
            retention_rates.append(retention_rate)
        
        return np.mean(retention_rates) * 100
    
    def create_dashboard_plots(self):
        """Create comprehensive visualization dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Revenue trend
        monthly_revenue = self.df.groupby('month')['revenue'].sum()
        axes[0, 0].plot(monthly_revenue.index, monthly_revenue.values, marker='o')
        axes[0, 0].set_title('Monthly Revenue Trend')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Revenue ($)')
        
        # Customer distribution
        sns.histplot(data=self.df, x='revenue_per_customer', bins=30, ax=axes[0, 1])
        axes[0, 1].set_title('Revenue per Customer Distribution')
        
        # Correlation heatmap
        correlation_matrix = self.df[['revenue', 'customers', 'marketing_spend']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlation Matrix')
        
        # Quarterly performance
        quarterly_data = self.df.groupby('quarter')['revenue'].sum()
        axes[1, 1].bar(quarterly_data.index, quarterly_data.values)
        axes[1, 1].set_title('Quarterly Revenue Performance')
        axes[1, 1].set_xlabel('Quarter')
        axes[1, 1].set_ylabel('Revenue ($)')
        
        plt.tight_layout()
        return fig`,
  },
  {
    id: "2",
    title: "Tableau Dashboard Automation",
    description:
      "Python script to automate Tableau dashboard creation and data refresh using Tableau Server API.",
    language: "python",
    code: `import tableauserverclient as TSC
import pandas as pd
import requests
from typing import Dict, List, Optional
import json
from datetime import datetime

class TableauDashboardManager:
    def __init__(self, server_url: str, username: str, password: str, site_id: str = ''):
        self.server_url = server_url
        self.username = username
        self.password = password
        self.site_id = site_id
        self.server = None
        self.auth_token = None
    
    def connect_to_server(self):
        """Establish connection to Tableau Server"""
        server = TSC.Server(self.server_url, use_server_version=True)
        
        tableau_auth = TSC.TableauAuth(self.username, self.password, site_id=self.site_id)
        
        try:
            server.auth.sign_in(tableau_auth)
            self.server = server
            print(f"Successfully connected to Tableau Server: {self.server_url}")
            return True
        except Exception as e:
            print(f"Failed to connect to Tableau Server: {e}")
            return False
    
    def refresh_data_source(self, datasource_id: str) -> bool:
        """Refresh a specific data source"""
        try:
            datasource = self.server.datasources.get_by_id(datasource_id)
            self.server.datasources.refresh(datasource)
            print(f"Data source {datasource.name} refreshed successfully")
            return True
        except Exception as e:
            print(f"Failed to refresh data source: {e}")
            return False
    
    def publish_workbook(self, workbook_path: str, project_name: str) -> Optional[str]:
        """Publish workbook to Tableau Server"""
        try:
            # Find the project
            all_projects, _ = self.server.projects.get()
            project = next((p for p in all_projects if p.name == project_name), None)
            
            if not project:
                print(f"Project '{project_name}' not found")
                return None
            
            # Create workbook item
            new_workbook = TSC.WorkbookItem(project.id)
            
            # Publish workbook
            new_workbook = self.server.workbooks.publish(
                new_workbook, 
                workbook_path, 
                mode=TSC.Server.PublishMode.Overwrite
            )
            
            print(f"Workbook published successfully. ID: {new_workbook.id}")
            return new_workbook.id
            
        except Exception as e:
            print(f"Failed to publish workbook: {e}")
            return None
    
    def get_dashboard_metrics(self, workbook_id: str) -> Dict:
        """Extract metrics from dashboard views"""
        try:
            workbook = self.server.workbooks.get_by_id(workbook_id)
            
            # Get workbook views
            self.server.workbooks.populate_views(workbook)
            
            metrics = {
                'workbook_name': workbook.name,
                'total_views': len(workbook.views),
                'view_details': [],
                'last_updated': workbook.updated_at
            }
            
            for view in workbook.views:
                view_info = {
                    'view_name': view.name,
                    'view_id': view.id,
                    'view_url': f"{self.server_url}/#/views/{view.content_url}"
                }
                metrics['view_details'].append(view_info)
            
            return metrics
            
        except Exception as e:
            print(f"Failed to get dashboard metrics: {e}")
            return {}
    
    def create_usage_report(self) -> pd.DataFrame:
        """Generate usage report for all dashboards"""
        try:
            all_workbooks, _ = self.server.workbooks.get()
            
            usage_data = []
            for workbook in all_workbooks:
                # Get detailed workbook info
                workbook = self.server.workbooks.get_by_id(workbook.id)
                
                usage_data.append({
                    'workbook_name': workbook.name,
                    'project_name': workbook.project_name,
                    'owner': workbook.owner_id,
                    'created_at': workbook.created_at,
                    'updated_at': workbook.updated_at,
                    'size': workbook.size,
                    'view_count': len(workbook.views) if hasattr(workbook, 'views') else 0
                })
            
            return pd.DataFrame(usage_data)
            
        except Exception as e:
            print(f"Failed to create usage report: {e}")
            return pd.DataFrame()
    
    def disconnect(self):
        """Sign out from Tableau Server"""
        if self.server:
            self.server.auth.sign_out()
            print("Disconnected from Tableau Server")

# Usage example
def main():
    # Initialize Tableau manager
    tableau_manager = TableauDashboardManager(
        server_url="https://your-tableau-server.com",
        username="your-username",
        password="your-password",
        site_id="your-site-id"
    )
    
    # Connect and perform operations
    if tableau_manager.connect_to_server():
        # Refresh data sources
        tableau_manager.refresh_data_source("datasource-id")
        
        # Generate usage report
        usage_report = tableau_manager.create_usage_report()
        usage_report.to_csv('tableau_usage_report.csv', index=False)
        
        # Disconnect
        tableau_manager.disconnect()

if __name__ == "__main__":
    main()`,
  },
  {
    id: "3",
    title: "Advanced Statistical Analysis",
    description:
      "Comprehensive statistical analysis toolkit with hypothesis testing, regression analysis, and predictive modeling.",
    language: "python",
    code: `import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List

class StatisticalAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}
    
    def descriptive_statistics(self) -> Dict:
        """Generate comprehensive descriptive statistics"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        stats_summary = {
            'basic_stats': self.data[numeric_cols].describe(),
            'skewness': self.data[numeric_cols].skew(),
            'kurtosis': self.data[numeric_cols].kurtosis(),
            'correlation_matrix': self.data[numeric_cols].corr()
        }
        
        # Outlier detection using IQR method
        outliers = {}
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers[col] = {
                'count': len(self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]),
                'percentage': len(self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]) / len(self.data) * 100
            }
        
        stats_summary['outliers'] = outliers
        return stats_summary
    
    def hypothesis_testing(self, group_col: str, target_col: str, test_type: str = 'ttest') -> Dict:
        """Perform various hypothesis tests"""
        groups = self.data[group_col].unique()
        
        if test_type == 'ttest' and len(groups) == 2:
            group1 = self.data[self.data[group_col] == groups[0]][target_col]
            group2 = self.data[self.data[group_col] == groups[1]][target_col]
            
            # Perform independent t-test
            statistic, p_value = stats.ttest_ind(group1, group2)
            
            result = {
                'test_type': 'Independent T-Test',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'group1_mean': group1.mean(),
                'group2_mean': group2.mean(),
                'effect_size': (group1.mean() - group2.mean()) / np.sqrt(((group1.var() + group2.var()) / 2))
            }
            
        elif test_type == 'anova' and len(groups) > 2:
            group_data = [self.data[self.data[group_col] == group][target_col] for group in groups]
            statistic, p_value = stats.f_oneway(*group_data)
            
            result = {
                'test_type': 'One-Way ANOVA',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'group_means': {group: self.data[self.data[group_col] == group][target_col].mean() for group in groups}
            }
        
        elif test_type == 'chi2':
            # Chi-square test for categorical variables
            contingency_table = pd.crosstab(self.data[group_col], self.data[target_col])
            statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            result = {
                'test_type': 'Chi-Square Test',
                'statistic': statistic,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05,
                'contingency_table': contingency_table
            }
        
        return result
    
    def regression_analysis(self, target_col: str, feature_cols: List[str]) -> Dict:
        """Perform comprehensive regression analysis"""
        X = self.data[feature_cols]
        y = self.data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        results['linear_regression'] = {
            'r2_score': r2_score(y_test, lr_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
            'coefficients': dict(zip(feature_cols, lr_model.coef_)),
            'intercept': lr_model.intercept_
        }
        
        # Random Forest Regression
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        results['random_forest'] = {
            'r2_score': r2_score(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_))
        }
        
        # Model comparison
        results['model_comparison'] = {
            'best_model': 'Linear Regression' if results['linear_regression']['r2_score'] > results['random_forest']['r2_score'] else 'Random Forest',
            'performance_difference': abs(results['linear_regression']['r2_score'] - results['random_forest']['r2_score'])
        }
        
        return results
    
    def time_series_analysis(self, date_col: str, value_col: str) -> Dict:
        """Perform time series analysis and forecasting"""
        # Ensure date column is datetime
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        ts_data = self.data.set_index(date_col)[value_col].sort_index()
        
        # Basic time series statistics
        results = {
            'trend': 'increasing' if ts_data.iloc[-1] > ts_data.iloc[0] else 'decreasing',
            'volatility': ts_data.std(),
            'mean': ts_data.mean(),
            'seasonal_decomposition': self._seasonal_decompose(ts_data)
        }
        
        # Stationarity test (Augmented Dickey-Fuller)
        adf_statistic, adf_p_value = stats.adfuller(ts_data.dropna())[:2]
        results['stationarity'] = {
            'adf_statistic': adf_statistic,
            'p_value': adf_p_value,
            'is_stationary': adf_p_value < 0.05
        }
        
        return results
    
    def _seasonal_decompose(self, ts_data: pd.Series) -> Dict:
        """Simple seasonal decomposition"""
        # Calculate moving averages for trend
        window = min(12, len(ts_data) // 4)  # Adjust window based on data length
        trend = ts_data.rolling(window=window, center=True).mean()
        
        # Calculate seasonal component (simplified)
        detrended = ts_data - trend
        seasonal = detrended.groupby(detrended.index.month).mean()
        
        return {
            'trend_strength': trend.std() / ts_data.std() if ts_data.std() > 0 else 0,
            'seasonal_strength': seasonal.std() / ts_data.std() if ts_data.std() > 0 else 0
        }
    
    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report"""
        report = []
        report.append("=== STATISTICAL ANALYSIS REPORT ===")
        report.append(f"Dataset Shape: {self.data.shape}")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n")
        
        # Add descriptive statistics summary
        desc_stats = self.descriptive_statistics()
        report.append("DESCRIPTIVE STATISTICS:")
        report.append(f"- Number of numeric variables: {len(desc_stats['basic_stats'].columns)}")
        report.append(f"- Variables with high skewness (>1): {len(desc_stats['skewness'][desc_stats['skewness'] > 1])}")
        
        # Add correlation insights
        corr_matrix = desc_stats['correlation_matrix']
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")
        
        if high_corr_pairs:
            report.append("\nHIGH CORRELATIONS (>0.7):")
            for pair in high_corr_pairs:
                report.append(f"- {pair}")
        
        return "\n".join(report)`,
  },
  {
    id: "4",
    title: "Data Pipeline Processor",
    description:
      "A robust data processing pipeline with error handling, retry logic, and monitoring capabilities.",
    language: "python",
    code: `import logging
import time
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    success: bool
    processed_records: int
    errors: List[str]
    execution_time: float

class DataPipelineProcessor:
    def __init__(self, max_retries: int = 3, batch_size: int = 1000):
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, data: List[Dict[str, Any]]) -> ProcessingResult:
        start_time = time.time()
        processed = 0
        errors = []
        
        for record in data:
            try:
                self._validate_record(record)
                self._transform_record(record)
                self._load_record(record)
                processed += 1
            except Exception as e:
                errors.append(f"Record {record.get('id', 'unknown')}: {str(e)}")
                self.logger.error(f"Processing failed: {e}")
        
        execution_time = time.time() - start_time
        return ProcessingResult(
            success=len(errors) == 0,
            processed_records=processed,
            errors=errors,
            execution_time=execution_time
        )
    
    def _validate_record(self, record: Dict[str, Any]) -> None:
        required_fields = ['id', 'timestamp', 'value']
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Missing required field: {field}")
    
    def _transform_record(self, record: Dict[str, Any]) -> None:
        # Apply business logic transformations
        if 'value' in record:
            record['normalized_value'] = float(record['value']) / 100
    
    def _load_record(self, record: Dict[str, Any]) -> None:
        # Simulate database insertion
        pass`,
  },
  {
    id: "5",
    title: "Machine Learning Model Pipeline",
    description:
      "End-to-end ML pipeline with feature engineering, model training, validation, and deployment preparation.",
    language: "python",
    code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.target_name = ''
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Comprehensive data preprocessing"""
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            X[col].fillna(X[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        
        # Feature engineering
        X = self._create_features(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        self.target_name = target_col
        
        return X, y
    
    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create additional features"""
        # Create interaction features for top correlated pairs
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Create polynomial features for top 2 numeric columns
            top_cols = numeric_cols[:2]
            X[f'{top_cols[0]}_x_{top_cols[1]}'] = X[top_cols[0]] * X[top_cols[1]]
            X[f'{top_cols[0]}_squared'] = X[top_cols[0]] ** 2
        
        # Create binned features
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            X[f'{col}_binned'] = pd.cut(X[col], bins=5, labels=False)
        
        return X
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple models and compare performance"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models
        models_to_train = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            # Use scaled data for logistic regression, original for tree-based models
            if name == 'logistic_regression':
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='accuracy')
            
            model_results = {
                'model': model,
                'accuracy': model.score(X_test_model, y_test),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            if y_pred_proba is not None:
                model_results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, model.feature_importances_))
                model_results['feature_importance'] = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            results[name] = model_results
            self.models[name] = model
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        results['best_model'] = best_model_name
        results['best_accuracy'] = results[best_model_name]['accuracy']
        
        return results
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, model_name: str = 'random_forest') -> Dict:
        """Perform hyperparameter tuning for specified model"""
        if model_name == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:
            return {'error': f'Hyperparameter tuning not implemented for {model_name}'}
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def save_model(self, model_name: str, filepath: str) -> bool:
        """Save trained model to disk"""
        try:
            if model_name in self.models:
                model_package = {
                    'model': self.models[model_name],
                    'scaler': self.scalers.get('standard'),
                    'encoders': self.encoders,
                    'feature_names': self.feature_names,
                    'target_name': self.target_name
                }
                joblib.dump(model_package, filepath)
                return True
            else:
                print(f"Model {model_name} not found")
                return False
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from disk"""
        try:
            model_package = joblib.load(filepath)
            self.models['loaded'] = model_package['model']
            self.scalers['standard'] = model_package.get('scaler')
            self.encoders = model_package.get('encoders', {})
            self.feature_names = model_package.get('feature_names', [])
            self.target_name = model_package.get('target_name', '')
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_new_data(self, new_data: pd.DataFrame, model_name: str = 'loaded') -> np.ndarray:
        """Make predictions on new data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Preprocess new data (apply same transformations)
        processed_data = new_data.copy()
        
        # Apply encoders
        for col, encoder in self.encoders.items():
            if col in processed_data.columns:
                processed_data[col] = encoder.transform(processed_data[col].astype(str))
        
        # Apply scaling if needed
        if model_name == 'logistic_regression' and 'standard' in self.scalers:
            processed_data = self.scalers['standard'].transform(processed_data)
        
        # Make predictions
        predictions = self.models[model_name].predict(processed_data)
        return predictions`,
  },
];

const faqs: FAQ[] = [
  {
    id: "1",
    question:
      "What programming languages and technologies do you specialize in?",
    answer:
      "I specialize in Python for automation and data processing, with expertise in Selenium for web scraping, API integration (especially Shopify API), SQL for database management, and tools like Pandas, NumPy for data analysis. I also have experience with FastAPI, Docker, and various database systems including PostgreSQL and SQLite.",
  },
  {
    id: "2",
    question: "How do you approach API integration and automation projects?",
    answer:
      "I start by understanding the business requirements and existing workflows. Then I design robust API integrations with proper error handling, retry logic, and monitoring. I focus on reducing manual work, improving accuracy, and creating scalable solutions. I always implement comprehensive logging and documentation for maintainability.",
  },
  {
    id: "3",
    question: "What is your experience with web scraping and data extraction?",
    answer:
      "I have extensive experience using Selenium to extract data from 500+ competitor websites. I implement robust scraping systems with error handling, rate limiting, and data validation. I focus on creating respectful scrapers that don't overload servers and can handle dynamic content, pagination, and various website structures.",
  },
  {
    id: "4",
    question: "How do you ensure code quality and system reliability?",
    answer:
      "I implement comprehensive error handling, retry logic with exponential backoff, detailed logging, and automated testing. I create thorough documentation for all systems and use version control best practices. I also implement monitoring systems to track performance and catch issues early.",
  },
  {
    id: "5",
    question: "What types of automation have you implemented?",
    answer:
      "I've automated Shopify product updates reducing manual work by 15+ hours per month, created inventory management systems that reduced waste by 15%, and built order processing automation that increased monthly orders by 300+. I focus on process optimization and creating measurable business value through automation.",
  },
  {
    id: "6",
    question: "How do you handle project management and team collaboration?",
    answer:
      "I work effectively in agile environments, participating in sprints and collaborating with cross-functional teams. I create clear documentation for onboarding new developers, conduct knowledge-sharing workshops, and focus on identifying and addressing workflow inefficiencies based on user feedback.",
  },
];

export default function Portfolio() {
  return (
    <div className="bg-background min-h-screen">
      {/* Hero Section */}
      <section className="brand-gradient text-white py-20">
        <div className="container mx-auto px-4 text-center">
          <div className="flex justify-center mb-8">
            <div className="logo-mark">
              <svg
                width="120"
                height="120"
                viewBox="0 0 120 120"
                className="drop-shadow-lg"
              >
                {/* Outer ring with gap */}
                <circle
                  cx="60"
                  cy="60"
                  r="50"
                  fill="none"
                  stroke="white"
                  strokeWidth="6"
                  strokeDasharray="280 35"
                  strokeDashoffset="-17.5"
                  className="opacity-90"
                />
                {/* Inner H */}
                <g fill="white">
                  <rect x="42" y="35" width="6" height="50" />
                  <rect x="72" y="35" width="6" height="50" />
                  <rect x="42" y="57" width="36" height="6" />
                </g>
              </svg>
            </div>
          </div>
          <div className="mb-6">
            <h1 className="brand-wordmark text-6xl font-semibold mb-2 text-white drop-shadow-sm">
              HELIO
            </h1>
            <p className="brand-tagline text-sm tracking-[0.18em] text-white/90">
              THE ANALYST
            </p>
          </div>
          <p className="text-xl mb-8 max-w-2xl mx-auto leading-relaxed text-white/95">
            Mid-level Python Developer & Data Analyst with 5+ years of
            experience in API integration, automation, and data processing.
            Specialized in Shopify API, web scraping with Selenium, and creating
            efficient data-driven solutions that reduce manual work and improve
            accuracy.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              size="lg"
              className="bg-white text-brand-plum hover:bg-white/90 font-medium"
              asChild
            >
              <a href="#projects">Explore My Work</a>
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="bg-transparent border-white text-white hover:bg-white hover:text-brand-plum font-medium"
              asChild
            >
              <a href="/cv">View CV</a>
            </Button>
          </div>
        </div>
      </section>

      {/* Projects Section */}
      <section id="projects" className="py-20 bg-muted/30">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 text-foreground brand-wordmark">
            Python Development & Automation Projects
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {projects.map((project) => (
              <Card
                key={project.id}
                className="overflow-hidden hover:shadow-lg transition-shadow"
              >
                <div className="aspect-video overflow-hidden">
                  <img
                    src={project.image}
                    alt={project.title}
                    className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                  />
                </div>
                <CardHeader>
                  <CardTitle className="text-xl">{project.title}</CardTitle>
                  <CardDescription>{project.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.technologies.map((tech) => (
                      <span
                        key={tech}
                        className="px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded-full"
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      className="flex items-center gap-2"
                      asChild
                    >
                      <a href={`/projects/${project.id}`}>View Project</a>
                    </Button>
                    {project.githubUrl && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex items-center gap-2"
                        asChild
                      >
                        <a
                          href={project.githubUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <Github className="w-4 h-4" />
                          Code
                        </a>
                      </Button>
                    )}
                    {project.liveUrl && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex items-center gap-2"
                        asChild
                      >
                        <a
                          href={project.liveUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <ExternalLink className="w-4 h-4" />
                          Live Demo
                        </a>
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Examples Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 text-foreground brand-wordmark">
            Python Development & API Integration Examples
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {examples.map((example) => (
              <Card key={example.id} className="overflow-hidden">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Code className="w-5 h-5" />
                    {example.title}
                  </CardTitle>
                  <CardDescription>{example.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                    <pre className="text-sm">
                      <code>{example.code.substring(0, 500)}...</code>
                    </pre>
                  </div>
                  <div className="mt-4">
                    <Button className="flex items-center gap-2" asChild>
                      <a href={`/examples/${example.id}`}>View Full Analysis</a>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-20 bg-muted/30">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 text-foreground brand-wordmark">
            Frequently Asked Questions
          </h2>
          <div className="max-w-3xl mx-auto">
            <Accordion type="single" collapsible className="space-y-4">
              {faqs.map((faq) => (
                <AccordionItem
                  key={faq.id}
                  value={faq.id}
                  className="bg-card rounded-lg px-6 border border-border"
                >
                  <AccordionTrigger className="text-left">
                    <div className="flex items-center gap-3">
                      <HelpCircle className="w-5 h-5 text-primary" />
                      <span className="font-medium">{faq.question}</span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="text-muted-foreground pb-4">
                    {faq.answer}
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <Testimonials />

      {/* Contact Section */}
      <section className="py-20 brand-gradient text-white">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-4xl font-bold mb-6 brand-wordmark">
            Let's Build Something Amazing
          </h2>
          <p className="text-xl mb-8 max-w-2xl mx-auto leading-relaxed text-white/95">
            Ready to automate your workflows and optimize your processes? I'm
            passionate about creating efficient API integrations, robust
            automation systems, and delivering data-driven solutions that reduce
            manual work and improve business efficiency.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              size="lg"
              className="bg-white text-brand-plum hover:bg-white/90 font-medium"
              asChild
            >
              <a href="/contact">Start a Project</a>
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="bg-transparent border-white text-white hover:bg-white hover:text-brand-plum font-medium"
              asChild
            >
              <a href="/cv">Download CV</a>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}
