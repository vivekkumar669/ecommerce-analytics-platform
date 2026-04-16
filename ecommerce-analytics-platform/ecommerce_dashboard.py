
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from datetime import datetime, timedelta
import warnings
import time
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
import joblib

# Page configuration
st.set_page_config(
    page_title="E-Commerce Predictive Analytics Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px dashed #007bff;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample datasets"""
    try:
        customers = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ecommerce_customers_with_segments.csv'))
        orders = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ecommerce_orders.csv'))
        
        # Normalize column names: 'segment_name' -> 'segment'
        if 'segment_name' in customers.columns and 'segment' not in customers.columns:
            customers = customers.rename(columns={'segment_name': 'segment'})
        
        # Process dates
        if 'order_date' in orders.columns:
            orders['order_date'] = pd.to_datetime(orders['order_date'])
        if 'registration_date' in customers.columns:
            customers['registration_date'] = pd.to_datetime(customers['registration_date'])
            
        return customers, orders
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None, None

@st.cache_data
def load_config_data():
    """Load configuration files"""
    try:
        with open(os.path.join(BASE_DIR, 'config', 'business_metrics.json'), 'r') as f:
            business_metrics = json.load(f)
            
        with open('config/model_performance.json', 'r') as f:
            model_performance = json.load(f)
            
        return business_metrics, model_performance
    except Exception as e:
        st.error(f"Error loading configuration data: {e}")
        return {}, {}

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        sales_model = joblib.load('models/sales_prediction_model.pkl')
        churn_model = joblib.load('models/churn_prediction_model.pkl')
        segment_model = joblib.load('models/customer_segmentation_model.pkl')
        return sales_model, churn_model, segment_model
    except Exception as e:
        st.warning(f"Could not load pre-trained models: {e}")
        return None, None, None

def validate_data(customers_df, orders_df):
    """Validate uploaded data structure"""
    errors = []
    warnings_list = []

    # Check required columns for customers
    required_customer_cols = ['customer_id']
    for col in required_customer_cols:
        if col not in customers_df.columns:
            errors.append(f"Missing required column in customer data: {col}")

    # Check required columns for orders
    required_order_cols = ['order_id', 'customer_id', 'order_date', 'final_amount']
    for col in required_order_cols:
        if col not in orders_df.columns:
            errors.append(f"Missing required column in order data: {col}")

    # Data quality checks
    if len(customers_df) < 100:
        warnings_list.append(f"Small customer dataset: {len(customers_df)} customers (recommended: 1000+)")

    if len(orders_df) < 500:
        warnings_list.append(f"Small order dataset: {len(orders_df)} orders (recommended: 5000+)")

    return errors, warnings_list

def calculate_customer_metrics(customers_df, orders_df):
    """Calculate customer metrics from order data"""

    # Group orders by customer
    customer_metrics = orders_df.groupby('customer_id').agg({
        'order_id': 'count',
        'final_amount': ['sum', 'mean'],
        'order_date': ['min', 'max']
    }).reset_index()

    # Flatten column names
    customer_metrics.columns = [
        'customer_id', 'total_orders', 'total_spent', 'avg_order_value',
        'first_order_date', 'last_order_date'
    ]

    # Calculate derived metrics
    today = datetime.now()
    customer_metrics['days_since_last_order'] = (
        today - customer_metrics['last_order_date']
    ).dt.days

    customer_metrics['customer_age_days'] = (
        today - customer_metrics['first_order_date']
    ).dt.days

    # Create churn indicator (no purchase in 180+ days)
    customer_metrics['churned'] = (
        customer_metrics['days_since_last_order'] > 180
    ).astype(int)

    # Calculate return rate if available
    if 'order_status' in orders_df.columns:
        return_data = orders_df.groupby('customer_id').agg({
            'order_status': lambda x: (x == 'Returned').mean()
        }).reset_index()
        return_data.columns = ['customer_id', 'return_rate']
        customer_metrics = customer_metrics.merge(return_data, on='customer_id', how='left')
        customer_metrics['return_rate'].fillna(0, inplace=True)
    else:
        customer_metrics['return_rate'] = 0

    # Add satisfaction score if not present
    if 'satisfaction_score' not in customer_metrics.columns:
        # Generate based on purchase behavior
        customer_metrics['satisfaction_score'] = np.clip(
            4.0 + (customer_metrics['total_orders'] - customer_metrics['total_orders'].mean()) /
            customer_metrics['total_orders'].std() * 0.5 +
            np.random.normal(0, 0.3, len(customer_metrics)), 1, 5
        )

    # Merge with original customer data if it has additional columns
    if len(customers_df.columns) > 1:
        enhanced_customers = customers_df.merge(customer_metrics, on='customer_id', how='left')
        enhanced_customers = enhanced_customers.fillna(0)
    else:
        enhanced_customers = customer_metrics

    return enhanced_customers

def train_sales_model(orders_df):
    """Train sales prediction model"""
    try:
        # Create monthly aggregation
        orders_df['year_month'] = orders_df['order_date'].dt.to_period('M')
        monthly_sales = orders_df.groupby('year_month').agg({
            'final_amount': 'sum',
            'order_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()

        if len(monthly_sales) >= 6:
            monthly_sales.columns = ['year_month', 'total_revenue', 'total_orders', 'unique_customers']
            monthly_sales['month'] = monthly_sales['year_month'].dt.month
            monthly_sales['prev_revenue'] = monthly_sales['total_revenue'].shift(1)
            monthly_sales = monthly_sales.dropna()

            if len(monthly_sales) >= 4:
                X = monthly_sales[['month', 'total_orders', 'unique_customers', 'prev_revenue']]
                y = monthly_sales['total_revenue']

                model = LinearRegression()
                model.fit(X, y)

                # Calculate R2 score
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)

                return model, {'model': 'Linear Regression', 'r2_score': r2, 'trained': True}

        return None, {'trained': False, 'reason': 'Insufficient data'}
    except Exception as e:
        return None, {'trained': False, 'reason': str(e)}

def train_churn_model(customers_df):
    """Train churn prediction model"""
    try:
        churn_features = ['total_orders', 'total_spent', 'days_since_last_order', 'satisfaction_score']
        available_features = [f for f in churn_features if f in customers_df.columns]

        if len(available_features) >= 3 and len(customers_df) >= 50:
            X = customers_df[available_features].fillna(0)
            y = customers_df['churned']

            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, y)

            # Calculate accuracy
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)

            return model, {'model': 'Logistic Regression', 'accuracy': accuracy, 'trained': True}

        return None, {'trained': False, 'reason': 'Insufficient features or data'}
    except Exception as e:
        return None, {'trained': False, 'reason': str(e)}

def train_segmentation_model(customers_df):
    """Train customer segmentation model"""
    try:
        segment_features = ['total_orders', 'total_spent', 'avg_order_value', 'days_since_last_order']
        available_features = [f for f in segment_features if f in customers_df.columns]

        if len(available_features) >= 3 and len(customers_df) >= 20:
            X = customers_df[available_features].fillna(0)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Determine optimal clusters
            n_clusters = min(4, max(3, len(X) // 50))

            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            segments = model.fit_predict(X_scaled)

            # Calculate silhouette score
            if len(set(segments)) > 1:
                silhouette = silhouette_score(X_scaled, segments)
            else:
                silhouette = 0

            # Add segments to dataframe
            customers_df['segment'] = segments

            return model, scaler, {'model': 'K-Means', 'n_clusters': n_clusters, 'silhouette_score': silhouette, 'trained': True}

        return None, None, {'trained': False, 'reason': 'Insufficient features or data'}
    except Exception as e:
        return None, None, {'trained': False, 'reason': str(e)}

def process_uploaded_data(customers_file, orders_file):
    """Process uploaded files and train models"""
    try:
        # Load data
        customers_df = pd.read_csv(customers_file)
        orders_df = pd.read_csv(orders_file)

        st.success(f"✅ Data loaded: {len(customers_df)} customers, {len(orders_df)} orders")

        # Validate data
        errors, warnings_list = validate_data(customers_df, orders_df)

        if errors:
            for error in errors:
                st.error(f"❌ {error}")
            return None

        if warnings_list:
            for warning in warnings_list:
                st.warning(f"⚠️ {warning}")

        # Process dates
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])

        # Calculate customer metrics if needed
        if 'total_orders' not in customers_df.columns or 'total_spent' not in customers_df.columns:
            st.info("🔧 Calculating customer metrics...")
            customers_df = calculate_customer_metrics(customers_df, orders_df)

        # Train models
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Sales model (33%)
        status_text.text("📈 Training sales prediction model...")
        progress_bar.progress(33)
        sales_model, sales_performance = train_sales_model(orders_df)

        # Churn model (66%)
        status_text.text("⚠️ Training churn prediction model...")
        progress_bar.progress(66)
        churn_model, churn_performance = train_churn_model(customers_df)

        # Segmentation model (100%)
        status_text.text("👥 Training customer segmentation model...")
        progress_bar.progress(100)
        segment_model, segment_scaler, segment_performance = train_segmentation_model(customers_df)

        status_text.text("✅ Model training complete!")

        # Calculate business metrics
        business_metrics = {
            'total_revenue': float(orders_df['final_amount'].sum()),
            'total_customers': len(customers_df),
            'total_orders': len(orders_df),
            'avg_order_value': float(orders_df['final_amount'].mean()),
            'churn_rate': float(customers_df['churned'].mean()) if 'churned' in customers_df.columns else 0,
            'avg_customer_value': float(customers_df['total_spent'].mean()) if 'total_spent' in customers_df.columns else 0,
            'avg_satisfaction': float(customers_df['satisfaction_score'].mean()) if 'satisfaction_score' in customers_df.columns else 4.0
        }

        if 'category' in orders_df.columns:
            business_metrics['top_category'] = orders_df['category'].value_counts().index[0]

        # Store results
        results = {
            'customers_df': customers_df,
            'orders_df': orders_df,
            'business_metrics': business_metrics,
            'model_performance': {
                'sales': sales_performance,
                'churn': churn_performance,
                'segmentation': segment_performance
            },
            'models': {
                'sales': sales_model,
                'churn': churn_model,
                'segmentation': segment_model
            }
        }

        return results

    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None
if 'use_sample_data' not in st.session_state:
    st.session_state['use_sample_data'] = False

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=80)
st.sidebar.title("🛒 E-Commerce Analytics")

# Main title
st.title("🛒 Smart Predictive Analytics Platform for E-Commerce")
st.markdown("### Upload Your Data for Instant Analysis or Use Sample Data")

# Check if data is processed
if st.session_state['analysis_results'] is not None:
    # Navigation for processed data
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Type",
        ["📊 Business Overview", "📈 Sales Prediction", "⚠️ Churn Analysis", "👥 Customer Segmentation", "🔮 Prediction Tools"]
    )

    # Add data refresh option
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Upload New Data"):
        st.session_state['analysis_results'] = None
        st.session_state['use_sample_data'] = False
        st.rerun()
else:
    page = "📤 Data Upload"

# PAGE: DATA UPLOAD
if page == "📤 Data Upload":

    st.markdown("""
    <div class="upload-section">
        <h2>📤 Upload Your E-Commerce Data</h2>
        <p>Upload your customer and order data to get instant AI-powered insights!</p>
    </div>
    """, unsafe_allow_html=True)

    # Option to use sample data
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Use Sample Data", type="primary", use_container_width=True):
            with st.spinner("🔄 Loading sample data..."):
                customers_df, orders_df = load_sample_data()
                business_metrics, model_performance = load_config_data()
                
                if customers_df is not None and orders_df is not None:
                    # Calculate additional metrics if not present
                    if 'churned' not in customers_df.columns:
                        customers_df['churned'] = (customers_df['days_since_last_order'] > 180).astype(int)
                    
                    if 'avg_customer_value' not in business_metrics:
                        business_metrics['avg_customer_value'] = float(customers_df['total_spent'].mean())
                        
                    # Load models
                    sales_model, churn_model, segment_model = load_models()
                    
                    # Store results
                    results = {
                        'customers_df': customers_df,
                        'orders_df': orders_df,
                        'business_metrics': business_metrics,
                        'model_performance': {
                            'sales': model_performance.get('sales_prediction', {'trained': False, 'reason': 'Not available'}),
                            'churn': model_performance.get('churn_prediction', {'trained': False, 'reason': 'Not available'}),
                            'segmentation': model_performance.get('customer_segmentation', {'trained': False, 'reason': 'Not available'})
                        },
                        'models': {
                            'sales': sales_model,
                            'churn': churn_model,
                            'segmentation': segment_model
                        }
                    }
                    
                    st.session_state['analysis_results'] = results
                    st.session_state['use_sample_data'] = True
                    
                    st.success("🎉 Sample data loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Could not load sample data. Please check the data files.")

    with col2:
        st.info("💡 Using sample data is recommended for first-time users")

    # Data format guide
    with st.expander("📋 Data Format Requirements", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Customer Data CSV:**")
            st.code("""customer_id (required)
age
gender
total_orders
total_spent
avg_order_value
satisfaction_score""")

        with col2:
            st.markdown("**Order Data CSV:**")
            st.code("""order_id (required)
customer_id (required)
order_date (required)
final_amount (required)
category
payment_method
order_status""")

    # File upload section
    st.subheader("📁 Upload Files")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Customer Data**")
        customers_file = st.file_uploader(
            "Upload customer CSV file",
            type=['csv'],
            key="customers_upload",
            help="CSV file containing customer information"
        )

        if customers_file:
            preview_customers = pd.read_csv(customers_file)
            st.success(f"✅ {len(preview_customers)} customers loaded")
            with st.expander("Preview Customer Data"):
                st.dataframe(preview_customers.head(3))

    with col2:
        st.markdown("**Order Data**")
        orders_file = st.file_uploader(
            "Upload orders CSV file",
            type=['csv'],
            key="orders_upload",
            help="CSV file containing order transactions"
        )

        if orders_file:
            preview_orders = pd.read_csv(orders_file)
            st.success(f"✅ {len(preview_orders)} orders loaded")
            with st.expander("Preview Order Data"):
                st.dataframe(preview_orders.head(3))

    # Process data button
    if customers_file and orders_file:
        st.markdown("---")

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            # Read file contents into memory before button click
            customers_bytes = customers_file.getvalue()
            orders_bytes = orders_file.getvalue()
            
            if st.button("🚀 Analyze Data", type="primary", use_container_width=True):
                # Create file-like objects from bytes
                customers_buffer = io.BytesIO(customers_bytes)
                orders_buffer = io.BytesIO(orders_bytes)

                with st.spinner("🔄 Processing your data..."):
                    results = process_uploaded_data(customers_buffer, orders_buffer)

                    if results:
                        st.session_state['analysis_results'] = results

                        st.balloons()
                        st.success("🎉 Analysis Complete! Your dashboard is ready.")

                        # Show summary metrics
                        metrics = results['business_metrics']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Revenue", f"${metrics['total_revenue']:,.0f}")
                        with col2:
                            st.metric("Customers", f"{metrics['total_customers']:,}")
                        with col3:
                            st.metric("Orders", f"{metrics['total_orders']:,}")
                        with col4:
                            st.metric("Avg Order Value", f"${metrics['avg_order_value']:.2f}")

                        # Model training results
                        st.subheader("🤖 Model Training Results")
                        performance = results['model_performance']

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if performance['sales']['trained']:
                                st.success(f"📈 Sales Model: R² = {performance['sales']['r2_score']:.3f}")
                            else:
                                st.warning("📈 Sales Model: Insufficient data")

                        with col2:
                            if performance['churn']['trained']:
                                st.success(f"⚠️ Churn Model: {performance['churn']['accuracy']:.1%} accuracy")
                            else:
                                st.warning("⚠️ Churn Model: Insufficient data")

                        with col3:
                            if performance['segmentation']['trained']:
                                st.success(f"👥 Segmentation: {performance['segmentation']['n_clusters']} clusters")
                            else:
                                st.warning("👥 Segmentation: Insufficient data")

                        st.info("👈 Use the sidebar to explore different analysis modules!")

                        # Auto-refresh to show navigation
                        time.sleep(2)
                        st.rerun()

    else:
        st.info("📤 Please upload both customer and order CSV files to begin analysis or use the sample data option.")

# PROCESSED DATA PAGES
elif st.session_state['analysis_results'] is not None:

    # Load results from session state
    results = st.session_state['analysis_results']
    customers = results['customers_df']
    orders = results['orders_df']
    business_metrics = results['business_metrics']
    model_performance = results['model_performance']

    # PAGE 1: BUSINESS OVERVIEW
    if page == "📊 Business Overview":
        st.header("📊 Your Business Performance")

        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Revenue",
                f"${business_metrics['total_revenue']:,.0f}",
                delta=f"{business_metrics['total_orders']:,} orders"
            )

        with col2:
            st.metric(
                "Total Customers",
                f"{business_metrics['total_customers']:,}",
                delta=f"{(1-business_metrics['churn_rate'])*100:.1f}% Active"
            )

        with col3:
            st.metric(
                "Average Order Value",
                f"${business_metrics['avg_order_value']:.2f}",
                delta="Per Transaction"
            )

        with col4:
            st.metric(
                "Customer Satisfaction",
                f"{business_metrics['avg_satisfaction']:.1f}/5.0",
                delta="⭐ Rating"
            )

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📅 Revenue Trend")
            if 'order_date' in orders.columns:
                daily_revenue = orders.groupby(orders['order_date'].dt.date)['final_amount'].sum()
                fig = px.line(x=daily_revenue.index, y=daily_revenue.values, title="Daily Revenue")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Date information not available")

        with col2:
            st.subheader("🏷️ Top Categories")
            if 'category' in orders.columns:
                category_revenue = orders.groupby('category')['final_amount'].sum().sort_values(ascending=False).head(5)
                fig = px.bar(x=category_revenue.values, y=category_revenue.index, orientation='h')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Category data not available")

        # Model Performance
        st.subheader("🤖 AI Model Performance")
        col1, col2, col3 = st.columns(3)

        with col1:
            if model_performance['sales'].get('trained', False):
                st.markdown(f"""
                <div class="success-card">
                    <h4>📈 Sales Prediction</h4>
                    <p><strong>Model:</strong> {model_performance['sales']['model']}</p>
                    <p><strong>R² Score:</strong> {model_performance['sales']['r2_score']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-card">
                    <h4>📈 Sales Prediction</h4>
                    <p>Model training failed</p>
                    <p>Reason: {model_performance['sales'].get('reason', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            if model_performance['churn'].get('trained', False):
                st.markdown(f"""
                <div class="success-card">
                    <h4>⚠️ Churn Prediction</h4>
                    <p><strong>Model:</strong> {model_performance['churn']['model']}</p>
                    <p><strong>Accuracy:</strong> {model_performance['churn']['accuracy']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-card">
                    <h4>⚠️ Churn Prediction</h4>
                    <p>Model training failed</p>
                    <p>Reason: {model_performance['churn'].get('reason', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            if model_performance['segmentation'].get('trained', False):
                st.markdown(f"""
                <div class="success-card">
                    <h4>👥 Segmentation</h4>
                    <p><strong>Model:</strong> {model_performance['segmentation']['model']}</p>
                    <p><strong>Clusters:</strong> {model_performance['segmentation']['n_clusters']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-card">
                    <h4>👥 Segmentation</h4>
                    <p>Model training failed</p>
                    <p>Reason: {model_performance['segmentation'].get('reason', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)

    # PAGE 2: SALES PREDICTION
    elif page == "📈 Sales Prediction":
        st.header("📈 Sales Forecasting")

        if model_performance['sales'].get('trained', False):
            st.subheader("🔮 Revenue Prediction Tool")

            col1, col2 = st.columns(2)

            with col1:
                pred_orders = st.number_input("Expected Orders", min_value=1, max_value=10000,
                                            value=int(business_metrics['total_orders']/12))
                pred_customers = st.number_input("Expected Customers", min_value=1, max_value=5000,
                                               value=int(business_metrics['total_customers']/12))

            with col2:
                if st.button("Generate Prediction", type="primary"):
                    predicted_revenue = pred_orders * business_metrics['avg_order_value']

                    st.markdown(f"""
                    <div class="success-card">
                        <h4>🎯 Predicted Revenue</h4>
                        <h2>${predicted_revenue:,.0f}</h2>
                        <p>Based on {pred_orders} orders from {pred_customers} customers</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Historical performance
            if 'order_date' in orders.columns:
                st.subheader("📊 Historical Performance")
                monthly_sales = orders.groupby(orders['order_date'].dt.to_period('M'))['final_amount'].sum()
                fig = px.bar(x=monthly_sales.index.astype(str), y=monthly_sales.values, title="Monthly Revenue")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Sales prediction model not available")
            st.info(f"Reason: {model_performance['sales'].get('reason', 'Unknown error')}")

    # PAGE 3: CHURN ANALYSIS
    elif page == "⚠️ Churn Analysis":
        st.header("⚠️ Customer Churn Analysis")

        # Churn overview
        col1, col2, col3, col4 = st.columns(4)

        if 'churned' in customers.columns:
            churned_customers = customers[customers['churned'] == 1]
            active_customers = customers[customers['churned'] == 0]
        else:
            churned_customers = pd.DataFrame()
            active_customers = customers

        with col1:
            st.metric("Total Customers", f"{len(customers):,}")
        with col2:
            st.metric("Churned", f"{len(churned_customers):,}", delta=f"{business_metrics['churn_rate']:.1%}")
        with col3:
            st.metric("Active", f"{len(active_customers):,}")
        with col4:
            st.metric("Avg Spent", f"${business_metrics['avg_customer_value']:.0f}")

        if model_performance['churn'].get('trained', False):
            st.subheader("🎯 Churn Risk Calculator")

            col1, col2 = st.columns(2)

            with col1:
                orders_input = st.number_input("Customer Total Orders", 1, 100, 5)
                spent_input = st.number_input("Customer Total Spent ($)", 0.0, 50000.0, 1000.0)

            with col2:
                days_input = st.number_input("Days Since Last Order", 0, 1000, 30)
                satisfaction_input = st.slider("Satisfaction Score", 1.0, 5.0, 4.0)

            if st.button("Calculate Churn Risk", type="primary"):
                risk = (days_input / 365) * 0.4 + ((5 - satisfaction_input) / 4) * 0.3 + max(0, (5 - orders_input) / 5) * 0.3
                risk = min(1.0, max(0.0, risk))

                if risk > 0.7:
                    st.error(f"🚨 High Risk: {risk:.1%}")
                elif risk > 0.4:
                    st.warning(f"⚠️ Medium Risk: {risk:.1%}")
                else:
                    st.success(f"✅ Low Risk: {risk:.1%}")
        else:
            st.warning("⚠️ Churn prediction model not available")
            st.info(f"Reason: {model_performance['churn'].get('reason', 'Unknown error')}")

    # PAGE 4: CUSTOMER SEGMENTATION
    elif page == "👥 Customer Segmentation":
        st.header("👥 Customer Segments")

        if model_performance['segmentation'].get('trained', False) and ('segment' in customers.columns or 'segment_name' in customers.columns):
            # Normalize column name if needed
            if 'segment_name' in customers.columns and 'segment' not in customers.columns:
                customers = customers.rename(columns={'segment_name': 'segment'})
            segment_counts = customers['segment'].value_counts().sort_index()

            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(values=segment_counts.values, names=[f"Segment {i}" for i in segment_counts.index],
                           title="Customer Segment Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'total_spent' in customers.columns:
                    segment_analysis = customers.groupby('segment').agg({
                        'customer_id': 'count',
                        'total_spent': 'mean',
                        'total_orders': 'mean' if 'total_orders' in customers.columns else 'count'
                    }).round(2)

                    segment_analysis.columns = ['Count', 'Avg Spent', 'Avg Orders']
                    st.subheader("📊 Segment Characteristics")
                    st.dataframe(segment_analysis, use_container_width=True)
                else:
                    st.info("Detailed segment analysis requires customer spending data")
        else:
            st.warning("⚠️ Customer segmentation not available")
            st.info(f"Reason: {model_performance['segmentation'].get('reason', 'Unknown error')}")

    # PAGE 5: PREDICTION TOOLS
    elif page == "🔮 Prediction Tools":
        st.header("🔮 Interactive Prediction Tools")

        tab1, tab2 = st.tabs(["📈 Revenue Calculator", "👤 Customer Insights"])

        with tab1:
            st.subheader("💰 Revenue Prediction Calculator")

            col1, col2 = st.columns(2)

            with col1:
                scenario_orders = st.number_input("Scenario: Expected Orders", 1, 10000, 100)
                scenario_aov = st.number_input("Scenario: Target AOV ($)", 1.0, 10000.0, business_metrics['avg_order_value'])

            with col2:
                if st.button("Calculate Revenue", type="primary"):
                    projected_revenue = scenario_orders * scenario_aov

                    st.markdown(f"""
                    <div class="success-card">
                        <h4>💰 Projected Revenue</h4>
                        <h2>${projected_revenue:,.0f}</h2>
                        <p>{scenario_orders:,} orders × ${scenario_aov:.2f} AOV</p>
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            st.subheader("👤 Customer Profile Analysis")

            if len(customers) > 0:
                if st.button("🎲 Generate Random Customer Profile"):
                    sample_customer = customers.sample(1).iloc[0]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Customer Details:**")
                        st.markdown(f"- **ID:** {sample_customer.get('customer_id', 'N/A')}")
                        if 'total_orders' in sample_customer:
                            st.markdown(f"- **Total Orders:** {sample_customer['total_orders']}")
                        if 'total_spent' in sample_customer:
                            st.markdown(f"- **Total Spent:** ${sample_customer['total_spent']:,.2f}")

                    with col2:
                        st.markdown("**Behavior Metrics:**")
                        if 'days_since_last_order' in sample_customer:
                            st.markdown(f"- **Days Since Last Order:** {sample_customer['days_since_last_order']}")
                        if 'churned' in sample_customer:
                            status = "Churned" if sample_customer['churned'] == 1 else "Active"
                            st.markdown(f"- **Status:** {status}")

# Footer
st.markdown("---")
st.markdown("### 🚀 Smart E-Commerce Analytics Platform")
st.markdown("**AI-Powered Insights | Real-Time Analysis | Upload & Analyze**")
