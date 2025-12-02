# 🚀 Complete Setup Guide

## Step-by-Step Installation

### 1. System Requirements
- **Python**: Version 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM recommended
- **Storage**: 100MB free space

### 2. Quick Installation
```bash
# Extract the zip file
# Navigate to project folder
cd ecommerce-analytics-platform

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run ecommerce_dashboard.py
```

### 3. Detailed Setup Process

#### For Windows:
1. Open Command Prompt (cmd)
2. Navigate to project folder: `cd path/to/ecommerce-analytics-platform`
3. Install requirements: `pip install -r requirements.txt`
4. Start dashboard: `streamlit run ecommerce_dashboard.py`
5. Open browser to: http://localhost:8501

#### For macOS/Linux:
1. Open Terminal
2. Navigate to project folder: `cd /path/to/ecommerce-analytics-platform`
3. Install requirements: `pip install -r requirements.txt`
4. Start dashboard: `streamlit run ecommerce_dashboard.py`
5. Open browser to: http://localhost:8501

### 4. Verification Steps
- ✅ Dashboard loads without errors
- ✅ All 5 pages are accessible
- ✅ Data visualizations render correctly
- ✅ Prediction tools are functional

### 5. Common Issues & Solutions

**Problem**: ModuleNotFoundError
**Solution**: `pip install --upgrade pip && pip install -r requirements.txt`

**Problem**: Dashboard not loading
**Solution**: Check terminal for errors, ensure correct directory

**Problem**: Charts not displaying  
**Solution**: Update browser, clear cache, try different browser

**Problem**: Prediction tools not working
**Solution**: Verify model files are in models/ folder

### 6. Advanced Configuration

#### Custom Port:
```bash
streamlit run ecommerce_dashboard.py --server.port 8502
```

#### Development Mode:
```bash
streamlit run ecommerce_dashboard.py --server.runOnSave true
```

### 7. File Organization Check
Ensure your folder structure matches:
```
ecommerce-analytics-platform/
├── ✅ ecommerce_dashboard.py
├── ✅ requirements.txt
├── ✅ README.md
├── 📁 data/
│   ├── ✅ ecommerce_customers_with_segments.csv
│   └── ✅ ecommerce_orders.csv
├── 📁 models/
│   ├── ✅ sales_prediction_model.pkl
│   ├── ✅ churn_prediction_model.pkl
│   └── ✅ customer_segmentation_model.pkl
└── 📁 config/
    ├── ✅ business_metrics.json
    ├── ✅ model_performance.json
    └── ✅ feature_sets.json
```

### 8. Success Indicators
When properly set up, you should see:
- Streamlit welcome message in terminal
- Dashboard URL (http://localhost:8501)
- No error messages in terminal
- Interactive dashboard in browser

🎉 **Congratulations! Your E-Commerce Analytics Platform is ready!**
