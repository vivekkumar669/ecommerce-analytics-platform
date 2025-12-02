import pandas as pd
import streamlit as st
import io

# Test the file upload functionality
def test_file_processing():
    print("Testing file processing...")
    
    # Read the sample files
    try:
        customers_df = pd.read_csv('data/ecommerce_customers_with_segments.csv')
        orders_df = pd.read_csv('data/ecommerce_orders.csv')
        
        print(f"Customers data shape: {customers_df.shape}")
        print(f"Orders data shape: {orders_df.shape}")
        
        # Check required columns
        required_customer_cols = ['customer_id']
        required_order_cols = ['order_id', 'customer_id', 'order_date', 'final_amount']
        
        print("Checking required columns...")
        for col in required_customer_cols:
            if col not in customers_df.columns:
                print(f"Missing customer column: {col}")
            else:
                print(f"Found customer column: {col}")
                
        for col in required_order_cols:
            if col not in orders_df.columns:
                print(f"Missing order column: {col}")
            else:
                print(f"Found order column: {col}")
                
        # Check date processing
        print("Processing dates...")
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        print(f"Date range: {orders_df['order_date'].min()} to {orders_df['order_date'].max()}")
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in test: {e}")
        return False

if __name__ == "__main__":
    test_file_processing()