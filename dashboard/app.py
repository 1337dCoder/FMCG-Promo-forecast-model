import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="FMCG Promo Intelligence Platform",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

# Load the trained model and feature list
# Using cache so it only loads once, not on every page refresh
@st.cache_resource
def load_model():
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

# Load the processed dataset
# Again cached so we don't reload 800k rows on every interaction
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/train_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

model, feature_cols = load_model()
df = load_data()

# Sidebar navigation
st.sidebar.title("FMCG Intelligence Platform")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Promotion Simulator",
    "Model Explainability",
    "Store Analytics"
])


# --------------------------------------------------
# PAGE 1 - OVERVIEW
# Shows high level business metrics and trends
# --------------------------------------------------
if page == "Overview":
    st.title("FMCG Promotion Impact Forecasting")
    st.markdown("AI-powered retail intelligence platform built on Rossmann store data")
    st.markdown("---")

    # Top level numbers a business stakeholder would want to see immediately
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stores", "1,115")
    with col2:
        st.metric("Training Records", "813,118")
    with col3:
        st.metric("Model MAPE", "9.85%")
    with col4:
        st.metric("Avg Promo Uplift", "38.8%")

    st.markdown("---")

    # Overall sales trend across all stores over the full date range
    st.subheader("Daily Average Sales Over Time")
    daily = df[df['Sales'] > 0].groupby('Date')['Sales'].mean().reset_index()
    fig = px.line(daily, x='Date', y='Sales',
                  title='Average Daily Sales Across All Stores')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Weekly pattern - shows which days drive the most revenue
        st.subheader("Sales by Day of Week")
        dow = df[df['Sales'] > 0].groupby('DayOfWeek')['Sales'].mean().reset_index()
        dow['Day'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig2 = px.bar(dow, x='Day', y='Sales',
                      color='Sales', color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Store type comparison - Type B is a major outlier worth highlighting
        st.subheader("Sales by Store Type")
        st_sales = df[df['Sales'] > 0].groupby('StoreType')['Sales'].mean().reset_index()
        st_sales['StoreType'] = ['Type A', 'Type B', 'Type C', 'Type D']
        fig3 = px.bar(st_sales, x='StoreType', y='Sales',
                      color='Sales', color_continuous_scale='Reds')
        st.plotly_chart(fig3, use_container_width=True)

    # The core business question - do promotions actually work?
    st.subheader("Promotion Impact on Sales")
    promo_data = df[df['Sales'] > 0].groupby('Promo')['Sales'].mean().reset_index()
    promo_data['Label'] = ['No Promo', 'Promo']
    fig4 = px.bar(promo_data, x='Label', y='Sales',
                  color='Label',
                  color_discrete_map={'No Promo': '#636EFA', 'Promo': '#EF553B'})
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)


# --------------------------------------------------
# PAGE 2 - PROMOTION SIMULATOR
# Manager picks a store and day, we tell them
# whether running a promotion is worth it
# --------------------------------------------------
elif page == "Promotion Simulator":
    st.title("Promotion Simulator")
    st.markdown("Simulate promotion outcomes before committing budget")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        store_id = st.selectbox("Select Store", sorted(df['Store'].unique()))
        day_of_week = st.slider("Day of Week", 1, 7, 4,
                                help="1 = Monday, 7 = Sunday")
        month = st.slider("Month", 1, 12, 7)
        promo2 = st.selectbox("Promo2 Participant", [0, 1])

    with col2:
        # Pull this store's characteristics from the dataset
        store_info = df[df['Store'] == store_id].iloc[-1]
        store_type = int(store_info['StoreType'])
        assortment = int(store_info['Assortment'])
        competition_distance = float(store_info['CompetitionDistance'])

        st.info(f"""
        **Store {store_id} Profile**
        - Type: {['A', 'B', 'C', 'D'][store_type]}
        - Assortment: {['Basic', 'Extra', 'Extended'][assortment]}
        - Competition Distance: {competition_distance:.0f}m
        """)

    if st.button("Run Simulation", type="primary"):

        def get_store_features(store_id):
            # Get the most recent lag and rolling values for this store
            # These represent what the model would know at prediction time
            store_data = df[df['Store'] == store_id].sort_values('Date')
            latest = store_data.iloc[-1]
            return {
                'Sales_Lag_7': latest.get('Sales_Lag_7', 0),
                'Sales_Lag_14': latest.get('Sales_Lag_14', 0),
                'Sales_Lag_28': latest.get('Sales_Lag_28', 0),
                'Sales_Rolling_Mean_7': latest.get('Sales_Rolling_Mean_7', 0),
                'Sales_Rolling_Mean_14': latest.get('Sales_Rolling_Mean_14', 0),
                'Sales_Rolling_Mean_28': latest.get('Sales_Rolling_Mean_28', 0),
                'Sales_Rolling_Std_7': latest.get('Sales_Rolling_Std_7', 0),
                'Sales_Rolling_Std_14': latest.get('Sales_Rolling_Std_14', 0),
                'Sales_Rolling_Std_28': latest.get('Sales_Rolling_Std_28', 0),
                'PromoRunLength': latest.get('PromoRunLength', 0),
                'PromoYesterday': latest.get('PromoYesterday', 0),
                'CompetitionOpenMonths': latest.get('CompetitionOpenMonths', 0),
            }

        store_feats = get_store_features(store_id)

        # Fixed values for the simulation
        day = 15
        year = 2015
        week_of_year = 29
        day_of_year = 196

        base_input = {
            'Store': store_id,
            'DayOfWeek': day_of_week,
            'StateHoliday': 0,
            'SchoolHoliday': 0,
            'StoreType': store_type,
            'Assortment': assortment,
            'CompetitionDistance': competition_distance,
            'Promo2': promo2,
            'Year': year,
            'Month': month,
            'Day': day,
            'WeekOfYear': week_of_year,
            'DayOfYear': day_of_year,
            'IsWeekend': 1 if day_of_week in [6, 7] else 0,
            'IsMonthStart': 0,
            'IsMonthEnd': 0,
            'Quarter': (month - 1) // 3 + 1,
            **store_feats
        }

        # Run the model twice - once with promo off, once with promo on
        # The difference is the predicted uplift
        no_promo_input = {**base_input, 'Promo': 0}
        no_promo_df = pd.DataFrame([no_promo_input]).reindex(
            columns=feature_cols, fill_value=0)
        no_promo_pred = max(0, float(model.predict(no_promo_df)[0]))

        promo_input = {**base_input, 'Promo': 1}
        promo_df = pd.DataFrame([promo_input]).reindex(
            columns=feature_cols, fill_value=0)
        promo_pred = max(0, float(model.predict(promo_df)[0]))

        uplift = promo_pred - no_promo_pred
        uplift_pct = (uplift / no_promo_pred) * 100

        # Simple ROI calculation assuming a flat promotion cost
        # In a real system this would come from the campaign budget
        cost = 500
        roi = ((uplift - cost) / cost) * 100

        st.markdown("---")
        st.subheader("Simulation Results")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sales Without Promo", f"€{no_promo_pred:,.0f}")
        with col2:
            st.metric("Sales With Promo", f"€{promo_pred:,.0f}",
                      delta=f"+€{uplift:,.0f}")
        with col3:
            st.metric("Uplift Percentage", f"{uplift_pct:.1f}%")
        with col4:
            st.metric("ROI", f"{roi:.1f}%")

        # Clear recommendation so the manager does not have to interpret numbers
        if roi > 0:
            st.success(f"Recommendation: Run the promotion. Expected ROI is {roi:.1f}%")
        else:
            st.error(f"Recommendation: Skip this promotion. ROI is negative at {roi:.1f}%")

        # Visual comparison of the two scenarios
        fig = go.Figure(data=[
            go.Bar(name='Without Promo', x=['Forecast'], y=[no_promo_pred],
                   marker_color='#636EFA'),
            go.Bar(name='With Promo', x=['Forecast'], y=[promo_pred],
                   marker_color='#EF553B')
        ])
        fig.update_layout(
            title='Promo vs No Promo Sales Forecast',
            barmode='group',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# PAGE 3 - MODEL EXPLAINABILITY
# Shows stakeholders why the model makes predictions
# This is what makes the system trustworthy
# --------------------------------------------------
elif page == "Model Explainability":
    st.title("Model Explainability")
    st.markdown("Understand why the model makes each prediction using SHAP values")
    st.markdown("---")

    st.subheader("Global Feature Importance")
    st.markdown("Each dot is one prediction. Red means high feature value, blue means low. Width shows how much that feature moves predictions.")

    # Display the SHAP summary plot we generated in Phase 6
    from PIL import Image
    img = Image.open('reports/shap_summary.png')
    st.image(img, use_container_width=True)

    st.markdown("---")
    st.subheader("What each feature means in plain English")

    # Plain language explanations for non-technical stakeholders
    explanations = {
        "Sales_Rolling_Mean_14": "The average sales over the last 14 days. If a store has been performing well recently, it will likely continue to do so.",
        "Sales_Rolling_Mean_28": "The monthly sales trend. Confirms whether the 2-week trend is part of a longer pattern.",
        "Promo": "Whether a promotion is active. Running a promo consistently pushes predictions higher.",
        "DayOfWeek": "The day of the week. Mondays and Sundays tend to be stronger than mid-week.",
        "PromoRunLength": "How many consecutive days a promotion has been running. Longer promos show diminishing returns.",
        "Sales_Rolling_Mean_7": "The weekly sales trend. Captures short term momentum.",
        "Sales_Lag_7": "What this store sold exactly one week ago on the same day.",
        "StoreType": "The store format. Type B stores have significantly higher baseline revenue than others."
    }

    for feature, explanation in explanations.items():
        st.markdown(f"**{feature}**")
        st.caption(explanation)
        st.markdown("")


# --------------------------------------------------
# PAGE 4 - STORE ANALYTICS
# Per store deep dive so managers can understand
# their specific store's patterns
# --------------------------------------------------
elif page == "Store Analytics":
    st.title("Store Analytics")
    st.markdown("Deep dive into individual store performance")
    st.markdown("---")

    store_id = st.selectbox("Select Store", sorted(df['Store'].unique()))
    store_df = df[df['Store'] == store_id].sort_values('Date')

    # Three key numbers for this store at a glance
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_sales = store_df['Sales'].mean()
        st.metric("Avg Daily Sales", f"€{avg_sales:,.0f}")
    with col2:
        promo_days = store_df['Promo'].sum()
        st.metric("Total Promo Days", f"{promo_days:,}")
    with col3:
        promo_sales = store_df[store_df['Promo'] == 1]['Sales'].mean()
        no_promo_sales = store_df[store_df['Promo'] == 0]['Sales'].mean()
        uplift = ((promo_sales - no_promo_sales) / no_promo_sales) * 100
        st.metric("Store Promo Uplift", f"{uplift:.1f}%")

    # Full sales history colored by whether a promo was running
    st.subheader(f"Sales History for Store {store_id}")
    fig = px.line(store_df, x='Date', y='Sales',
                  color='Promo',
                  title=f'Store {store_id} Daily Sales — Red indicates promo days',
                  color_discrete_map={0: '#636EFA', 1: '#EF553B'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly breakdown to spot seasonal patterns for this store
    st.subheader("Monthly Sales Breakdown")
    monthly = store_df.groupby('Month')['Sales'].mean().reset_index()
    fig2 = px.bar(monthly, x='Month', y='Sales',
                  title=f'Store {store_id} Average Sales by Month',
                  color='Sales',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig2, use_container_width=True)