import streamlit as st
import json
from demand_agent import calculate_demand_forecast, calculate_inventory_reorder

st.set_page_config(page_title="AI Demand Forecasting Agent", layout="wide")

st.title("📦 Autonomous Demand Forecasting & Inventory Agent")
st.caption("Symbiosis Institute of Technology, Nagpur | Agentic AI & Automation")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Parameters")
    sku = st.selectbox("Select Product SKU", ["SKU-101", "SKU-202", "SKU-303"])
    periods = st.slider("Forecast Periods (Weeks)", min_value=1, max_value=8, value=4)
    current_stock = st.number_input("Current Warehouse Stock (Units)", value=150, min_value=0)
    lead_time = st.number_input("Supplier Lead Time (Weeks)", value=2, min_value=1)
    
    run_btn = st.button("Run Agent Evaluation", type="primary")

with col2:
    st.subheader("Agent Execution & Recommendations")
    if run_btn:
        with st.spinner("Agent running forecasting and inventory tools..."):
            # Execute forecasting tool
            forecast_raw = calculate_demand_forecast(sku, periods)
            forecast = json.loads(forecast_raw)
            
            # Execute inventory tool
            inv_raw = calculate_inventory_reorder(
                projected_demand=forecast["total_forecasted_demand"],
                lead_time_weeks=lead_time,
                current_stock=current_stock
            )
            inv = json.loads(inv_raw)
            
            # Metrics Display
            m1, m2, m3 = st.columns(3)
            m1.metric("Weekly Trend", f"{forecast['weekly_trend_slope']} units/wk", forecast["trend_trajectory"])
            m2.metric("Projected Demand", f"{forecast['total_forecasted_demand']} units")
            m3.metric("Reorder Point", f"{inv['reorder_point_threshold']} units")
            
            st.divider()
            
            # Decision Status Banner
            if inv["replenishment_trigger"] == "REORDER_NOW":
                st.error(f"🚨 **Action Required:** Place immediate order for **{inv['recommended_order_quantity']} units**.")
            else:
                st.success("✅ **Stock Sufficient:** No immediate purchase needed.")
                
            st.write("#### Detailed Projections:")
            st.write(f"- **Next {periods} Weeks Forecast (units/wk):** {forecast['period_projections']}")
            st.write(f"- **Safety Stock Allocated:** {inv['computed_safety_stock']} units")
            st.write(f"- **Risk Profile:** {inv['stockout_risk_profile']}")
    else:
        st.info("Select parameters on the left and click **Run Agent Evaluation** to trigger the agent.")
      
