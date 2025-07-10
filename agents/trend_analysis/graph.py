"""
LangGraph-based workflow definition for the Trend Analysis Agent.

This module defines the graph structure and flow for trend analysis.
"""

from typing import Dict, List, Any, TypedDict, Optional, Annotated, Literal
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
from utils.bigquery import BigQueryConnector
from utils.visualization import (
    create_time_series_plot, create_bar_chart, create_pie_chart,
    create_heatmap, save_matplotlib_fig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the state type for the graph
class TrendAnalysisState(TypedDict):
    """State for the trend analysis workflow."""
    # Control flow
    current_agent: str
    next_agent: Optional[str]
    error: Optional[str]
    done: bool
    
    # Configuration
    project_id: str
    dataset_id: str
    timeframe_days: int
    
    # Data
    raw_data: Optional[Dict[str, Any]]
    customer_data: Optional[Dict[str, Any]]
    product_data: Optional[Dict[str, Any]]
    transaction_data: Optional[Dict[str, Any]]
    online_behavior_data: Optional[Dict[str, Any]]
    
    # Analysis results
    category_analysis: Optional[Dict[str, Any]]
    channel_analysis: Optional[Dict[str, Any]]
    customer_analysis: Optional[Dict[str, Any]]
    
    # Visualizations
    visualizations: List[str]
    
    # Report
    insights: List[str]
    report_path: Optional[str]

# Initialize the LLM
def get_llm():
    """Initialize and return the LLM."""
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        temperature=config.GEMINI_TEMPERATURE,
        top_p=config.GEMINI_TOP_P,
        top_k=config.GEMINI_TOP_K,
        max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        google_api_key=config.GEMINI_API_KEY,
    )

# Define the sub-agent functions
def data_collector(state: TrendAnalysisState) -> TrendAnalysisState:
    """
    Collect data from BigQuery for trend analysis.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with collected data
    """
    logger.info("Data Collector: Starting data collection from BigQuery")
    
    try:
        # Initialize BigQuery connector
        bq_connector = BigQueryConnector(project_id=state["project_id"])
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=state["timeframe_days"])
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Collecting data from {start_date_str} to {end_date_str}")
        
        # Get customer data
        customer_data = bq_connector.get_customer_data()
        
        # Get product data
        product_data = bq_connector.get_product_data()
        
        # Get transaction data within timeframe
        transaction_data = bq_connector.get_transactions_data(
            start_date=start_date_str,
            end_date=end_date_str
        )
        
        # Get online behavior data within timeframe
        online_behavior_data = bq_connector.get_online_behavior_data(
            start_date=start_date_str,
            end_date=end_date_str
        )
        
        # Update state with collected data
        state["customer_data"] = {
            "df": customer_data,
            "count": len(customer_data),
            "columns": customer_data.columns.tolist()
        }
        
        state["product_data"] = {
            "df": product_data,
            "count": len(product_data),
            "columns": product_data.columns.tolist()
        }
        
        state["transaction_data"] = {
            "df": transaction_data,
            "count": len(transaction_data),
            "columns": transaction_data.columns.tolist(),
            "start_date": start_date_str,
            "end_date": end_date_str
        }
        
        state["online_behavior_data"] = {
            "df": online_behavior_data,
            "count": len(online_behavior_data),
            "columns": online_behavior_data.columns.tolist(),
            "start_date": start_date_str,
            "end_date": end_date_str
        }
        
        # Set next agent
        state["current_agent"] = "data_collector"
        state["next_agent"] = "category_analyzer"
        state["error"] = None
        
        logger.info("Data Collector: Data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Data Collector: Error collecting data - {str(e)}")
        state["error"] = f"Error collecting data: {str(e)}"
        state["next_agent"] = END
        
    return state

def category_analyzer(state: TrendAnalysisState) -> TrendAnalysisState:
    """
    Analyze product categories based on transaction data.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with category analysis results
    """
    logger.info("Category Analyzer: Starting category analysis")
    
    try:
        # Get data from state
        transaction_data = state["transaction_data"]["df"]
        product_data = state["product_data"]["df"]
        
        # Merge transaction data with product data
        merged_data = transaction_data.merge(
            product_data,
            on="product_id",
            how="left"
        )
        
        # Perform category analysis
        # 1. Category sales summary
        category_sales = merged_data.groupby("category_level_1").agg({
            "total_amount": "sum",
            "transaction_id": "count",
            "quantity": "sum"
        }).reset_index()
        
        category_sales = category_sales.rename(columns={
            "transaction_id": "transaction_count",
            "total_amount": "sales_amount"
        })
        
        # 2. Category growth analysis (comparing first half vs second half of period)
        merged_data["transaction_date"] = pd.to_datetime(merged_data["transaction_date"])
        mid_date = merged_data["transaction_date"].min() + (merged_data["transaction_date"].max() - merged_data["transaction_date"].min()) / 2
        
        first_half = merged_data[merged_data["transaction_date"] < mid_date]
        second_half = merged_data[merged_data["transaction_date"] >= mid_date]
        
        first_half_sales = first_half.groupby("category_level_1")["total_amount"].sum().reset_index()
        first_half_sales = first_half_sales.rename(columns={"total_amount": "first_half_sales"})
        
        second_half_sales = second_half.groupby("category_level_1")["total_amount"].sum().reset_index()
        second_half_sales = second_half_sales.rename(columns={"total_amount": "second_half_sales"})
        
        category_growth = first_half_sales.merge(second_half_sales, on="category_level_1", how="outer").fillna(0)
        category_growth["growth_rate"] = ((category_growth["second_half_sales"] - category_growth["first_half_sales"]) /
                                        category_growth["first_half_sales"] * 100).fillna(0)
        
        # 3. Subcategory analysis
        subcategory_sales = merged_data.groupby(["category_level_1", "category_level_2"]).agg({
            "total_amount": "sum",
            "transaction_id": "count",
            "quantity": "sum"
        }).reset_index()
        
        subcategory_sales = subcategory_sales.rename(columns={
            "transaction_id": "transaction_count",
            "total_amount": "sales_amount"
        })
        
        # 4. Price range analysis
        price_range_sales = merged_data.groupby(["category_level_1", "price_range"]).agg({
            "total_amount": "sum",
            "transaction_id": "count"
        }).reset_index()
        
        price_range_sales = price_range_sales.rename(columns={
            "transaction_id": "transaction_count",
            "total_amount": "sales_amount"
        })
        
        # Store analysis results in state
        state["category_analysis"] = {
            "category_sales": category_sales,
            "category_growth": category_growth,
            "subcategory_sales": subcategory_sales,
            "price_range_sales": price_range_sales
        }
        
        # Set next agent
        state["current_agent"] = "category_analyzer"
        state["next_agent"] = "channel_analyzer"
        state["error"] = None
        
        logger.info("Category Analyzer: Category analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Category Analyzer: Error analyzing categories - {str(e)}")
        state["error"] = f"Error analyzing categories: {str(e)}"
        state["next_agent"] = END
        
    return state

def channel_analyzer(state: TrendAnalysisState) -> TrendAnalysisState:
    """
    Analyze online vs offline channel performance.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with channel analysis results
    """
    logger.info("Channel Analyzer: Starting channel analysis")
    
    try:
        # Get data from state
        transaction_data = state["transaction_data"]["df"]
        online_behavior_data = state["online_behavior_data"]["df"]
        
        # Extract online purchases
        online_purchases = online_behavior_data[online_behavior_data["event_type"] == "Purchase Complete"]
        
        # Prepare data for comparison
        # Note: In real implementation, we would need to handle the mapping between
        # online purchases and actual transactions, but for this example we'll
        # treat them as separate channels
        
        # Offline channel analysis
        offline_sales_by_date = transaction_data.groupby(
            pd.to_datetime(transaction_data["transaction_date"]).dt.date
        ).agg({
            "total_amount": "sum",
            "transaction_id": "count"
        }).reset_index()
        
        offline_sales_by_date = offline_sales_by_date.rename(columns={
            "transaction_id": "transaction_count",
            "total_amount": "sales_amount",
            "transaction_date": "date"
        })
        
        # Online channel analysis (simplified)
        online_events_by_date = online_behavior_data.groupby(
            pd.to_datetime(online_behavior_data["event_timestamp"]).dt.date
        ).agg({
            "event_id": "count"
        }).reset_index()
        
        online_events_by_date = online_events_by_date.rename(columns={
            "event_id": "event_count",
            "event_timestamp": "date"
        })
        
        # Online event type analysis
        event_type_counts = online_behavior_data.groupby("event_type").size().reset_index(name="count")
        
        # Online to offline conversion analysis
        # This is a simplified version - in reality would need more complex logic
        online_product_views = online_behavior_data[online_behavior_data["event_type"] == "Product View"]
        
        # Store analysis results in state
        state["channel_analysis"] = {
            "offline_sales_by_date": offline_sales_by_date,
            "online_events_by_date": online_events_by_date,
            "event_type_counts": event_type_counts,
            "offline_transaction_count": len(transaction_data),
            "online_event_count": len(online_behavior_data),
            "online_purchase_count": len(online_purchases)
        }
        
        # Set next agent
        state["current_agent"] = "channel_analyzer"
        state["next_agent"] = "customer_analyzer"
        state["error"] = None
        
        logger.info("Channel Analyzer: Channel analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Channel Analyzer: Error analyzing channels - {str(e)}")
        state["error"] = f"Error analyzing channels: {str(e)}"
        state["next_agent"] = END
        
    return state

def customer_analyzer(state: TrendAnalysisState) -> TrendAnalysisState:
    """
    Analyze customer behavior and segments.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with customer analysis results
    """
    logger.info("Customer Analyzer: Starting customer analysis")
    
    try:
        # Get data from state
        customer_data = state["customer_data"]["df"]
        transaction_data = state["transaction_data"]["df"]
        online_behavior_data = state["online_behavior_data"]["df"]
        
        # Merge customer data with transactions
        customer_transactions = transaction_data.merge(
            customer_data,
            on="customer_id",
            how="left"
        )
        
        # 1. Customer segment analysis
        segment_analysis = customer_transactions.groupby([
            "life_stage", "gender", "membership_tier"
        ]).agg({
            "total_amount": "sum",
            "transaction_id": "count",
            "customer_id": "nunique"
        }).reset_index()
        
        segment_analysis = segment_analysis.rename(columns={
            "transaction_id": "transaction_count",
            "customer_id": "customer_count",
            "total_amount": "sales_amount"
        })
        
        # 2. Customer age group analysis
        customer_transactions["age_group"] = pd.cut(
            customer_transactions["age"],
            bins=[0, 20, 30, 40, 50, 60, 100],
            labels=["~20", "21-30", "31-40", "41-50", "51-60", "61+"]
        )
        
        age_group_analysis = customer_transactions.groupby("age_group").agg({
            "total_amount": "sum",
            "transaction_id": "count",
            "customer_id": "nunique"
        }).reset_index()
        
        age_group_analysis = age_group_analysis.rename(columns={
            "transaction_id": "transaction_count",
            "customer_id": "customer_count",
            "total_amount": "sales_amount"
        })
        
        # 3. Region analysis
        region_analysis = customer_transactions.groupby("region").agg({
            "total_amount": "sum",
            "transaction_id": "count",
            "customer_id": "nunique"
        }).reset_index()
        
        region_analysis = region_analysis.rename(columns={
            "transaction_id": "transaction_count",
            "customer_id": "customer_count",
            "total_amount": "sales_amount"
        })
        
        # 4. Customer purchase frequency
        purchase_frequency = transaction_data.groupby("customer_id").size().reset_index(name="purchase_count")
        purchase_frequency_distribution = purchase_frequency["purchase_count"].value_counts().reset_index()
        purchase_frequency_distribution.columns = ["frequency", "customer_count"]
        
        # 5. Average transaction value by segment
        customer_transactions["transaction_value"] = customer_transactions["total_amount"] / customer_transactions["quantity"]
        avg_transaction_value = customer_transactions.groupby("life_stage")["transaction_value"].mean().reset_index()
        
        # Store analysis results in state
        state["customer_analysis"] = {
            "segment_analysis": segment_analysis,
            "age_group_analysis": age_group_analysis,
            "region_analysis": region_analysis,
            "purchase_frequency": purchase_frequency,
            "purchase_frequency_distribution": purchase_frequency_distribution,
            "avg_transaction_value": avg_transaction_value
        }
        
        # Set next agent
        state["current_agent"] = "customer_analyzer"
        state["next_agent"] = "trend_visualizer"
        state["error"] = None
        
        logger.info("Customer Analyzer: Customer analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Customer Analyzer: Error analyzing customers - {str(e)}")
        state["error"] = f"Error analyzing customers: {str(e)}"
        state["next_agent"] = END
        
    return state

def trend_visualizer(state: TrendAnalysisState) -> TrendAnalysisState:
    """
    Create visualizations based on analysis results.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with visualization paths
    """
    logger.info("Trend Visualizer: Creating visualizations")
    
    try:
        visualizations = []
        
        # 1. Category Sales Visualization
        category_sales = state["category_analysis"]["category_sales"].copy()
        # Convert to proper data types to avoid dtype errors
        category_sales['category_level_1'] = category_sales['category_level_1'].astype(str)
        category_sales['sales_amount'] = pd.to_numeric(category_sales['sales_amount'], errors='coerce')
        category_sales['transaction_count'] = pd.to_numeric(category_sales['transaction_count'], errors='coerce')
        category_sales['quantity'] = pd.to_numeric(category_sales['quantity'], errors='coerce')
        
        # Create a clean DataFrame with only the needed columns
        plot_df = pd.DataFrame({
            'category_level_1': category_sales['category_level_1'],
            'sales_amount': category_sales['sales_amount']
        })
        
        fig = create_bar_chart(
            plot_df,
            x_column="category_level_1",
            y_column="sales_amount",
            title="Sales by Category"
        )
        
        viz_path = save_matplotlib_fig(fig, "category_sales.png")
        visualizations.append({
            "path": viz_path,
            "title": "Sales by Category",
            "description": "Chart showing total sales amount for each product category."
        })
        
        # 2. Category Growth Visualization
        category_growth = state["category_analysis"]["category_growth"].copy()
        # Convert to proper data types
        category_growth['category_level_1'] = category_growth['category_level_1'].astype(str)
        category_growth['growth_rate'] = pd.to_numeric(category_growth['growth_rate'], errors='coerce')
        category_growth['first_half_sales'] = pd.to_numeric(category_growth['first_half_sales'], errors='coerce')
        category_growth['second_half_sales'] = pd.to_numeric(category_growth['second_half_sales'], errors='coerce')
        
        # Create a clean DataFrame with only the needed columns
        plot_df = pd.DataFrame({
            'category_level_1': category_growth['category_level_1'],
            'growth_rate': category_growth['growth_rate']
        })
        
        fig = create_bar_chart(
            plot_df,
            x_column="category_level_1",
            y_column="growth_rate",
            title="Growth Rate by Category"
        )
        
        viz_path = save_matplotlib_fig(fig, "category_growth.png")
        visualizations.append({
            "path": viz_path,
            "title": "Growth Rate by Category",
            "description": "Chart showing growth rate for each product category."
        })
        
        # 3. Online vs Offline Channel Comparison
        offline_sales = state["channel_analysis"]["offline_sales_by_date"].copy()
        # Convert to proper data types
        offline_sales['date'] = pd.to_datetime(offline_sales['date'])
        offline_sales['sales_amount'] = pd.to_numeric(offline_sales['sales_amount'], errors='coerce')
        offline_sales['transaction_count'] = pd.to_numeric(offline_sales['transaction_count'], errors='coerce')
        
        # Create a clean DataFrame with only the needed columns
        plot_df = pd.DataFrame({
            'date': offline_sales['date'],
            'sales_amount': offline_sales['sales_amount']
        })
        
        # Create a time series plot for offline sales
        fig = create_time_series_plot(
            plot_df,
            x_column="date",
            y_columns="sales_amount",
            title="Daily Offline Sales Trend"
        )
        
        viz_path = save_matplotlib_fig(fig, "offline_sales_trend.png")
        visualizations.append({
            "path": viz_path,
            "title": "Daily Offline Sales Trend",
            "description": "Chart showing daily sales trend for offline channel."
        })
        
        # 4. Customer Segment Analysis
        segment_analysis = state["customer_analysis"]["segment_analysis"].copy()
        # Convert sales_amount to numeric
        segment_analysis['sales_amount'] = pd.to_numeric(segment_analysis['sales_amount'], errors='coerce')
        
        # Create pivot table
        pivot_df = segment_analysis.pivot_table(
            index="life_stage",
            columns="gender",
            values="sales_amount",
            aggfunc="sum"
        ).fillna(0)
        
        # Convert all values to numeric
        pivot_df = pivot_df.astype(float)
        
        fig = create_heatmap(
            pivot_df,
            title="Sales by Life Stage and Gender"
        )
        
        viz_path = save_matplotlib_fig(fig, "segment_heatmap.png")
        visualizations.append({
            "path": viz_path,
            "title": "Sales by Life Stage and Gender",
            "description": "Heatmap showing sales distribution by customer life stage and gender."
        })
        
        # 5. Age Group Analysis
        age_group_analysis = state["customer_analysis"]["age_group_analysis"].copy()
        # Convert to proper data types
        age_group_analysis['age_group'] = age_group_analysis['age_group'].astype(str)
        age_group_analysis['sales_amount'] = pd.to_numeric(age_group_analysis['sales_amount'], errors='coerce')
        
        # Create a clean DataFrame with only the needed columns
        plot_df = pd.DataFrame({
            'age_group': age_group_analysis['age_group'],
            'sales_amount': age_group_analysis['sales_amount']
        })
        
        fig = create_bar_chart(
            plot_df,
            x_column="age_group",
            y_column="sales_amount",
            title="Sales by Age Group"
        )
        
        viz_path = save_matplotlib_fig(fig, "age_group_sales.png")
        visualizations.append({
            "path": viz_path,
            "title": "Sales by Age Group",
            "description": "Chart showing total sales amount by customer age group."
        })
        
        # 6. Region Analysis
        region_analysis = state["customer_analysis"]["region_analysis"].copy()
        # Convert to proper data types
        region_analysis['region'] = region_analysis['region'].astype(str)
        region_analysis['sales_amount'] = pd.to_numeric(region_analysis['sales_amount'], errors='coerce')
        
        # Create a clean DataFrame with only the needed columns
        plot_df = pd.DataFrame({
            'region': region_analysis['region'],
            'sales_amount': region_analysis['sales_amount']
        })
        
        fig = create_pie_chart(
            plot_df,
            values_column="sales_amount",
            names_column="region",
            title="Sales Distribution by Region"
        )
        
        viz_path = save_matplotlib_fig(fig, "region_sales.png")
        visualizations.append({
            "path": viz_path,
            "title": "Sales Distribution by Region",
            "description": "Pie chart showing sales distribution by region."
        })
        
        # Store visualization paths in state
        state["visualizations"] = visualizations
        
        # Set next agent
        state["current_agent"] = "trend_visualizer"
        state["next_agent"] = "report_generator"
        state["error"] = None
        
        logger.info(f"Trend Visualizer: Created {len(visualizations)} visualizations")
        
    except Exception as e:
        logger.error(f"Trend Visualizer: Error creating visualizations - {str(e)}")
        state["error"] = f"Error creating visualizations: {str(e)}"
        state["next_agent"] = END
        
    return state

def report_generator(state: TrendAnalysisState) -> TrendAnalysisState:
    """
    Generate a PDF report with analysis results and insights.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with report path and insights
    """
    logger.info("Report Generator: Generating trend analysis report")
    
    try:
        from utils.pdf_generator import ReportPDF
        import pandas as pd
        
        # Initialize insights list if not present
        if "insights" not in state or state["insights"] is None:
            state["insights"] = []
        
        # Generate insights using LLM if not already present
        if len(state["insights"]) == 0:
            llm = get_llm()
            
            # Prepare prompt with analysis results
            category_analysis = state["category_analysis"]
            channel_analysis = state["channel_analysis"]
            customer_analysis = state["customer_analysis"]
            
            # Convert DataFrames to string representations for the prompt
            category_sales_str = category_analysis["category_sales"].to_string()
            category_growth_str = category_analysis["category_growth"].to_string()
            segment_analysis_str = customer_analysis["segment_analysis"].to_string()
            
            prompt = f"""
            You are a data analyst for an electronics retail company. Please review the following analysis results and identify key trends and insights.

            ## Category Analysis

            Category Sales:
            {category_sales_str}

            Category Growth Rates:
            {category_growth_str}

            ## Channel Analysis

            Offline Transaction Count: {channel_analysis["offline_transaction_count"]}
            Online Event Count: {channel_analysis["online_event_count"]}
            Online Purchase Count: {channel_analysis["online_purchase_count"]}

            ## Customer Analysis

            Segment Analysis:
            {segment_analysis_str}

            Based on the data above, please provide:

            1. Five key trends
            2. Three business opportunities
            3. Two areas needing improvement
            4. Three marketing strategy recommendations

            Please provide specific, data-driven insights for each item.
            """
            
            # Get insights from LLM
            response = llm.invoke([HumanMessage(content=prompt)])
            insights_text = response.content
            
            # Parse insights into list
            lines = insights_text.split('\n')
            insights = []
            current_insight = ""
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or
                             line.startswith('3.') or line.startswith('4.') or
                             line.startswith('5.') or line.startswith('- ')):
                    if current_insight:
                        insights.append(current_insight)
                    current_insight = line
                elif line and current_insight:
                    current_insight += " " + line
                    
            if current_insight:
                insights.append(current_insight)
                
            state["insights"] = insights
        
        # Create PDF report
        report = ReportPDF()
        report.set_title("Electronics Retail CDP Trend Analysis Report")
        report.add_date()
        
        # Add executive summary
        summary = "\n".join(state["insights"][:3])  # Use first 3 insights as summary
        report.add_executive_summary(summary)
        
        # Add category analysis section
        report.add_section("Category Analysis")
        report.add_text("Analysis of product category sales trends and growth rates.")
        
        # Add category visualizations
        for viz in state["visualizations"]:
            if "category" in viz["title"].lower():
                report.add_image(viz["path"], width=160, caption=viz["title"])
                report.add_text(viz["description"])
        
        # Add channel analysis section
        report.add_section("Channel Analysis")
        report.add_text("Comparison of online and offline channel performance.")
        
        # Add channel visualizations
        for viz in state["visualizations"]:
            if "offline" in viz["title"].lower() or "online" in viz["title"].lower():
                report.add_image(viz["path"], width=160, caption=viz["title"])
                report.add_text(viz["description"])
        
        # Add customer analysis section
        report.add_section("Customer Analysis")
        report.add_text("Analysis of customer segments and purchasing patterns.")
        
        # Add customer visualizations
        for viz in state["visualizations"]:
            if "segment" in viz["title"].lower() or "age" in viz["title"].lower() or "region" in viz["title"].lower():
                report.add_image(viz["path"], width=160, caption=viz["title"])
                report.add_text(viz["description"])
        
        # Add insights section
        report.add_section("Key Insights")
        for insight in state["insights"]:
            report.add_text("- " + insight)
        
        # Add recommendations section
        report.add_section("Recommendations")
        report.add_text("Based on the analysis, the following actions are recommended:")
        
        # Extract recommendations from insights
        recommendations = [insight for insight in state["insights"] if "recommend" in insight.lower() or "suggest" in insight.lower()]
        
        if recommendations:
            for rec in recommendations:
                report.add_text("- " + rec)
        else:
            report.add_text("- Strengthen marketing for high-growth categories")
            report.add_text("- Develop personalized promotions for different customer segments")
            report.add_text("- Enhance synergy between online and offline channels")
        
        # Generate and save the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(config.REPORTS_DIR, f"trend_analysis_{timestamp}.pdf")
        report.output(report_path)
        
        # Update state with report path
        state["report_path"] = report_path
        
        # Set next agent
        state["current_agent"] = "report_generator"
        state["next_agent"] = END
        state["done"] = True
        state["error"] = None
        
        logger.info(f"Report Generator: Report generated successfully at {report_path}")
        
    except Exception as e:
        logger.error(f"Report Generator: Error generating report - {str(e)}")
        state["error"] = f"Error generating report: {str(e)}"
        state["next_agent"] = END
        
    return state

# Define the routing logic for the graph
def router(state: TrendAnalysisState) -> str:
    """
    Route to the next agent based on the state.
    
    Args:
        state: Current workflow state
        
    Returns:
        Name of the next agent to call
    """
    if state.get("error"):
        return END
        
    return state.get("next_agent", END)

# Build the graph
def build_trend_analysis_graph() -> StateGraph:
    """
    Build and return the trend analysis workflow graph.
    
    Returns:
        StateGraph for trend analysis
    """
    # Create a new graph
    workflow = StateGraph(TrendAnalysisState)
    
    # Add nodes for each sub-agent
    workflow.add_node("data_collector", data_collector)
    workflow.add_node("category_analyzer", category_analyzer)
    workflow.add_node("channel_analyzer", channel_analyzer)
    workflow.add_node("customer_analyzer", customer_analyzer)
    workflow.add_node("trend_visualizer", trend_visualizer)
    workflow.add_node("report_generator", report_generator)
    
    # Set the entry point
    workflow.set_entry_point("data_collector")
    
    # Add conditional edges based on the router function
    workflow.add_conditional_edges(
        "data_collector",
        router,
        {
            "category_analyzer": "category_analyzer",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "category_analyzer",
        router,
        {
            "channel_analyzer": "channel_analyzer",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "channel_analyzer",
        router,
        {
            "customer_analyzer": "customer_analyzer",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "customer_analyzer",
        router,
        {
            "trend_visualizer": "trend_visualizer",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "trend_visualizer",
        router,
        {
            "report_generator": "report_generator",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "report_generator",
        router,
        {
            END: END
        }
    )
    
    # Compile the graph
    return workflow.compile()

# Create a TrendAnalysisGraph class for easy usage
class TrendAnalysisGraph:
    """Class for running the trend analysis workflow."""
    
    def __init__(self):
        """Initialize the trend analysis graph."""
        self.graph = build_trend_analysis_graph()
    
    def run(
        self,
        project_id: str = None,
        dataset_id: str = None,
        timeframe_days: int = None
    ) -> Dict[str, Any]:
        """
        Run the trend analysis workflow.
        
        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
            timeframe_days: Number of days to analyze
            
        Returns:
            Dictionary with analysis results and report path
        """
        # Set default values if not provided
        project_id = project_id or config.GCP_PROJECT_ID
        dataset_id = dataset_id or config.BQ_DATASET_ID
        timeframe_days = timeframe_days or config.TREND_ANALYSIS_TIMEFRAME_DAYS
        
        # Initialize the state
        initial_state = TrendAnalysisState(
            current_agent="data_collector",
            next_agent=None,
            error=None,
            done=False,
            project_id=project_id,
            dataset_id=dataset_id,
            timeframe_days=timeframe_days,
            raw_data=None,
            customer_data=None,
            product_data=None,
            transaction_data=None,
            online_behavior_data=None,
            category_analysis=None,
            channel_analysis=None,
            customer_analysis=None,
            visualizations=[],
            insights=[],
            report_path=None
        )
        
        # Run the graph
        logger.info(f"Starting trend analysis for project {project_id}, dataset {dataset_id}")
        result = self.graph.invoke(initial_state)
        
        # Check for errors
        if result.get("error"):
            logger.error(f"Trend analysis failed: {result['error']}")
            raise Exception(result["error"])
        
        # Return the result
        logger.info("Trend analysis completed successfully")
        return {
            "report_path": result.get("report_path"),
            "insights": result.get("insights", []),
            "visualizations": result.get("visualizations", [])
        }

# For testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the graph
    trend_analysis = TrendAnalysisGraph()
    result = trend_analysis.run()
    
    print(f"Report generated at: {result['report_path']}")
    print(f"Generated {len(result['visualizations'])} visualizations")
    print(f"Found {len(result['insights'])} insights")
