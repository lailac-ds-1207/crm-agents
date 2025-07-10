"""
LangGraph-based workflow definition for the Demand Forecast Agent.
This module defines the graph structure and flow for demand forecasting.
"""
from typing import Dict, List, Any, TypedDict, Optional, Annotated, Literal
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import config
from utils.bigquery import BigQueryConnector
from utils.visualization import create_time_series_plot, save_plotly_fig

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the state type for the graph
class ForecastState(TypedDict):
    """State for the demand forecast workflow."""
    # Control flow
    current_agent: str
    next_agent: Optional[str]
    error: Optional[str]
    done: bool
    
    # Configuration
    project_id: str
    dataset_id: str
    forecast_horizon_weeks: int
    
    # Data
    raw_data: Optional[Dict[str, Any]]
    time_series_data: Optional[Dict[str, Any]]
    category_data: Optional[Dict[str, pd.DataFrame]]
    
    # Model selection and training
    selected_models: Optional[Dict[str, str]]
    trained_models: Optional[Dict[str, Any]]
    model_metrics: Optional[Dict[str, Dict[str, float]]]
    
    # Forecast results
    total_forecast: Optional[Dict[str, Any]]
    category_forecasts: Optional[Dict[str, Dict[str, Any]]]
    growth_rates: Optional[Dict[str, Any]]
    seasonality: Optional[Dict[str, Any]]
    
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

def data_preparer(state: ForecastState) -> ForecastState:
    """
    Prepare time series data for forecasting.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with prepared time series data
    """
    logger.info("Data Preparer: Starting time series data preparation")
    
    try:
        # Initialize BigQuery connector
        bq_connector = BigQueryConnector(project_id=state["project_id"])
        
        # Get transaction data for time series preparation
        query = f"""
        SELECT 
            PARSE_DATE('%Y-%m-%d', transaction_date) AS date,
            p.category_level_1,
            COUNT(t.transaction_id) as sales_count,
            SUM(t.total_amount) as sales_amount,
            SUM(t.quantity) as quantity_sold
        FROM 
            `{state["dataset_id"]}.offline_transactions` t
        JOIN 
            `{state["dataset_id"]}.product_master` p
        ON 
            t.product_id = p.product_id
        GROUP BY 
            date, p.category_level_1
        ORDER BY 
            date ASC
        """
        
        sales_data = bq_connector.run_query(query)
        sales_data['date'] = pd.to_datetime(sales_data['date'])
        
        # Create total sales time series
        total_sales = sales_data.groupby('date').agg({
            'sales_amount': 'sum',
            'sales_count': 'sum',
            'quantity_sold': 'sum'
        }).reset_index()
        
        # Create category-specific time series
        categories = sales_data['category_level_1'].unique()
        category_data = {}
        
        for category in categories:
            category_sales = sales_data[sales_data['category_level_1'] == category]
            
            # Group by date if there are multiple entries per date
            if len(category_sales) > len(category_sales['date'].unique()):
                category_sales = category_sales.groupby('date').agg({
                    'sales_amount': 'sum',
                    'sales_count': 'sum',
                    'quantity_sold': 'sum',
                    'category_level_1': 'first'  # Keep category information
                }).reset_index()
            
            category_data[category] = category_sales
        
        # Check for data quality issues
        # 1. Missing dates
        date_range = pd.date_range(start=total_sales['date'].min(), end=total_sales['date'].max())
        missing_dates = set(date_range) - set(total_sales['date'])
        
        # 2. Fill missing dates with zeros or interpolated values
        if missing_dates:
            logger.info(f"Found {len(missing_dates)} missing dates in the time series. Filling with interpolation.")
            
            # Create a complete date range DataFrame
            full_date_range = pd.DataFrame({'date': date_range})
            
            # Merge with total sales and fill missing values
            total_sales = pd.merge(full_date_range, total_sales, on='date', how='left')
            total_sales = total_sales.sort_values('date')
            
            # Fill missing values with interpolation
            total_sales = total_sales.interpolate(method='linear')
            
            # Do the same for category data
            for category in category_data:
                cat_df = category_data[category]
                cat_df = pd.merge(full_date_range, cat_df, on='date', how='left')
                cat_df = cat_df.sort_values('date')
                cat_df['category_level_1'] = category  # Ensure category is filled
                
                # Fill missing values with interpolation
                numeric_cols = ['sales_amount', 'sales_count', 'quantity_sold']
                for col in numeric_cols:
                    if col in cat_df.columns:
                        cat_df[col] = cat_df[col].interpolate(method='linear').fillna(0)
                
                category_data[category] = cat_df
        
        # Check for outliers and handle them
        def handle_outliers(df, column, threshold=3):
            """Handle outliers in a time series column using z-score."""
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
            
            if outliers.sum() > 0:
                logger.info(f"Found {outliers.sum()} outliers in {column}. Replacing with median.")
                df.loc[outliers, column] = df[column].median()
            
            return df
        
        # Handle outliers in total sales
        for col in ['sales_amount', 'sales_count', 'quantity_sold']:
            total_sales = handle_outliers(total_sales, col)
        
        # Handle outliers in category data
        for category in category_data:
            for col in ['sales_amount', 'sales_count', 'quantity_sold']:
                if col in category_data[category].columns:
                    category_data[category] = handle_outliers(category_data[category], col)
        
        # Store prepared data in state
        state["time_series_data"] = {
            "total_sales": total_sales,
            "start_date": total_sales['date'].min().strftime('%Y-%m-%d'),
            "end_date": total_sales['date'].max().strftime('%Y-%m-%d'),
            "num_observations": len(total_sales),
            "categories": list(categories)
        }
        
        state["category_data"] = category_data
        
        # Set next agent
        state["current_agent"] = "data_preparer"
        state["next_agent"] = "model_selector"
        state["error"] = None
        
        logger.info("Data Preparer: Time series data preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Data Preparer: Error preparing time series data - {str(e)}")
        state["error"] = f"시계열 데이터 준비 중 오류 발생: {str(e)}"
        state["next_agent"] = END
    
    return state

def model_selector(state: ForecastState) -> ForecastState:
    """
    Select appropriate forecasting models for the data.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with selected models
    """
    logger.info("Model Selector: Starting model selection")
    
    try:
        # Get time series data from state
        total_sales = state["time_series_data"]["total_sales"]
        categories = state["time_series_data"]["categories"]
        category_data = state["category_data"]
        
        # Initialize model selection dictionary
        selected_models = {}
        
        # Function to evaluate different models on a time series
        def select_best_model(df, target_column='sales_amount'):
            """Select the best forecasting model for a given time series."""
            # Prepare data
            df = df.copy()
            df = df.sort_values('date')
            
            # Split into train and test sets (80/20)
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            if len(test_df) < 5:  # Not enough test data
                return "prophet", None  # Default to Prophet
            
            # Prepare Prophet data
            prophet_train = train_df[['date', target_column]].rename(columns={'date': 'ds', target_column: 'y'})
            
            # Prepare ARIMA data
            arima_train = train_df[target_column].values
            arima_test = test_df[target_column].values
            
            # Model evaluation metrics
            metrics = {}
            
            # Try Prophet
            try:
                prophet_model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative'
                )
                prophet_model.fit(prophet_train)
                
                # Make predictions
                future = prophet_model.make_future_dataframe(periods=len(test_df))
                forecast = prophet_model.predict(future)
                prophet_predictions = forecast.iloc[-len(test_df):]['yhat'].values
                
                # Calculate metrics
                prophet_rmse = np.sqrt(mean_squared_error(test_df[target_column].values, prophet_predictions))
                prophet_mae = mean_absolute_error(test_df[target_column].values, prophet_predictions)
                
                metrics['prophet'] = {
                    'rmse': prophet_rmse,
                    'mae': prophet_mae
                }
                
                logger.debug(f"Prophet RMSE: {prophet_rmse}, MAE: {prophet_mae}")
                
            except Exception as e:
                logger.warning(f"Error fitting Prophet model: {str(e)}")
                metrics['prophet'] = {'rmse': float('inf'), 'mae': float('inf')}
            
            # Try ARIMA
            try:
                # Find best ARIMA parameters
                best_aic = float('inf')
                best_order = None
                best_model = None
                
                # Try different ARIMA parameters
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                model = ARIMA(arima_train, order=(p, d, q))
                                model_fit = model.fit()
                                
                                if model_fit.aic < best_aic:
                                    best_aic = model_fit.aic
                                    best_order = (p, d, q)
                                    best_model = model_fit
                            except:
                                continue
                
                if best_model is not None:
                    # Make predictions
                    arima_predictions = best_model.forecast(steps=len(test_df))
                    
                    # Calculate metrics
                    arima_rmse = np.sqrt(mean_squared_error(arima_test, arima_predictions))
                    arima_mae = mean_absolute_error(arima_test, arima_predictions)
                    
                    metrics['arima'] = {
                        'rmse': arima_rmse,
                        'mae': arima_mae,
                        'order': best_order
                    }
                    
                    logger.debug(f"ARIMA RMSE: {arima_rmse}, MAE: {arima_mae}, Order: {best_order}")
                else:
                    metrics['arima'] = {'rmse': float('inf'), 'mae': float('inf')}
                    
            except Exception as e:
                logger.warning(f"Error fitting ARIMA model: {str(e)}")
                metrics['arima'] = {'rmse': float('inf'), 'mae': float('inf')}
            
            # Select the best model based on RMSE
            best_model = min(metrics, key=lambda x: metrics[x]['rmse'])
            
            return best_model, metrics
        
        # Select model for total sales
        best_model_total, metrics_total = select_best_model(total_sales)
        selected_models['total'] = {
            'model_type': best_model_total,
            'target_column': 'sales_amount',
            'metrics': metrics_total
        }
        
        # Select models for each category
        for category in categories:
            cat_df = category_data[category]
            best_model_cat, metrics_cat = select_best_model(cat_df)
            selected_models[category] = {
                'model_type': best_model_cat,
                'target_column': 'sales_amount',
                'metrics': metrics_cat
            }
        
        # Store selected models in state
        state["selected_models"] = selected_models
        
        # Set next agent
        state["current_agent"] = "model_selector"
        state["next_agent"] = "model_trainer"
        state["error"] = None
        
        logger.info(f"Model Selector: Selected {len(selected_models)} models")
        
    except Exception as e:
        logger.error(f"Model Selector: Error selecting models - {str(e)}")
        state["error"] = f"모델 선택 중 오류 발생: {str(e)}"
        state["next_agent"] = END
    
    return state

def model_trainer(state: ForecastState) -> ForecastState:
    """
    Train the selected forecasting models.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with trained models
    """
    logger.info("Model Trainer: Starting model training")
    
    try:
        # Get data from state
        total_sales = state["time_series_data"]["total_sales"]
        category_data = state["category_data"]
        selected_models = state["selected_models"]
        
        # Initialize trained models and metrics dictionaries
        trained_models = {}
        model_metrics = {}
        
        # Train model for total sales
        total_model_info = selected_models['total']
        target_column = total_model_info['target_column']
        
        if total_model_info['model_type'] == 'prophet':
            # Prepare Prophet data
            prophet_data = total_sales[['date', target_column]].rename(columns={'date': 'ds', target_column: 'y'})
            
            # Train Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model.fit(prophet_data)
            
            # Store trained model
            trained_models['total'] = {
                'model_type': 'prophet',
                'model': model,
                'data': prophet_data
            }
            
            # Calculate in-sample metrics
            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)
            
            # Calculate metrics
            y_true = prophet_data['y'].values
            y_pred = forecast['yhat'].values
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            model_metrics['total'] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
        elif total_model_info['model_type'] == 'arima':
            # Prepare ARIMA data
            arima_data = total_sales[target_column].values
            
            # Get best order from model selection
            if 'metrics' in total_model_info and 'order' in total_model_info['metrics']['arima']:
                order = total_model_info['metrics']['arima']['order']
            else:
                order = (1, 1, 1)  # Default order
            
            # Train ARIMA model
            model = ARIMA(arima_data, order=order)
            model_fit = model.fit()
            
            # Store trained model
            trained_models['total'] = {
                'model_type': 'arima',
                'model': model_fit,
                'data': arima_data,
                'order': order
            }
            
            # Calculate in-sample metrics
            y_true = arima_data
            y_pred = model_fit.fittedvalues
            
            rmse = np.sqrt(mean_squared_error(y_true[len(y_true)-len(y_pred):], y_pred))
            mae = mean_absolute_error(y_true[len(y_true)-len(y_pred):], y_pred)
            r2 = r2_score(y_true[len(y_true)-len(y_pred):], y_pred)
            
            model_metrics['total'] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        
        # Train models for each category
        for category, cat_model_info in selected_models.items():
            if category == 'total':
                continue
                
            cat_df = category_data[category]
            target_column = cat_model_info['target_column']
            
            if cat_model_info['model_type'] == 'prophet':
                # Prepare Prophet data
                prophet_data = cat_df[['date', target_column]].rename(columns={'date': 'ds', target_column: 'y'})
                
                # Train Prophet model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative'
                )
                model.fit(prophet_data)
                
                # Store trained model
                trained_models[category] = {
                    'model_type': 'prophet',
                    'model': model,
                    'data': prophet_data
                }
                
                # Calculate in-sample metrics
                future = model.make_future_dataframe(periods=0)
                forecast = model.predict(future)
                
                # Calculate metrics
                y_true = prophet_data['y'].values
                y_pred = forecast['yhat'].values
                
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                model_metrics[category] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
            elif cat_model_info['model_type'] == 'arima':
                # Prepare ARIMA data
                arima_data = cat_df[target_column].values
                
                # Get best order from model selection
                if 'metrics' in cat_model_info and 'order' in cat_model_info['metrics']['arima']:
                    order = cat_model_info['metrics']['arima']['order']
                else:
                    order = (1, 1, 1)  # Default order
                
                # Train ARIMA model
                model = ARIMA(arima_data, order=order)
                model_fit = model.fit()
                
                # Store trained model
                trained_models[category] = {
                    'model_type': 'arima',
                    'model': model_fit,
                    'data': arima_data,
                    'order': order
                }
                
                # Calculate in-sample metrics
                y_true = arima_data
                y_pred = model_fit.fittedvalues
                
                rmse = np.sqrt(mean_squared_error(y_true[len(y_true)-len(y_pred):], y_pred))
                mae = mean_absolute_error(y_true[len(y_true)-len(y_pred):], y_pred)
                r2 = r2_score(y_true[len(y_true)-len(y_pred):], y_pred)
                
                model_metrics[category] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
        
        # Store trained models and metrics in state
        state["trained_models"] = trained_models
        state["model_metrics"] = model_metrics
        
        # Set next agent
        state["current_agent"] = "model_trainer"
        state["next_agent"] = "forecaster"
        state["error"] = None
        
        logger.info(f"Model Trainer: Trained {len(trained_models)} models")
        
    except Exception as e:
        logger.error(f"Model Trainer: Error training models - {str(e)}")
        state["error"] = f"모델 학습 중 오류 발생: {str(e)}"
        state["next_agent"] = END
    
    return state

def forecaster(state: ForecastState) -> ForecastState:
    """
    Generate forecasts using the trained models.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with forecast results
    """
    logger.info("Forecaster: Starting forecast generation")
    
    try:
        # Get data from state
        trained_models = state["trained_models"]
        forecast_horizon_weeks = state["forecast_horizon_weeks"]
        forecast_horizon_days = forecast_horizon_weeks * 7  # Convert weeks to days
        
        # Initialize forecast results dictionaries
        total_forecast = {}
        category_forecasts = {}
        
        # Generate forecast for total sales
        total_model_info = trained_models['total']
        
        if total_model_info['model_type'] == 'prophet':
            model = total_model_info['model']
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=forecast_horizon_days)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Store forecast results
            total_forecast = {
                'forecast': forecast,
                'model': model,
                'model_type': 'prophet'
            }
            
        elif total_model_info['model_type'] == 'arima':
            model_fit = total_model_info['model']
            
            # Generate forecast
            forecast = model_fit.forecast(steps=forecast_horizon_days)
            
            # Create a DataFrame similar to Prophet's output for consistency
            last_date = pd.to_datetime(state["time_series_data"]["end_date"])
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon_days)
            
            forecast_df = pd.DataFrame({
                'ds': pd.concat([pd.Series(model_fit.model.data.dates), pd.Series(future_dates)]),
                'yhat': pd.concat([pd.Series(model_fit.fittedvalues), pd.Series(forecast)]),
                'yhat_lower': pd.concat([pd.Series(model_fit.fittedvalues * 0.9), pd.Series(forecast * 0.9)]),
                'yhat_upper': pd.concat([pd.Series(model_fit.fittedvalues * 1.1), pd.Series(forecast * 1.1)])
            })
            
            # Store forecast results
            total_forecast = {
                'forecast': forecast_df,
                'model': model_fit,
                'model_type': 'arima'
            }
        
        # Generate forecasts for each category
        for category, cat_model_info in trained_models.items():
            if category == 'total':
                continue
                
            if cat_model_info['model_type'] == 'prophet':
                model = cat_model_info['model']
                
                # Make future dataframe
                future = model.make_future_dataframe(periods=forecast_horizon_days)
                
                # Generate forecast
                forecast = model.predict(future)
                
                # Store forecast results
                category_forecasts[category] = {
                    'forecast': forecast,
                    'model': model,
                    'model_type': 'prophet'
                }
                
            elif cat_model_info['model_type'] == 'arima':
                model_fit = cat_model_info['model']
                
                # Generate forecast
                forecast = model_fit.forecast(steps=forecast_horizon_days)
                
                # Create a DataFrame similar to Prophet's output for consistency
                last_date = pd.to_datetime(state["time_series_data"]["end_date"])
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon_days)
                
                forecast_df = pd.DataFrame({
                    'ds': pd.concat([pd.Series(model_fit.model.data.dates), pd.Series(future_dates)]),
                    'yhat': pd.concat([pd.Series(model_fit.fittedvalues), pd.Series(forecast)]),
                    'yhat_lower': pd.concat([pd.Series(model_fit.fittedvalues * 0.9), pd.Series(forecast * 0.9)]),
                    'yhat_upper': pd.concat([pd.Series(model_fit.fittedvalues * 1.1), pd.Series(forecast * 1.1)])
                })
                
                # Store forecast results
                category_forecasts[category] = {
                    'forecast': forecast_df,
                    'model': model_fit,
                    'model_type': 'arima'
                }
        
        # Store forecast results in state
        state["total_forecast"] = total_forecast
        state["category_forecasts"] = category_forecasts
        
        # Set next agent
        state["current_agent"] = "forecaster"
        state["next_agent"] = "result_analyzer"
        state["error"] = None
        
        logger.info(f"Forecaster: Generated forecasts for total sales and {len(category_forecasts)} categories")
        
    except Exception as e:
        logger.error(f"Forecaster: Error generating forecasts - {str(e)}")
        state["error"] = f"예측 생성 중 오류 발생: {str(e)}"
        state["next_agent"] = END
    
    return state

def result_analyzer(state: ForecastState) -> ForecastState:
    """
    Analyze forecast results and calculate growth rates and seasonality.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with analysis results
    """
    logger.info("Result Analyzer: Starting forecast analysis")
    
    try:
        # Get forecast results from state
        total_forecast = state["total_forecast"]
        category_forecasts = state["category_forecasts"]
        
        # Initialize visualizations list
        visualizations = []
        
        # Analyze total sales forecast
        # 1. Create visualization
        if total_forecast['model_type'] == 'prophet':
            forecast_df = total_forecast['forecast']
            model = total_forecast['model']
            
            # Create time series plot
            fig = create_time_series_plot(
                forecast_df,
                x_column='ds',
                y_column='yhat',
                title='총 매출액 예측',
                xlabel='날짜',
                ylabel='매출액',
                use_plotly=True
            )
            
            # Add confidence intervals
            fig.add_trace({
                'x': forecast_df['ds'],
                'y': forecast_df['yhat_upper'],
                'mode': 'lines',
                'line': {'width': 0},
                'showlegend': False
            })
            
            fig.add_trace({
                'x': forecast_df['ds'],
                'y': forecast_df['yhat_lower'],
                'mode': 'lines',
                'line': {'width': 0},
                'fill': 'tonexty',
                'fillcolor': 'rgba(0, 100, 80, 0.2)',
                'name': '95% 신뢰 구간'
            })
            
            # Save visualization
            viz_path = save_plotly_fig(fig, "total_sales_forecast.png")
            visualizations.append({
                'path': viz_path,
                'title': '총 매출액 예측',
                'description': '향후 기간 동안의 총 매출액 예측 결과입니다.'
            })
            
            # 2. Analyze seasonality
            seasonality_components = {
                'weekly': pd.DataFrame({
                    'day': ['월', '화', '수', '목', '금', '토', '일'],
                    'effect': model.seasonalities['weekly'].effect
                }),
                'yearly': pd.DataFrame({
                    'day_of_year': range(1, 366),
                    'effect': model.seasonalities['yearly'].effect
                })
            }
            
            # Add month information to yearly seasonality
            seasonality_components['yearly']['month'] = pd.to_datetime(
                seasonality_components['yearly']['day_of_year'].apply(lambda x: f"2022-{x}"), 
                format="%Y-%j"
            ).dt.month
            
            # Calculate monthly effect
            monthly_effect = seasonality_components['yearly'].groupby('month')['effect'].mean().reset_index()
            month_names = ['1월', '2월', '3월', '4월', '5월', '6월', 
                         '7월', '8월', '9월', '10월', '11월', '12월']
            monthly_effect['month_name'] = monthly_effect['month'].apply(lambda x: month_names[x-1])
            
            seasonality_components['monthly'] = monthly_effect
            
        else:  # ARIMA model
            forecast_df = total_forecast['forecast']
            
            # Create time series plot
            fig = create_time_series_plot(
                forecast_df,
                x_column='ds',
                y_column='yhat',
                title='총 매출액 예측',
                xlabel='날짜',
                ylabel='매출액',
                use_plotly=True
            )
            
            # Add confidence intervals
            fig.add_trace({
                'x': forecast_df['ds'],
                'y': forecast_df['yhat_upper'],
                'mode': 'lines',
                'line': {'width': 0},
                'showlegend': False
            })
            
            fig.add_trace({
                'x': forecast_df['ds'],
                'y': forecast_df['yhat_lower'],
                'mode': 'lines',
                'line': {'width': 0},
                'fill': 'tonexty',
                'fillcolor': 'rgba(0, 100, 80, 0.2)',
                'name': '95% 신뢰 구간'
            })
            
            # Save visualization
            viz_path = save_plotly_fig(fig, "total_sales_forecast.png")
            visualizations.append({
                'path': viz_path,
                'title': '총 매출액 예측',
                'description': '향후 기간 동안의 총 매출액 예측 결과입니다.'
            })
            
            # For ARIMA, we don't have built-in seasonality components
            # We'll create a simple approximation
            
            # Get the time series data
            time_series_data = state["time_series_data"]["total_sales"]
            
            # Calculate weekly seasonality
            time_series_data['day_of_week'] = time_series_data['date'].dt.dayofweek
            weekly_effect = time_series_data.groupby('day_of_week')['sales_amount'].mean()
            weekly_effect = weekly_effect / weekly_effect.mean()  # Normalize
            
            weekly_seasonality = pd.DataFrame({
                'day': ['월', '화', '수', '목', '금', '토', '일'],
                'effect': weekly_effect.values
            })
            
            # Calculate monthly seasonality
            time_series_data['month'] = time_series_data['date'].dt.month
            monthly_effect = time_series_data.groupby('month')['sales_amount'].mean()
            monthly_effect = monthly_effect / monthly_effect.mean()  # Normalize
            
            monthly_seasonality = pd.DataFrame({
                'month': range(1, 13),
                'effect': [monthly_effect.get(i, 1.0) for i in range(1, 13)]
            })
            
            month_names = ['1월', '2월', '3월', '4월', '5월', '6월', 
                         '7월', '8월', '9월', '10월', '11월', '12월']
            monthly_seasonality['month_name'] = monthly_seasonality['month'].apply(lambda x: month_names[x-1])
            
            seasonality_components = {
                'weekly': weekly_seasonality,
                'monthly': monthly_seasonality
            }
        
        # 3. Calculate growth rates
        # Current date
        current_date = datetime.now().date()
        
        # Get forecast for future periods
        future_forecast = forecast_df[pd.to_datetime(forecast_df['ds']).dt.date > current_date]
        
        # Get historical data for past 30 days
        past_30d_data = forecast_df[
            (pd.to_datetime(forecast_df['ds']).dt.date <= current_date) &
            (pd.to_datetime(forecast_df['ds']).dt.date >= current_date - timedelta(days=30))
        ]
        
        # Calculate average values
        if len(past_30d_data) > 0:
            past_30d_avg = past_30d_data['yhat'].mean()
        else:
            past_30d_avg = forecast_df[pd.to_datetime(forecast_df['ds']).dt.date <= current_date]['yhat'].mean()
            
        future_avg = future_forecast['yhat'].mean()
        
        # Calculate growth rate
        total_growth_rate = ((future_avg / past_30d_avg) - 1) * 100 if past_30d_avg > 0 else 0
        
        # Store seasonality and growth rate
        state["seasonality"] = {
            'components': seasonality_components,
            'visualization_paths': []  # Will be filled with paths to seasonality visualizations
        }
        
        # Create visualizations for seasonality
        # Weekly seasonality
        weekly_df = seasonality_components['weekly']
        fig = create_time_series_plot(
            weekly_df,
            x_column='day',
            y_column='effect',
            title='요일별 판매 효과',
            xlabel='요일',
            ylabel='효과',
            use_plotly=True
        )
        viz_path = save_plotly_fig(fig, "weekly_seasonality.png")
        visualizations.append({
            'path': viz_path,
            'title': '요일별 판매 효과',
            'description': '요일에 따른 판매 패턴을 보여줍니다.'
        })
        state["seasonality"]["visualization_paths"].append(viz_path)
        
        # Monthly seasonality
        monthly_df = seasonality_components['monthly']
        fig = create_time_series_plot(
            monthly_df,
            x_column='month_name',
            y_column='effect',
            title='월별 판매 효과',
            xlabel='월',
            ylabel='효과',
            use_plotly=True
        )
        viz_path = save_plotly_fig(fig, "monthly_seasonality.png")
        visualizations.append({
            'path': viz_path,
            'title': '월별 판매 효과',
            'description': '월별 판매 패턴을 보여줍니다.'
        })
        state["seasonality"]["visualization_paths"].append(viz_path)
        
        # Analyze category forecasts and calculate growth rates
        category_growth = []
        
        for category, forecast_data in category_forecasts.items():
            forecast_df = forecast_data['forecast']
            
            # Create visualization
            fig = create_time_series_plot(
                forecast_df,
                x_column='ds',
                y_column='yhat',
                title=f'{category} 카테고리 매출액 예측',
                xlabel='날짜',
                ylabel='매출액',
                use_plotly=True
            )
            
            # Add confidence intervals
            fig.add_trace({
                'x': forecast_df['ds'],
                'y': forecast_df['yhat_upper'],
                'mode': 'lines',
                'line': {'width': 0},
                'showlegend': False
            })
            
            fig.add_trace({
                'x': forecast_df['ds'],
                'y': forecast_df['yhat_lower'],
                'mode': 'lines',
                'line': {'width': 0},
                'fill': 'tonexty',
                'fillcolor': 'rgba(0, 100, 80, 0.2)',
                'name': '95% 신뢰 구간'
            })
            
            # Save visualization
            viz_path = save_plotly_fig(fig, f"{category}_forecast.png")
            visualizations.append({
                'path': viz_path,
                'title': f'{category} 카테고리 매출액 예측',
                'description': f'{category} 카테고리의 향후 기간 동안의 매출액 예측 결과입니다.'
            })
            
            # Calculate growth rate
            future_forecast = forecast_df[pd.to_datetime(forecast_df['ds']).dt.date > current_date]
            
            past_30d_data = forecast_df[
                (pd.to_datetime(forecast_df['ds']).dt.date <= current_date) &
                (pd.to_datetime(forecast_df['ds']).dt.date >= current_date - timedelta(days=30))
            ]
            
            if len(past_30d_data) > 0:
                past_30d_avg = past_30d_data['yhat'].mean()
            else:
                past_30d_avg = forecast_df[pd.to_datetime(forecast_df['ds']).dt.date <= current_date]['yhat'].mean()
                
            future_avg = future_forecast['yhat'].mean()
            
            growth_rate = ((future_avg / past_30d_avg) - 1) * 100 if past_30d_avg > 0 else 0
            
            category_growth.append({
                'category': category,
                'past_30d_avg': past_30d_avg,
                'future_avg': future_avg,
                'growth_rate': growth_rate
            })
        
        # Create category growth rate visualization
        category_growth_df = pd.DataFrame(category_growth)
        
        if len(category_growth_df) > 0:
            fig = create_time_series_plot(
                category_growth_df,
                x_column='category',
                y_column='growth_rate',
                title='카테고리별 예상 성장률',
                xlabel='카테고리',
                ylabel='성장률 (%)',
                use_plotly=True
            )
            viz_path = save_plotly_fig(fig, "category_growth_rates.png")
            visualizations.append({
                'path': viz_path,
                'title': '카테고리별 예상 성장률',
                'description': '각 카테고리의 향후 예상 성장률을 보여줍니다.'
            })
        
        # Store growth rates in state
        state["growth_rates"] = {
            'total_growth_rate': total_growth_rate,
            'category_growth': category_growth_df,
            'visualization_path': viz_path if len(category_growth_df) > 0 else None
        }
        
        # Store visualizations in state
        state["visualizations"] = visualizations
        
        # Set next agent
        state["current_agent"] = "result_analyzer"
        state["next_agent"] = "report_generator"
        state["error"] = None
        
        logger.info("Result Analyzer: Forecast analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Result Analyzer: Error analyzing forecasts - {str(e)}")
        state["error"] = f"예측 분석 중 오류 발생: {str(e)}"
        state["next_agent"] = END
    
    return state

def report_generator(state: ForecastState) -> ForecastState:
    """
    Generate a PDF report with forecast results and insights.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with report path and insights
    """
    logger.info("Report Generator: Generating forecast report")
    
    try:
        from utils.pdf_generator import ReportPDF
        
        # Initialize insights list if not present
        if "insights" not in state or state["insights"] is None:
            state["insights"] = []
        
        # Generate insights using LLM if not already present
        if len(state["insights"]) == 0:
            llm = get_llm()
            
            # Prepare prompt with forecast results
            total_growth_rate = state["growth_rates"]["total_growth_rate"]
            category_growth = state["growth_rates"]["category_growth"]
            seasonality = state["seasonality"]["components"]
            
            # Convert DataFrames to string representations for the prompt
            category_growth_str = category_growth.to_string() if not category_growth.empty else "No category growth data available"
            weekly_seasonality_str = seasonality['weekly'].to_string()
            monthly_seasonality_str = seasonality['monthly'].to_string()
            
            prompt = f"""
            당신은 가전 리테일 업체의 수요 예측 전문가입니다. 다음 예측 결과를 검토하고 주요 인사이트를 도출해주세요.
            
            ## 전체 매출 예측
            - 향후 {state["forecast_horizon_weeks"]}주 동안의 예상 총 매출 성장률: {total_growth_rate:.2f}%
            
            ## 카테고리별 성장률 예측
            {category_growth_str}
            
            ## 계절성 분석
            - 요일별 판매 효과:
            {weekly_seasonality_str}
            
            - 월별 판매 효과:
            {monthly_seasonality_str}
            
            위 데이터를 종합적으로 분석하여 다음을 작성해주세요:
            1. 향후 {state["forecast_horizon_weeks"]}주 동안의 주요 매출 트렌드 예측
            2. 카테고리별 성과 예측 및 성장 기회
            3. 계절성 요소를 활용한 판매 전략 제안
            4. 수요 예측에 따른 재고 관리 및 마케팅 제안
            5. 예측 결과에 따른 리스크 요소 및 대응 방안
            
            각 항목은 데이터에 기반한 구체적인 인사이트와 함께 작성해주세요.
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
        report.set_title("가전 리테일 CDP 수요 예측 리포트")
        report.add_date()
        
        # Add executive summary
        summary = "\n".join(state["insights"][:3])  # Use first 3 insights as summary
        report.add_executive_summary(summary)
        
        # Add total forecast section
        report.add_section("전체 매출 예측")
        report.add_text(f"향후 {state['forecast_horizon_weeks']}주간의 총 매출액 예측 결과입니다. "
                      f"예상 성장률은 {state['growth_rates']['total_growth_rate']:.2f}%입니다.")
        
        # Add total forecast visualization
        total_viz = next((v for v in state["visualizations"] if "총 매출액" in v["title"]), None)
        if total_viz:
            report.add_image(total_viz["path"], width=160, caption=total_viz["title"])
            report.add_text(total_viz["description"])
        
        # Add category forecasts section
        report.add_section("카테고리별 예측")
        report.add_text("주요 제품 카테고리별 매출액 예측 결과입니다.")
        
        # Add category forecast visualizations
        for viz in state["visualizations"]:
            if "카테고리 매출액" in viz["title"]:
                report.add_image(viz["path"], width=160, caption=viz["title"])
                report.add_text(viz["description"])
        
        # Add growth rate analysis section
        report.add_section("성장률 분석")
        report.add_text(f"향후 {state['forecast_horizon_weeks']}주간 예상되는 카테고리별 성장률입니다.")
        
        # Add growth rate visualization
        growth_viz = next((v for v in state["visualizations"] if "성장률" in v["title"]), None)
        if growth_viz:
            report.add_image(growth_viz["path"], width=160, caption=growth_viz["title"])
            report.add_text(growth_viz["description"])
        
        # Add seasonality analysis section
        report.add_section("계절성 분석")
        report.add_text("요일별, 월별 판매 패턴의 계절성을 분석한 결과입니다.")
        
        # Add seasonality visualizations
        for viz_path in state["seasonality"]["visualization_paths"]:
            viz = next((v for v in state["visualizations"] if v["path"] == viz_path), None)
            if viz:
                report.add_image(viz["path"], width=160, caption=viz["title"])
                report.add_text(viz["description"])
        
        # Add insights section
        report.add_section("예측 인사이트")
        for insight in state["insights"]:
            report.add_text("• " + insight)
        
        # Add recommendations section
        report.add_section("권장 사항")
        report.add_text("수요 예측 결과를 활용하여 다음과 같은 방안을 추진하는 것이 권장됩니다:")
        
        # Extract recommendations from insights
        recommendations = [insight for insight in state["insights"] if "제안" in insight or "권장" in insight or "전략" in insight]
        if recommendations:
            for rec in recommendations:
                report.add_text("• " + rec)
        else:
            report.add_text("• 성장이 예상되는 카테고리에 대한 재고 및 판촉 강화")
            report.add_text("• 계절성이 강한 제품에 대한 시즌별 프로모션 계획 수립")
            report.add_text("• 요일별 수요 패턴에 맞춘 인력 배치 및 마케팅 타이밍 조절")
            report.add_text("• 수요가 감소할 것으로 예상되는 카테고리의 재고 관리 최적화")
        
        # Generate and save the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(config.REPORTS_DIR, f"demand_forecast_{timestamp}.pdf")
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
        state["error"] = f"리포트 생성 중 오류 발생: {str(e)}"
        state["next_agent"] = END
    
    return state

# Define the routing logic for the graph
def router(state: ForecastState) -> str:
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
def build_forecast_graph() -> StateGraph:
    """
    Build and return the demand forecast workflow graph.
    
    Returns:
        StateGraph for demand forecasting
    """
    # Create a new graph
    workflow = StateGraph(ForecastState)
    
    # Add nodes for each sub-agent
    workflow.add_node("data_preparer", data_preparer)
    workflow.add_node("model_selector", model_selector)
    workflow.add_node("model_trainer", model_trainer)
    workflow.add_node("forecaster", forecaster)
    workflow.add_node("result_analyzer", result_analyzer)
    workflow.add_node("report_generator", report_generator)
    
    # Set the entry point
    workflow.set_entry_point("data_preparer")
    
    # Add conditional edges based on the router function
    workflow.add_conditional_edges(
        "data_preparer",
        router,
        {
            "model_selector": "model_selector",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "model_selector",
        router,
        {
            "model_trainer": "model_trainer",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "model_trainer",
        router,
        {
            "forecaster": "forecaster",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "forecaster",
        router,
        {
            "result_analyzer": "result_analyzer",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "result_analyzer",
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

# Create a ForecastGraph class for easy usage
class ForecastGraph:
    """Class for running the demand forecast workflow."""
    
    def __init__(self):
        """Initialize the demand forecast graph."""
        self.graph = build_forecast_graph()
    
    def run(
        self,
        project_id: str = None,
        dataset_id: str = None,
        forecast_horizon_weeks: int = None
    ) -> Dict[str, Any]:
        """
        Run the demand forecast workflow.
        
        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
            forecast_horizon_weeks: Number of weeks to forecast
            
        Returns:
            Dictionary with forecast results and report path
        """
        # Set default values if not provided
        project_id = project_id or config.GCP_PROJECT_ID
        dataset_id = dataset_id or config.BQ_DATASET_ID
        forecast_horizon_weeks = forecast_horizon_weeks or config.FORECAST_HORIZON_WEEKS
        
        # Initialize the state
        initial_state = ForecastState(
            current_agent="data_preparer",
            next_agent=None,
            error=None,
            done=False,
            project_id=project_id,
            dataset_id=dataset_id,
            forecast_horizon_weeks=forecast_horizon_weeks,
            raw_data=None,
            time_series_data=None,
            category_data=None,
            selected_models=None,
            trained_models=None,
            model_metrics=None,
            total_forecast=None,
            category_forecasts=None,
            growth_rates=None,
            seasonality=None,
            visualizations=[],
            insights=[],
            report_path=None
        )
        
        # Run the graph
        logger.info(f"Starting demand forecast for project {project_id}, dataset {dataset_id}")
        result = self.graph.invoke(initial_state)
        
        # Check for errors
        if result.get("error"):
            logger.error(f"Demand forecast failed: {result['error']}")
            raise Exception(result["error"])
        
        # Return the result
        logger.info("Demand forecast completed successfully")
        return {
            "report_path": result.get("report_path"),
            "insights": result.get("insights", []),
            "visualizations": result.get("visualizations", []),
            "growth_rates": result.get("growth_rates", {}),
            "seasonality": result.get("seasonality", {})
        }

# For testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the graph
    forecast_graph = ForecastGraph()
    result = forecast_graph.run()
    
    print(f"Report generated at: {result['report_path']}")
    print(f"Generated {len(result['visualizations'])} visualizations")
    print(f"Found {len(result['insights'])} insights")
