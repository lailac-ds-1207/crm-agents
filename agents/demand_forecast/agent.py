"""
Demand Forecast Agent for the CRM-Agent system.
This module defines the main class and interface for the demand forecasting functionality.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

import config
from utils.bigquery import BigQueryConnector
from .graph import ForecastGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemandForecastAgent:
    """
    Agent for forecasting demand in CDP data.
    
    This agent orchestrates the process of preparing time series data,
    selecting and training forecasting models, generating predictions,
    analyzing results, and creating a comprehensive report.
    """
    
    def __init__(
        self,
        project_id: str = None,
        dataset_id: str = None,
        forecast_horizon_weeks: int = None,
        bq_connector: BigQueryConnector = None
    ):
        """
        Initialize the Demand Forecast Agent.
        
        Args:
            project_id: Google Cloud project ID (if None, uses the value from config)
            dataset_id: BigQuery dataset ID (if None, uses the value from config)
            forecast_horizon_weeks: Number of weeks to forecast (if None, uses the value from config)
            bq_connector: Optional existing BigQuery connector to reuse
        """
        self.project_id = project_id or config.GCP_PROJECT_ID
        self.dataset_id = dataset_id or config.BQ_DATASET_ID
        self.forecast_horizon_weeks = forecast_horizon_weeks or config.FORECAST_HORIZON_WEEKS
        
        # Initialize BigQuery connector if not provided
        self.bq_connector = bq_connector or BigQueryConnector(project_id=self.project_id)
        
        # Initialize the workflow graph
        self.graph = ForecastGraph()
        
        # Initialize results storage
        self.results = None
        self.report_path = None
        self.insights = []
        self.visualizations = []
        self.growth_rates = None
        self.seasonality = None
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_TEMPERATURE,
            top_p=config.GEMINI_TOP_P,
            top_k=config.GEMINI_TOP_K,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
            google_api_key=config.GEMINI_API_KEY,
        )
        
        logger.info(f"Demand Forecast Agent initialized for dataset {self.dataset_id} "
                   f"with {self.forecast_horizon_weeks} weeks forecast horizon")
    
    def run(self) -> str:
        """
        Run the demand forecast workflow.
        
        Returns:
            Path to the generated report
        """
        logger.info("Starting demand forecast workflow")
        
        try:
            # Run the workflow graph
            self.results = self.graph.run(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                forecast_horizon_weeks=self.forecast_horizon_weeks
            )
            
            # Store results
            self.report_path = self.results.get("report_path")
            self.insights = self.results.get("insights", [])
            self.visualizations = self.results.get("visualizations", [])
            self.growth_rates = self.results.get("growth_rates", {})
            self.seasonality = self.results.get("seasonality", {})
            
            logger.info(f"Demand forecast completed successfully. Report: {self.report_path}")
            return self.report_path
            
        except Exception as e:
            logger.error(f"Error running demand forecast: {str(e)}")
            raise
    
    def get_insights(self) -> List[str]:
        """
        Get the insights generated from the forecast.
        
        Returns:
            List of insight strings
        """
        return self.insights
    
    def get_visualizations(self) -> List[Dict[str, Any]]:
        """
        Get the visualizations generated from the forecast.
        
        Returns:
            List of visualization dictionaries with paths and metadata
        """
        return self.visualizations
    
    def get_report_path(self) -> str:
        """
        Get the path to the generated report.
        
        Returns:
            Path to the PDF report
        """
        return self.report_path
    
    def get_growth_rates(self) -> Dict[str, Any]:
        """
        Get the calculated growth rates.
        
        Returns:
            Dictionary with growth rate information
        """
        return self.growth_rates
    
    def get_seasonality(self) -> Dict[str, Any]:
        """
        Get the seasonality components.
        
        Returns:
            Dictionary with seasonality information
        """
        return self.seasonality
    
    def generate_custom_insight(self, question: str) -> str:
        """
        Generate a custom insight based on a specific question about the forecast.
        
        Args:
            question: Question to analyze
            
        Returns:
            Generated insight text
        """
        if not self.results:
            raise ValueError("Forecast has not been run yet. Call run() first.")
        
        # Prepare prompt with forecast results and the question
        prompt = f"""
        당신은 가전 리테일 업체의 수요 예측 전문가입니다. 다음 예측 결과를 바탕으로 질문에 답변해주세요.
        
        ## 주요 인사이트
        {chr(10).join([f"- {insight}" for insight in self.insights])}
        
        ## 성장률 정보
        - 전체 성장률: {self.growth_rates.get('total_growth_rate', 'N/A')}%
        
        ## 질문
        {question}
        
        위 데이터와 인사이트를 기반으로 질문에 대한 답변을 작성해주세요. 
        데이터에 기반한 구체적인 인사이트와 함께 작성하고, 필요하다면 추가 분석이 필요한 부분을 언급해주세요.
        """
        
        # Get insight from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def summarize_forecast(self, max_length: int = 500) -> str:
        """
        Generate a concise summary of the key forecast results.
        
        Args:
            max_length: Maximum length of the summary in characters
            
        Returns:
            Summary text
        """
        if not self.insights:
            raise ValueError("Forecast has not been run yet. Call run() first.")
        
        # Prepare prompt for summarization
        insights_text = "\n".join([f"- {insight}" for insight in self.insights])
        
        prompt = f"""
        다음은 가전 리테일 업체의 수요 예측에서 도출된 인사이트입니다:
        
        {insights_text}
        
        위 인사이트를 바탕으로 {max_length}자 이내의 간결한 요약을 작성해주세요.
        가장 중요한 예측 결과와 비즈니스 기회를 중심으로 작성하세요.
        """
        
        # Get summary from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        summary = response.content
        
        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
            
        return summary
    
    def get_category_forecast(self, category: str) -> Dict[str, Any]:
        """
        Get forecast details for a specific product category.
        
        Args:
            category: Product category to get forecast for
            
        Returns:
            Dictionary with category forecast information
        """
        if not self.results:
            raise ValueError("Forecast has not been run yet. Call run() first.")
        
        # Find growth rate for the category
        category_growth = None
        if self.growth_rates and 'category_growth' in self.growth_rates:
            category_row = self.growth_rates['category_growth'][self.growth_rates['category_growth']['category'] == category]
            if not category_row.empty:
                category_growth = category_row.iloc[0].to_dict()
        
        # Find visualization for the category
        category_viz = next((v for v in self.visualizations if category in v.get('title', '')), None)
        
        # Get category-specific insights using LLM
        prompt = f"""
        당신은 가전 리테일 업체의 수요 예측 전문가입니다. 다음 예측 결과를 바탕으로 
        '{category}' 카테고리에 대한 구체적인 예측 인사이트를 3가지 작성해주세요.
        
        ## 주요 인사이트
        {chr(10).join([f"- {insight}" for insight in self.insights])}
        
        ## 카테고리 성장률
        {category_growth}
        
        위 정보를 기반으로 '{category}' 카테고리의 수요 예측, 성장 기회, 계절성 등에 관한
        구체적인 인사이트를 작성해주세요. 각 인사이트는 데이터에 기반한 근거와 함께 제시해주세요.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        category_insights = response.content
        
        return {
            'category': category,
            'growth_rate': category_growth.get('growth_rate') if category_growth else None,
            'past_30d_avg': category_growth.get('past_30d_avg') if category_growth else None,
            'future_avg': category_growth.get('future_avg') if category_growth else None,
            'visualization': category_viz['path'] if category_viz else None,
            'insights': category_insights
        }
    
    def adjust_forecast_parameters(self, horizon_weeks: int = None) -> Dict[str, Any]:
        """
        Adjust forecast parameters and re-run with new settings.
        
        Args:
            horizon_weeks: New forecast horizon in weeks
            
        Returns:
            Dictionary with adjusted forecast results
        """
        if horizon_weeks:
            self.forecast_horizon_weeks = horizon_weeks
            
        logger.info(f"Re-running forecast with adjusted parameters: horizon={self.forecast_horizon_weeks} weeks")
        
        # Re-run the forecast
        self.run()
        
        return {
            'report_path': self.report_path,
            'insights': self.insights[:3],  # Return just the top 3 insights for brevity
            'horizon_weeks': self.forecast_horizon_weeks
        }
    
    @classmethod
    def from_existing_data(cls, data_path: str):
        """
        Create an agent instance from existing forecast data.
        
        Args:
            data_path: Path to saved forecast data
            
        Returns:
            DemandForecastAgent instance with loaded data
        """
        # This is a placeholder for loading from existing data
        # In a real implementation, this would load forecast results from a saved file
        
        logger.info(f"Loading from existing data at {data_path} is not fully implemented")
        
        agent = cls()
        # Here we would load data from the path and populate the agent's attributes
        
        return agent

# For direct execution
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run the agent
    agent = DemandForecastAgent()
    report_path = agent.run()
    
    print(f"Report generated at: {report_path}")
    print(f"Generated {len(agent.get_visualizations())} visualizations")
    print(f"Found {len(agent.get_insights())} insights")
    
    # Print growth rates
    growth_rates = agent.get_growth_rates()
    if growth_rates and 'total_growth_rate' in growth_rates:
        print(f"Total growth rate: {growth_rates['total_growth_rate']:.2f}%")
