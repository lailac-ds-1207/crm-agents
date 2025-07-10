"""
Trend Analysis Agent for the CRM-Agent system.
This module defines the main class and interface for the trend analysis functionality.
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
from .graph import TrendAnalysisGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrendAnalysisAgent:
    """
    Agent for analyzing trends in CDP data.
    
    This agent orchestrates the process of collecting data, performing various
    analyses, creating visualizations, and generating a comprehensive report.
    """
    
    def __init__(
        self,
        project_id: str = None,
        dataset_id: str = None,
        timeframe_days: int = None,
        bq_connector: BigQueryConnector = None
    ):
        """
        Initialize the Trend Analysis Agent.
        
        Args:
            project_id: Google Cloud project ID (if None, uses the value from config)
            dataset_id: BigQuery dataset ID (if None, uses the value from config)
            timeframe_days: Number of days to analyze (if None, uses the value from config)
            bq_connector: Optional existing BigQuery connector to reuse
        """
        self.project_id = project_id or config.GCP_PROJECT_ID
        self.dataset_id = dataset_id or config.BQ_DATASET_ID
        self.timeframe_days = timeframe_days or config.TREND_ANALYSIS_TIMEFRAME_DAYS
        
        # Initialize BigQuery connector if not provided
        self.bq_connector = bq_connector or BigQueryConnector(project_id=self.project_id)
        
        # Initialize the workflow graph
        self.graph = TrendAnalysisGraph()
        
        # Initialize results storage
        self.results = None
        self.report_path = None
        self.insights = []
        self.visualizations = []
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_TEMPERATURE,
            top_p=config.GEMINI_TOP_P,
            top_k=config.GEMINI_TOP_K,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
            google_api_key=config.GEMINI_API_KEY,
        )
        
        logger.info(f"Trend Analysis Agent initialized for dataset {self.dataset_id} "
                   f"with {self.timeframe_days} days timeframe")
    
    def run(self) -> str:
        """
        Run the trend analysis workflow.
        
        Returns:
            Path to the generated report
        """
        logger.info("Starting trend analysis workflow")
        
        try:
            # Run the workflow graph
            self.results = self.graph.run(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                timeframe_days=self.timeframe_days
            )
            
            # Store results
            self.report_path = self.results.get("report_path")
            self.insights = self.results.get("insights", [])
            self.visualizations = self.results.get("visualizations", [])
            
            logger.info(f"Trend analysis completed successfully. Report: {self.report_path}")
            return self.report_path
            
        except Exception as e:
            logger.error(f"Error running trend analysis: {str(e)}")
            raise
    
    def get_insights(self) -> List[str]:
        """
        Get the insights generated from the analysis.
        
        Returns:
            List of insight strings
        """
        return self.insights
    
    def get_visualizations(self) -> List[Dict[str, Any]]:
        """
        Get the visualizations generated from the analysis.
        
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
    
    def generate_custom_insight(self, question: str) -> str:
        """
        Generate a custom insight based on a specific question.
        
        Args:
            question: Question to analyze
            
        Returns:
            Generated insight text
        """
        if not self.results:
            raise ValueError("Analysis has not been run yet. Call run() first.")
        
        # Prepare prompt with analysis results and the question
        prompt = f"""
        당신은 가전 리테일 업체의 데이터 분석가입니다. 다음 분석 결과를 바탕으로 질문에 답변해주세요.
        
        ## 주요 인사이트
        {chr(10).join([f"- {insight}" for insight in self.insights])}
        
        ## 질문
        {question}
        
        위 데이터와 인사이트를 기반으로 질문에 대한 답변을 작성해주세요. 
        데이터에 기반한 구체적인 인사이트와 함께 작성하고, 필요하다면 추가 분석이 필요한 부분을 언급해주세요.
        """
        
        # Get insight from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def summarize_trends(self, max_length: int = 500) -> str:
        """
        Generate a concise summary of the key trends.
        
        Args:
            max_length: Maximum length of the summary in characters
            
        Returns:
            Summary text
        """
        if not self.insights:
            raise ValueError("Analysis has not been run yet. Call run() first.")
        
        # Prepare prompt for summarization
        insights_text = "\n".join([f"- {insight}" for insight in self.insights])
        
        prompt = f"""
        다음은 가전 리테일 업체의 트렌드 분석에서 도출된 인사이트입니다:
        
        {insights_text}
        
        위 인사이트를 바탕으로 {max_length}자 이내의 간결한 요약을 작성해주세요.
        가장 중요한 트렌드와 비즈니스 기회를 중심으로 작성하세요.
        """
        
        # Get summary from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        summary = response.content
        
        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
            
        return summary
    
    def get_category_recommendations(self, category: str) -> List[str]:
        """
        Get specific recommendations for a product category.
        
        Args:
            category: Product category to get recommendations for
            
        Returns:
            List of recommendation strings
        """
        if not self.results:
            raise ValueError("Analysis has not been run yet. Call run() first.")
        
        # Prepare prompt for category-specific recommendations
        prompt = f"""
        당신은 가전 리테일 업체의 데이터 분석가입니다. 다음 분석 인사이트를 바탕으로
        '{category}' 카테고리에 대한 구체적인 추천 사항을 3가지 작성해주세요.
        
        ## 주요 인사이트
        {chr(10).join([f"- {insight}" for insight in self.insights])}
        
        위 인사이트를 기반으로 '{category}' 카테고리의 매출 증대, 고객 경험 개선, 마케팅 전략 등에 관한
        구체적인 추천 사항을 작성해주세요. 각 추천은 데이터에 기반한 근거와 함께 제시해주세요.
        """
        
        # Get recommendations from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        recommendations_text = response.content
        
        # Parse recommendations into list
        lines = recommendations_text.split('\n')
        recommendations = []
        current_rec = ""
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('1.') or line.startswith('2.') or 
                      line.startswith('3.') or line.startswith('- ')):
                if current_rec:
                    recommendations.append(current_rec)
                current_rec = line
            elif line and current_rec:
                current_rec += " " + line
        
        if current_rec:
            recommendations.append(current_rec)
        
        return recommendations if recommendations else [
            f"1. {category} 카테고리에 대한 맞춤형 프로모션 전략 개발",
            f"2. {category} 제품의 온라인-오프라인 연계 판매 강화",
            f"3. {category} 관련 고객 세그먼트 타겟팅 최적화"
        ]
    
    def compare_timeframes(self, previous_days: int = None) -> Dict[str, Any]:
        """
        Compare current analysis with a previous timeframe.
        
        Args:
            previous_days: Number of days for previous timeframe (if None, uses same as current)
            
        Returns:
            Dictionary with comparison results
        """
        # This is a placeholder for the comparison functionality
        # In a real implementation, this would run another analysis for the previous timeframe
        # and compare the results
        
        logger.info("Timeframe comparison functionality is not fully implemented")
        
        return {
            "status": "not_implemented",
            "message": "Timeframe comparison is not fully implemented yet"
        }
    
    @classmethod
    def from_existing_data(cls, data_path: str):
        """
        Create an agent instance from existing analysis data.
        
        Args:
            data_path: Path to saved analysis data
            
        Returns:
            TrendAnalysisAgent instance with loaded data
        """
        # This is a placeholder for loading from existing data
        # In a real implementation, this would load analysis results from a saved file
        
        logger.info(f"Loading from existing data at {data_path} is not fully implemented")
        
        agent = cls()
        # Here we would load data from the path and populate the agent's attributes
        
        return agent

# For direct execution
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run the agent
    agent = TrendAnalysisAgent()
    report_path = agent.run()
    
    print(f"Report generated at: {report_path}")
    print(f"Generated {len(agent.get_visualizations())} visualizations")
    print(f"Found {len(agent.get_insights())} insights")
