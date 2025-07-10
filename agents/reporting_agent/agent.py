"""
Reporting Agent for the CRM-Agent system.
This module defines the main class and interface for the reporting functionality.
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
from utils.pdf_generator import ReportPDF
from agents.trend_analysis.agent import TrendAnalysisAgent
from agents.demand_forecast.agent import DemandForecastAgent
from agents.decision_agent.agent import DecisionAgent
from agents.segmentation_agent.agent import SegmentationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReportingAgent:
    """
    Agent for generating comprehensive reports from multiple agent outputs.
    
    This agent orchestrates the process of collecting reports from other agents,
    extracting key insights, selecting visualizations, generating executive
    summaries, and compiling everything into professional formatted reports.
    """
    
    def __init__(
        self,
        project_id: str = None,
        dataset_id: str = None,
        report_type: str = "comprehensive",
        trend_report_path: str = None,
        forecast_report_path: str = None,
        decision_report_path: str = None,
        segmentation_report_path: str = None,
        bq_connector: BigQueryConnector = None
    ):
        """
        Initialize the Reporting Agent.
        
        Args:
            project_id: Google Cloud project ID (if None, uses the value from config)
            dataset_id: BigQuery dataset ID (if None, uses the value from config)
            report_type: Type of report to generate ("comprehensive", "executive", "marketing", "operations")
            trend_report_path: Path to trend analysis report (if None, will run trend analysis)
            forecast_report_path: Path to demand forecast report (if None, will run demand forecast)
            decision_report_path: Path to decision report (if None, will run decision agent)
            segmentation_report_path: Path to segmentation report (if None, will run segmentation agent)
            bq_connector: Optional existing BigQuery connector to reuse
        """
        self.project_id = project_id or config.GCP_PROJECT_ID
        self.dataset_id = dataset_id or config.BQ_DATASET_ID
        self.report_type = report_type
        
        # Store report paths
        self.report_paths = {
            "trend": trend_report_path,
            "forecast": forecast_report_path,
            "decision": decision_report_path,
            "segmentation": segmentation_report_path
        }
        
        # Initialize BigQuery connector if not provided
        self.bq_connector = bq_connector or BigQueryConnector(project_id=self.project_id)
        
        # Initialize agent states
        self.collected_reports = {}
        self.combined_insights = []
        self.selected_visualizations = []
        self.executive_summary = ""
        self.report_path = None
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_TEMPERATURE,
            top_p=config.GEMINI_TOP_P,
            top_k=config.GEMINI_TOP_K,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
            google_api_key=config.GEMINI_API_KEY,
        )
        
        logger.info(f"Reporting Agent initialized for {report_type} report")
    
    def run(self) -> str:
        """
        Run the reporting workflow.
        
        Returns:
            Path to the generated report
        """
        logger.info(f"Starting {self.report_type} reporting workflow")
        
        try:
            # Step 1: Collect reports from other agents
            self._collect_reports()
            
            # Step 2: Extract and combine insights
            self._extract_insights()
            
            # Step 3: Select visualizations
            self._select_visualizations()
            
            # Step 4: Generate executive summary
            self._generate_executive_summary()
            
            # Step 5: Compile final report
            self._compile_report()
            
            logger.info(f"Reporting completed successfully. Report: {self.report_path}")
            return self.report_path
            
        except Exception as e:
            logger.error(f"Error running reporting process: {str(e)}")
            raise
    
    def _collect_reports(self) -> None:
        """Collect reports from other agents."""
        logger.info("Collecting reports from other agents")
        
        # Run trend analysis if report path not provided
        if not self.report_paths["trend"]:
            logger.info("Running trend analysis")
            trend_agent = TrendAnalysisAgent(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                bq_connector=self.bq_connector
            )
            self.report_paths["trend"] = trend_agent.run()
            self.collected_reports["trend"] = {
                "path": self.report_paths["trend"],
                "insights": trend_agent.get_insights(),
                "visualizations": trend_agent.get_visualizations()
            }
        else:
            # Extract insights from existing trend report
            logger.info(f"Using existing trend report: {self.report_paths['trend']}")
            self.collected_reports["trend"] = {
                "path": self.report_paths["trend"],
                "insights": self._extract_insights_from_report(self.report_paths["trend"], "trend"),
                "visualizations": self._extract_visualizations_from_report(self.report_paths["trend"])
            }
        
        # Run demand forecast if report path not provided
        if not self.report_paths["forecast"]:
            logger.info("Running demand forecast")
            forecast_agent = DemandForecastAgent(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                bq_connector=self.bq_connector
            )
            self.report_paths["forecast"] = forecast_agent.run()
            self.collected_reports["forecast"] = {
                "path": self.report_paths["forecast"],
                "insights": forecast_agent.get_insights(),
                "visualizations": forecast_agent.get_visualizations(),
                "growth_rates": forecast_agent.get_growth_rates(),
                "seasonality": forecast_agent.get_seasonality()
            }
        else:
            # Extract insights from existing forecast report
            logger.info(f"Using existing forecast report: {self.report_paths['forecast']}")
            self.collected_reports["forecast"] = {
                "path": self.report_paths["forecast"],
                "insights": self._extract_insights_from_report(self.report_paths["forecast"], "forecast"),
                "visualizations": self._extract_visualizations_from_report(self.report_paths["forecast"])
            }
        
        # Run decision agent if report path not provided and report type requires it
        if (not self.report_paths["decision"] and 
            (self.report_type in ["comprehensive", "executive", "marketing"])):
            logger.info("Running decision agent")
            decision_agent = DecisionAgent(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                trend_report_path=self.report_paths["trend"],
                forecast_report_path=self.report_paths["forecast"],
                bq_connector=self.bq_connector
            )
            self.report_paths["decision"] = decision_agent.run()
            self.collected_reports["decision"] = {
                "path": self.report_paths["decision"],
                "insights": decision_agent.get_strategic_advice(),
                "visualizations": [],  # Decision agent might not have visualizations
                "campaigns": decision_agent.get_selected_campaigns(),
                "resource_allocation": decision_agent.get_resource_allocation()
            }
        elif self.report_paths["decision"]:
            # Extract insights from existing decision report
            logger.info(f"Using existing decision report: {self.report_paths['decision']}")
            self.collected_reports["decision"] = {
                "path": self.report_paths["decision"],
                "insights": self._extract_insights_from_report(self.report_paths["decision"], "decision"),
                "visualizations": self._extract_visualizations_from_report(self.report_paths["decision"])
            }
        
        # Run segmentation agent if report path not provided and report type requires it
        if (not self.report_paths["segmentation"] and 
            (self.report_type in ["comprehensive", "marketing"])):
            logger.info("Running segmentation agent")
            segmentation_agent = SegmentationAgent(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                bq_connector=self.bq_connector
            )
            self.report_paths["segmentation"] = segmentation_agent.run()
            self.collected_reports["segmentation"] = {
                "path": self.report_paths["segmentation"],
                "insights": self._extract_insights_from_report(self.report_paths["segmentation"], "segmentation"),
                "visualizations": segmentation_agent.get_visualizations(),
                "segments": {k: v for k, v in segmentation_agent.get_segments().items()}
            }
        elif self.report_paths["segmentation"]:
            # Extract insights from existing segmentation report
            logger.info(f"Using existing segmentation report: {self.report_paths['segmentation']}")
            self.collected_reports["segmentation"] = {
                "path": self.report_paths["segmentation"],
                "insights": self._extract_insights_from_report(self.report_paths["segmentation"], "segmentation"),
                "visualizations": self._extract_visualizations_from_report(self.report_paths["segmentation"])
            }
        
        logger.info(f"Collected reports from {len(self.collected_reports)} agents")
    
    def _extract_insights_from_report(self, report_path: str, report_type: str) -> List[str]:
        """Extract insights from a PDF report."""
        # In a real implementation, this would use PDF text extraction
        # For now, we'll use a placeholder approach with the LLM
        
        logger.info(f"Extracting insights from {report_type} report: {report_path}")
        
        # Use LLM to generate insights based on the report type
        if report_type == "trend":
            prompt = """
            가전 리테일 업체의 트렌드 분석 리포트에서 추출한 주요 인사이트를 5가지 작성해주세요.
            각 인사이트는 구체적이고 실행 가능해야 합니다.
            """
        elif report_type == "forecast":
            prompt = """
            가전 리테일 업체의 수요 예측 리포트에서 추출한 주요 인사이트를 5가지 작성해주세요.
            각 인사이트는 구체적이고 실행 가능해야 합니다.
            """
        elif report_type == "decision":
            prompt = """
            가전 리테일 업체의 마케팅 의사결정 리포트에서 추출한 주요 전략적 조언을 5가지 작성해주세요.
            각 조언은 구체적이고 실행 가능해야 합니다.
            """
        elif report_type == "segmentation":
            prompt = """
            가전 리테일 업체의 고객 세그먼테이션 리포트에서 추출한 주요 인사이트를 5가지 작성해주세요.
            각 인사이트는 구체적이고 실행 가능해야 합니다.
            """
        else:
            prompt = f"""
            가전 리테일 업체의 {report_type} 리포트에서 추출한 주요 인사이트를 5가지 작성해주세요.
            각 인사이트는 구체적이고 실행 가능해야 합니다.
            """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse insights from response
        insights = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line and (line.startswith("1.") or line.startswith("2.") or 
                      line.startswith("3.") or line.startswith("4.") or
                      line.startswith("5.") or line.startswith("- ")):
                insights.append(line)
        
        return insights
    
    def _extract_visualizations_from_report(self, report_path: str) -> List[Dict[str, Any]]:
        """Extract visualizations from a PDF report."""
        # In a real implementation, this would extract images from the PDF
        # For now, we'll return an empty list as a placeholder
        return []
    
    def _extract_insights(self) -> None:
        """Extract and combine insights from all reports."""
        logger.info("Extracting and combining insights")
        
        all_insights = []
        
        # Collect insights from each report
        for report_type, report_data in self.collected_reports.items():
            if "insights" in report_data and report_data["insights"]:
                for insight in report_data["insights"]:
                    all_insights.append({
                        "text": insight,
                        "source": report_type
                    })
        
        # Use LLM to combine and prioritize insights
        if all_insights:
            insights_text = "\n\n".join([f"[{insight['source']}] {insight['text']}" for insight in all_insights])
            
            prompt = f"""
            당신은 가전 리테일 업체의 데이터 분석 전문가입니다. 다음은 여러 분석 리포트에서 추출한 인사이트입니다.
            이 인사이트들을 검토하고, 중복을 제거하며, 우선순위를 정해 10가지 핵심 인사이트로 정리해주세요.
            
            ## 원본 인사이트
            {insights_text}
            
            각 인사이트는 다음 형식으로 작성해주세요:
            1. [카테고리] 인사이트 내용
            
            카테고리는 '트렌드', '예측', '전략', '세그먼트' 중 하나를 선택하세요.
            인사이트는 중요도 순으로 정렬하고, 구체적이며 실행 가능해야 합니다.
            중복된 내용은 통합하고, 상충되는 내용은 더 신뢰할 수 있는 정보를 선택하세요.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse combined insights
            combined_insights = []
            for line in response.content.split("\n"):
                line = line.strip()
                if line and (line.startswith("1.") or line.startswith("2.") or 
                          line.startswith("3.") or line.startswith("4.") or
                          line.startswith("5.") or line.startswith("6.") or
                          line.startswith("7.") or line.startswith("8.") or
                          line.startswith("9.") or line.startswith("10.")):
                    combined_insights.append(line)
            
            self.combined_insights = combined_insights
        
        logger.info(f"Combined {len(all_insights)} insights into {len(self.combined_insights)} key insights")
    
    def _select_visualizations(self) -> None:
        """Select the most relevant visualizations for the report."""
        logger.info("Selecting visualizations")
        
        all_visualizations = []
        
        # Collect visualizations from each report
        for report_type, report_data in self.collected_reports.items():
            if "visualizations" in report_data and report_data["visualizations"]:
                for viz in report_data["visualizations"]:
                    # Add source information
                    viz["source"] = report_type
                    all_visualizations.append(viz)
        
        # Filter visualizations based on report type
        if self.report_type == "executive":
            # For executive reports, select only high-level summary visualizations
            selected_keywords = ["overall", "summary", "trend", "forecast", "growth", "총", "요약", "트렌드", "예측", "성장"]
            self.selected_visualizations = [
                viz for viz in all_visualizations 
                if any(keyword in viz.get("title", "").lower() for keyword in selected_keywords)
            ]
            
            # Limit to top 5 visualizations
            self.selected_visualizations = self.selected_visualizations[:5]
            
        elif self.report_type == "marketing":
            # For marketing reports, focus on customer segments and campaigns
            selected_keywords = ["segment", "campaign", "customer", "channel", "세그먼트", "캠페인", "고객", "채널"]
            self.selected_visualizations = [
                viz for viz in all_visualizations 
                if any(keyword in viz.get("title", "").lower() for keyword in selected_keywords)
            ]
            
        elif self.report_type == "operations":
            # For operations reports, focus on demand forecasts and inventory
            selected_keywords = ["forecast", "demand", "inventory", "supply", "예측", "수요", "재고", "공급"]
            self.selected_visualizations = [
                viz for viz in all_visualizations 
                if any(keyword in viz.get("title", "").lower() for keyword in selected_keywords)
            ]
            
        else:  # comprehensive
            # For comprehensive reports, include all visualizations but limit per category
            # Group by source
            viz_by_source = {}
            for viz in all_visualizations:
                source = viz.get("source", "unknown")
                if source not in viz_by_source:
                    viz_by_source[source] = []
                viz_by_source[source].append(viz)
            
            # Select top visualizations from each source
            selected = []
            for source, vizs in viz_by_source.items():
                selected.extend(vizs[:5])  # Top 5 from each source
            
            self.selected_visualizations = selected
        
        logger.info(f"Selected {len(self.selected_visualizations)} visualizations from {len(all_visualizations)} total")
    
    def _generate_executive_summary(self) -> None:
        """Generate an executive summary for the report."""
        logger.info("Generating executive summary")
        
        # Prepare inputs for the summary
        insights_text = "\n".join([f"- {insight}" for insight in self.combined_insights[:5]])
        
        # Get key metrics if available
        key_metrics = {}
        
        if "forecast" in self.collected_reports and "growth_rates" in self.collected_reports["forecast"]:
            growth_rates = self.collected_reports["forecast"]["growth_rates"]
            if growth_rates and "total_growth_rate" in growth_rates:
                key_metrics["total_growth_rate"] = growth_rates["total_growth_rate"]
        
        if "decision" in self.collected_reports and "campaigns" in self.collected_reports["decision"]:
            campaigns = self.collected_reports["decision"]["campaigns"]
            if campaigns:
                key_metrics["campaign_count"] = len(campaigns)
        
        if "segmentation" in self.collected_reports and "segments" in self.collected_reports["segmentation"]:
            segments = self.collected_reports["segmentation"]["segments"]
            if segments:
                key_metrics["segment_count"] = len(segments)
        
        metrics_text = "\n".join([f"- {k}: {v}" for k, v in key_metrics.items()])
        
        # Create prompt for executive summary
        prompt = f"""
        당신은 가전 리테일 업체의 최고 경영진을 위한 보고서 작성 전문가입니다.
        다음 정보를 바탕으로 간결하고 명확한 경영진용 요약을 작성해주세요.
        
        ## 주요 인사이트
        {insights_text}
        
        ## 핵심 지표
        {metrics_text}
        
        다음 가이드라인을 따라주세요:
        1. 요약은 200-300단어로 제한하세요.
        2. 비즈니스 임팩트와 실행 가능한 조치에 집중하세요.
        3. 전문 용어를 최소화하고 명확한 언어를 사용하세요.
        4. 가장 중요한 발견과 권장사항을 강조하세요.
        5. 요약은 보고서를 읽지 않고도 핵심 내용을 이해할 수 있어야 합니다.
        """
        
        # Get summary from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        self.executive_summary = response.content
        
        logger.info("Executive summary generated")
    
    def _compile_report(self) -> None:
        """Compile the final report."""
        logger.info(f"Compiling {self.report_type} report")
        
        from utils.pdf_generator import ReportPDF
        
        # Create PDF report
        report = ReportPDF()
        
        # Set title based on report type
        if self.report_type == "executive":
            report.set_title("가전 리테일 CDP 경영진 요약 보고서")
        elif self.report_type == "marketing":
            report.set_title("가전 리테일 CDP 마케팅 전략 보고서")
        elif self.report_type == "operations":
            report.set_title("가전 리테일 CDP 운영 계획 보고서")
        else:  # comprehensive
            report.set_title("가전 리테일 CDP 종합 분석 보고서")
        
        report.add_date()
        
        # Add executive summary
        report.add_executive_summary(self.executive_summary)
        
        # Add sections based on report type
        if self.report_type == "executive":
            # Executive report: Focus on high-level insights and recommendations
            report.add_section("핵심 인사이트")
            for insight in self.combined_insights[:5]:
                report.add_text("• " + insight)
            
            # Add key visualizations
            if self.selected_visualizations:
                report.add_section("주요 지표 및 트렌드")
                for viz in self.selected_visualizations[:3]:
                    if "path" in viz and os.path.exists(viz["path"]):
                        report.add_image(viz["path"], width=160, caption=viz.get("title", ""))
                        if "description" in viz:
                            report.add_text(viz["description"])
            
            # Add strategic recommendations
            report.add_section("전략적 권장사항")
            if "decision" in self.collected_reports and "insights" in self.collected_reports["decision"]:
                for advice in self.collected_reports["decision"]["insights"][:3]:
                    report.add_text("• " + advice)
            
        elif self.report_type == "marketing":
            # Marketing report: Focus on segments and campaigns
            
            # Add customer segmentation section
            if "segmentation" in self.collected_reports:
                report.add_section("고객 세그먼테이션")
                
                # Add segment insights
                if "insights" in self.collected_reports["segmentation"]:
                    for insight in self.collected_reports["segmentation"]["insights"][:5]:
                        report.add_text("• " + insight)
                
                # Add segment visualizations
                segment_vizs = [v for v in self.selected_visualizations if v.get("source") == "segmentation"]
                for viz in segment_vizs[:4]:
                    if "path" in viz and os.path.exists(viz["path"]):
                        report.add_image(viz["path"], width=160, caption=viz.get("title", ""))
                        if "description" in viz:
                            report.add_text(viz["description"])
            
            # Add marketing campaigns section
            if "decision" in self.collected_reports:
                report.add_section("마케팅 캠페인")
                
                # Add campaign details
                if "campaigns" in self.collected_reports["decision"]:
                    campaigns = self.collected_reports["decision"]["campaigns"]
                    for i, campaign in enumerate(campaigns):
                        report.add_subsection(f"{i+1}. {campaign.get('name', '캠페인')}")
                        report.add_text(f"목표: {campaign.get('objective', 'N/A')}")
                        report.add_text(f"타겟 고객: {campaign.get('target', 'N/A')}")
                        report.add_text(f"주요 제품/카테고리: {campaign.get('products', 'N/A')}")
                
                # Add resource allocation
                if "resource_allocation" in self.collected_reports["decision"]:
                    report.add_subsection("리소스 할당")
                    allocation = self.collected_reports["decision"]["resource_allocation"]
                    report.add_text(allocation.get("raw_text", "리소스 할당 정보가 없습니다."))
            
        elif self.report_type == "operations":
            # Operations report: Focus on demand forecasts and inventory planning
            
            # Add demand forecast section
            if "forecast" in self.collected_reports:
                report.add_section("수요 예측")
                
                # Add forecast insights
                if "insights" in self.collected_reports["forecast"]:
                    for insight in self.collected_reports["forecast"]["insights"][:5]:
                        report.add_text("• " + insight)
                
                # Add forecast visualizations
                forecast_vizs = [v for v in self.selected_visualizations if v.get("source") == "forecast"]
                for viz in forecast_vizs[:4]:
                    if "path" in viz and os.path.exists(viz["path"]):
                        report.add_image(viz["path"], width=160, caption=viz.get("title", ""))
                        if "description" in viz:
                            report.add_text(viz["description"])
                
                # Add growth rates
                if "growth_rates" in self.collected_reports["forecast"]:
                    growth_rates = self.collected_reports["forecast"]["growth_rates"]
                    if growth_rates and "total_growth_rate" in growth_rates:
                        report.add_subsection("성장률 분석")
                        report.add_text(f"전체 예상 성장률: {growth_rates['total_growth_rate']:.2f}%")
                        
                        if "category_growth" in growth_rates and not growth_rates["category_growth"].empty:
                            report.add_text("카테고리별 성장률:")
                            category_growth = growth_rates["category_growth"]
                            for _, row in category_growth.iterrows():
                                report.add_text(f"• {row['category']}: {row['growth_rate']:.2f}%")
            
            # Add inventory planning section
            report.add_section("재고 계획")
            
            # Generate inventory recommendations based on forecasts
            if "forecast" in self.collected_reports and "insights" in self.collected_reports["forecast"]:
                prompt = f"""
                당신은 가전 리테일 업체의 재고 관리 전문가입니다. 다음 수요 예측 인사이트를 바탕으로
                효과적인 재고 관리 전략을 5가지 제안해주세요.
                
                ## 수요 예측 인사이트
                {chr(10).join([f"- {insight}" for insight in self.collected_reports["forecast"]["insights"]])}
                
                각 제안은 구체적이고 실행 가능해야 합니다.
                """
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                inventory_recommendations = response.content
                
                # Add inventory recommendations
                for line in inventory_recommendations.split("\n"):
                    line = line.strip()
                    if line and (line.startswith("1.") or line.startswith("2.") or 
                              line.startswith("3.") or line.startswith("4.") or
                              line.startswith("5.") or line.startswith("- ")):
                        report.add_text("• " + line.split(".", 1)[1].strip() if "." in line else line)
            
        else:  # comprehensive
            # Comprehensive report: Include all sections
            
            # Add trend analysis section
            if "trend" in self.collected_reports:
                report.add_section("트렌드 분석")
                
                # Add trend insights
                if "insights" in self.collected_reports["trend"]:
                    for insight in self.collected_reports["trend"]["insights"][:5]:
                        report.add_text("• " + insight)
                
                # Add trend visualizations
                trend_vizs = [v for v in self.selected_visualizations if v.get("source") == "trend"]
                for viz in trend_vizs[:3]:
                    if "path" in viz and os.path.exists(viz["path"]):
                        report.add_image(viz["path"], width=160, caption=viz.get("title", ""))
                        if "description" in viz:
                            report.add_text(viz["description"])
            
            # Add demand forecast section
            if "forecast" in self.collected_reports:
                report.add_section("수요 예측")
                
                # Add forecast insights
                if "insights" in self.collected_reports["forecast"]:
                    for insight in self.collected_reports["forecast"]["insights"][:5]:
                        report.add_text("• " + insight)
                
                # Add forecast visualizations
                forecast_vizs = [v for v in self.selected_visualizations if v.get("source") == "forecast"]
                for viz in forecast_vizs[:3]:
                    if "path" in viz and os.path.exists(viz["path"]):
                        report.add_image(viz["path"], width=160, caption=viz.get("title", ""))
                        if "description" in viz:
                            report.add_text(viz["description"])
            
            # Add customer segmentation section
            if "segmentation" in self.collected_reports:
                report.add_section("고객 세그먼테이션")
                
                # Add segment insights
                if "insights" in self.collected_reports["segmentation"]:
                    for insight in self.collected_reports["segmentation"]["insights"][:5]:
                        report.add_text("• " + insight)
                
                # Add segment visualizations
                segment_vizs = [v for v in self.selected_visualizations if v.get("source") == "segmentation"]
                for viz in segment_vizs[:3]:
                    if "path" in viz and os.path.exists(viz["path"]):
                        report.add_image(viz["path"], width=160, caption=viz.get("title", ""))
                        if "description" in viz:
                            report.add_text(viz["description"])
            
            # Add marketing strategy section
            if "decision" in self.collected_reports:
                report.add_section("마케팅 전략")
                
                # Add strategic advice
                if "insights" in self.collected_reports["decision"]:
                    for advice in self.collected_reports["decision"]["insights"][:5]:
                        report.add_text("• " + advice)
                
                # Add campaign details
                if "campaigns" in self.collected_reports["decision"]:
                    report.add_subsection("선정된 마케팅 캠페인")
                    campaigns = self.collected_reports["decision"]["campaigns"]
                    for i, campaign in enumerate(campaigns):
                        report.add_text(f"{i+1}. {campaign.get('name', '캠페인')}: {campaign.get('objective', 'N/A')}")
        
        # Add recommendations section for all report types
        report.add_section("권장사항 및 다음 단계")
        
        # Generate recommendations based on all insights
        prompt = f"""
        당신은 가전 리테일 업체의 데이터 분석 전문가입니다. 다음 인사이트를 바탕으로
        구체적인 권장사항과 다음 단계를 5가지 작성해주세요.
        
        ## 인사이트
        {chr(10).join([f"- {insight}" for insight in self.combined_insights])}
        
        각 권장사항은 다음을 포함해야 합니다:
        1. 구체적인 실행 방안
        2. 예상되는 비즈니스 임팩트
        3. 실행 우선순위
        
        권장사항은 간결하고 실행 가능해야 합니다.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        recommendations = response.content
        
        # Add recommendations
        for line in recommendations.split("\n"):
            line = line.strip()
            if line and (line.startswith("1.") or line.startswith("2.") or 
                      line.startswith("3.") or line.startswith("4.") or
                      line.startswith("5.") or line.startswith("- ")):
                report.add_text("• " + line)
        
        # Generate and save the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(config.REPORTS_DIR, f"{self.report_type}_report_{timestamp}.pdf")
        report.output(report_path)
        
        # Update state with report path
        self.report_path = report_path
        
        logger.info(f"{self.report_type} report generated successfully at {report_path}")
    
    def get_report_path(self) -> str:
        """
        Get the path to the generated report.
        
        Returns:
            Path to the PDF report
        """
        return self.report_path
    
    def get_combined_insights(self) -> List[str]:
        """
        Get the combined insights from all reports.
        
        Returns:
            List of combined insight strings
        """
        return self.combined_insights
    
    def get_executive_summary(self) -> str:
        """
        Get the executive summary.
        
        Returns:
            Executive summary text
        """
        return self.executive_summary
    
    def generate_custom_report(self, report_config: Dict[str, Any]) -> str:
        """
        Generate a custom report based on specific configuration.
        
        Args:
            report_config: Dictionary with custom report configuration
            
        Returns:
            Path to the generated custom report
        """
        logger.info("Generating custom report")
        
        # Store original report type
        original_report_type = self.report_type
        
        # Update report type if specified
        if "report_type" in report_config:
            self.report_type = report_config["report_type"]
        
        # Run the reporting process
        self._collect_reports()
        
        # Apply custom filters for insights if specified
        if "insight_filters" in report_config:
            self._extract_insights()
            filters = report_config["insight_filters"]
            
            filtered_insights = []
            for insight in self.combined_insights:
                if any(f.lower() in insight.lower() for f in filters):
                    filtered_insights.append(insight)
            
            self.combined_insights = filtered_insights
        else:
            self._extract_insights()
        
        # Apply custom filters for visualizations if specified
        if "visualization_filters" in report_config:
            filters = report_config["visualization_filters"]
            
            all_visualizations = []
            for report_type, report_data in self.collected_reports.items():
                if "visualizations" in report_data and report_data["visualizations"]:
                    for viz in report_data["visualizations"]:
                        viz["source"] = report_type
                        all_visualizations.append(viz)
            
            self.selected_visualizations = [
                viz for viz in all_visualizations 
                if any(f.lower() in viz.get("title", "").lower() for f in filters)
            ]
        else:
            self._select_visualizations()
        
        # Generate custom executive summary if specified
        if "custom_summary" in report_config:
            self.executive_summary = report_config["custom_summary"]
        else:
            self._generate_executive_summary()
        
        # Compile the report
        self._compile_report()
        
        # Restore original report type
        self.report_type = original_report_type
        
        return self.report_path
    
    def distribute_report(self, distribution_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distribute the generated report according to configuration.
        
        Args:
            distribution_config: Dictionary with distribution configuration
            
        Returns:
            Dictionary with distribution results
        """
        logger.info("Distributing report")
        
        if not self.report_path:
            raise ValueError("No report has been generated yet. Run the reporting process first.")
        
        # In a real implementation, this would handle email distribution, cloud storage, etc.
        # For now, we'll just return a placeholder result
        
        return {
            "status": "success",
            "report_path": self.report_path,
            "distribution_method": distribution_config.get("method", "local"),
            "timestamp": datetime.now().isoformat()
        }

# For direct execution
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run the agent
    agent = ReportingAgent(report_type="comprehensive")
    report_path = agent.run()
    
    print(f"Report generated at: {report_path}")
    print(f"Combined {len(agent.get_combined_insights())} insights")
    print(f"Executive summary length: {len(agent.get_executive_summary())} characters")
