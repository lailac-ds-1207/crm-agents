"""
Decision Agent for the CRM-Agent system.
This module defines the main class and interface for the decision-making functionality.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import config
from utils.bigquery import BigQueryConnector
from agents.trend_analysis.agent import TrendAnalysisAgent
from agents.demand_forecast.agent import DemandForecastAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DecisionAgent:
    """
    Agent for making strategic decisions based on CDP data analysis.
    
    This agent orchestrates the process of evaluating options, selecting optimal
    marketing campaigns, allocating resources, and generating strategic advice
    through multi-agent discussion and collaboration.
    """
    
    def __init__(
        self,
        project_id: str = None,
        dataset_id: str = None,
        trend_report_path: str = None,
        forecast_report_path: str = None,
        bq_connector: BigQueryConnector = None
    ):
        """
        Initialize the Decision Agent.
        
        Args:
            project_id: Google Cloud project ID (if None, uses the value from config)
            dataset_id: BigQuery dataset ID (if None, uses the value from config)
            trend_report_path: Path to trend analysis report (if None, will run trend analysis)
            forecast_report_path: Path to demand forecast report (if None, will run demand forecast)
            bq_connector: Optional existing BigQuery connector to reuse
        """
        self.project_id = project_id or config.GCP_PROJECT_ID
        self.dataset_id = dataset_id or config.BQ_DATASET_ID
        self.trend_report_path = trend_report_path
        self.forecast_report_path = forecast_report_path
        
        # Initialize BigQuery connector if not provided
        self.bq_connector = bq_connector or BigQueryConnector(project_id=self.project_id)
        
        # Initialize agent states
        self.trend_insights = []
        self.forecast_insights = []
        self.campaign_options = []
        self.selected_campaigns = []
        self.resource_allocation = {}
        self.strategic_advice = []
        self.discussion_summary = []
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
        
        logger.info(f"Decision Agent initialized for dataset {self.dataset_id}")
    
    def run(self) -> str:
        """
        Run the decision-making workflow.
        
        Returns:
            Path to the generated report
        """
        logger.info("Starting decision-making workflow")
        
        try:
            # Step 1: Get trend analysis and demand forecast insights
            self._gather_insights()
            
            # Step 2: Generate campaign options
            self._generate_campaign_options()
            
            # Step 3: Conduct multi-agent discussion
            self._conduct_discussion()
            
            # Step 4: Select optimal campaigns
            self._select_campaigns()
            
            # Step 5: Allocate resources
            self._allocate_resources()
            
            # Step 6: Generate strategic advice
            self._generate_strategic_advice()
            
            # Step 7: Generate report
            self._generate_report()
            
            logger.info(f"Decision-making completed successfully. Report: {self.report_path}")
            return self.report_path
            
        except Exception as e:
            logger.error(f"Error running decision-making process: {str(e)}")
            raise
    
    def _gather_insights(self):
        """Gather insights from trend analysis and demand forecast."""
        logger.info("Gathering insights from trend analysis and demand forecast")
        
        # Run trend analysis if report path not provided
        if not self.trend_report_path:
            logger.info("Running trend analysis")
            trend_agent = TrendAnalysisAgent(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                bq_connector=self.bq_connector
            )
            self.trend_report_path = trend_agent.run()
            self.trend_insights = trend_agent.get_insights()
        else:
            # Extract insights from existing trend report
            logger.info(f"Using existing trend report: {self.trend_report_path}")
            self.trend_insights = self._extract_insights_from_report(self.trend_report_path)
        
        # Run demand forecast if report path not provided
        if not self.forecast_report_path:
            logger.info("Running demand forecast")
            forecast_agent = DemandForecastAgent(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                bq_connector=self.bq_connector
            )
            self.forecast_report_path = forecast_agent.run()
            self.forecast_insights = forecast_agent.get_insights()
        else:
            # Extract insights from existing forecast report
            logger.info(f"Using existing forecast report: {self.forecast_report_path}")
            self.forecast_insights = self._extract_insights_from_report(self.forecast_report_path)
        
        logger.info(f"Gathered {len(self.trend_insights)} trend insights and {len(self.forecast_insights)} forecast insights")
    
    def _extract_insights_from_report(self, report_path: str) -> List[str]:
        """Extract insights from a PDF report."""
        # In a real implementation, this would use PDF text extraction
        # For now, we'll use a placeholder approach
        
        logger.info(f"Extracting insights from report: {report_path}")
        
        # Use LLM to generate insights based on the report name
        if "trend" in report_path.lower():
            prompt = """
            가전 리테일 업체의 트렌드 분석 리포트에서 추출한 주요 인사이트를 5가지 작성해주세요.
            각 인사이트는 구체적이고 실행 가능해야 합니다.
            """
        else:  # forecast report
            prompt = """
            가전 리테일 업체의 수요 예측 리포트에서 추출한 주요 인사이트를 5가지 작성해주세요.
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
    
    def _generate_campaign_options(self):
        """Generate marketing campaign options based on insights."""
        logger.info("Generating marketing campaign options")
        
        # Combine insights
        all_insights = self.trend_insights + self.forecast_insights
        insights_text = "\n".join([f"- {insight}" for insight in all_insights])
        
        # Use LLM to generate campaign options
        prompt = f"""
        당신은 가전 리테일 업체의 마케팅 전략가입니다. 다음 인사이트를 바탕으로 
        효과적인 마케팅 캠페인 옵션을 5가지 제안해주세요.
        
        ## 인사이트
        {insights_text}
        
        각 캠페인 옵션은 다음 형식으로 작성해주세요:
        1. 캠페인 이름: [이름]
        2. 목표: [목표]
        3. 타겟 고객: [타겟]
        4. 주요 제품/카테고리: [제품/카테고리]
        5. 채널 전략: [채널]
        6. 예상 ROI: [ROI]
        7. 실행 기간: [기간]
        
        각 캠페인은 인사이트에 기반하여 구체적이고 실행 가능해야 합니다.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse campaign options
        campaign_text = response.content
        campaign_sections = campaign_text.split("\n\n")
        
        self.campaign_options = []
        current_campaign = {}
        
        for section in campaign_sections:
            if "캠페인 이름:" in section or "1. 캠페인 이름:" in section:
                if current_campaign:
                    self.campaign_options.append(current_campaign)
                current_campaign = {"raw_text": section}
                
                # Parse campaign details
                lines = section.split("\n")
                for line in lines:
                    if "캠페인 이름:" in line or "1. 캠페인 이름:" in line:
                        current_campaign["name"] = line.split(":", 1)[1].strip()
                    elif "목표:" in line or "2. 목표:" in line:
                        current_campaign["objective"] = line.split(":", 1)[1].strip()
                    elif "타겟 고객:" in line or "3. 타겟 고객:" in line:
                        current_campaign["target"] = line.split(":", 1)[1].strip()
                    elif "주요 제품/카테고리:" in line or "4. 주요 제품/카테고리:" in line:
                        current_campaign["products"] = line.split(":", 1)[1].strip()
                    elif "채널 전략:" in line or "5. 채널 전략:" in line:
                        current_campaign["channels"] = line.split(":", 1)[1].strip()
                    elif "예상 ROI:" in line or "6. 예상 ROI:" in line:
                        current_campaign["roi"] = line.split(":", 1)[1].strip()
                    elif "실행 기간:" in line or "7. 실행 기간:" in line:
                        current_campaign["duration"] = line.split(":", 1)[1].strip()
        
        # Add the last campaign
        if current_campaign:
            self.campaign_options.append(current_campaign)
        
        logger.info(f"Generated {len(self.campaign_options)} campaign options")
    
    def _conduct_discussion(self):
        """Conduct a multi-agent discussion to evaluate options."""
        logger.info("Conducting multi-agent discussion")
        
        # Define agent personas
        personas = [
            {"name": "마케팅 전략가", "expertise": "마케팅 캠페인 기획 및 전략 수립", "focus": "브랜드 인지도와 고객 참여"},
            {"name": "데이터 분석가", "expertise": "고객 데이터 분석 및 세그먼테이션", "focus": "데이터 기반 의사결정과 타겟팅"},
            {"name": "재무 분석가", "expertise": "ROI 분석 및 예산 할당", "focus": "비용 효율성과 수익성"},
            {"name": "고객 경험 전문가", "expertise": "고객 여정 및 경험 디자인", "focus": "고객 만족도와 충성도"},
            {"name": "제품 관리자", "expertise": "제품 포트폴리오 및 카테고리 관리", "focus": "제품 판매 및 재고 최적화"}
        ]
        
        # Format campaign options for discussion
        campaigns_text = "\n\n".join([
            f"캠페인 {i+1}: {campaign['name']}\n"
            f"목표: {campaign.get('objective', 'N/A')}\n"
            f"타겟: {campaign.get('target', 'N/A')}\n"
            f"제품/카테고리: {campaign.get('products', 'N/A')}\n"
            f"채널: {campaign.get('channels', 'N/A')}\n"
            f"예상 ROI: {campaign.get('roi', 'N/A')}\n"
            f"기간: {campaign.get('duration', 'N/A')}"
            for i, campaign in enumerate(self.campaign_options)
        ])
        
        # Initialize discussion
        discussion = []
        
        # Moderator introduces the discussion
        discussion.append({
            "role": "토론 진행자",
            "content": "안녕하세요, 오늘은 트렌드 분석과 수요 예측 결과를 바탕으로 최적의 마케팅 캠페인을 선정하는 회의를 진행하겠습니다. "
                      "각자의 전문 분야에서 제안된 캠페인 옵션을 평가해주시기 바랍니다."
        })
        
        # Each persona evaluates the campaigns
        for persona in personas:
            prompt = f"""
            당신은 가전 리테일 업체의 {persona['name']}입니다. {persona['expertise']}을 전문으로 하며, {persona['focus']}에 중점을 둡니다.
            
            다음 마케팅 캠페인 옵션들을 평가해주세요:
            
            {campaigns_text}
            
            당신의 전문 분야와 관점에서 각 캠페인을 평가하고, 가장 효과적일 것으로 생각되는 1-2개의 캠페인을 추천해주세요.
            추천 이유와 함께 개선 제안도 포함해주세요.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            discussion.append({
                "role": persona["name"],
                "content": response.content
            })
        
        # Moderator asks for response to others' points
        discussion.append({
            "role": "토론 진행자",
            "content": "각자 다른 전문가들의 의견을 들었습니다. 이제 서로의 의견에 대한 생각과 최종 추천을 공유해주세요."
        })
        
        # Generate responses to others' points
        for i, persona in enumerate(personas):
            # Collect other personas' opinions
            others_opinions = "\n\n".join([
                f"{discussion[j+1]['role']}의 의견:\n{discussion[j+1]['content']}"
                for j in range(len(personas)) if j != i
            ])
            
            prompt = f"""
            당신은 가전 리테일 업체의 {persona['name']}입니다. {persona['expertise']}을 전문으로 하며, {persona['focus']}에 중점을 둡니다.
            
            다른 전문가들이 제시한 의견을 검토하고 최종 의견을 제시해주세요:
            
            {others_opinions}
            
            다른 전문가들의 의견을 고려하여, 당신의 최종 추천과 그 이유를 간결하게 작성해주세요.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            discussion.append({
                "role": f"{persona['name']} (최종 의견)",
                "content": response.content
            })
        
        # Moderator summarizes the discussion
        all_opinions = "\n\n".join([
            f"{entry['role']}: {entry['content']}" for entry in discussion[1:]  # Skip the first moderator intro
        ])
        
        prompt = f"""
        당신은 가전 리테일 업체의 의사결정 회의 진행자입니다. 다음 전문가들의 토론 내용을 요약하고,
        가장 많은 지지를 받은 캠페인과 그 이유를 정리해주세요.
        
        {all_opinions}
        
        다음 형식으로 요약해주세요:
        1. 토론 요약: [주요 논점과 의견 차이]
        2. 최다 지지 캠페인: [캠페인 이름과 번호]
        3. 지지 이유: [전문가들이 제시한 주요 지지 이유]
        4. 개선 제안: [캠페인 실행 시 고려할 개선 사항]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        discussion.append({
            "role": "토론 진행자 (요약)",
            "content": response.content
        })
        
        # Store the discussion
        self.discussion_summary = discussion
        
        logger.info(f"Completed multi-agent discussion with {len(discussion)} entries")
    
    def _select_campaigns(self):
        """Select optimal campaigns based on the discussion."""
        logger.info("Selecting optimal campaigns")
        
        # Extract the moderator's summary
        moderator_summary = self.discussion_summary[-1]["content"]
        
        # Use LLM to identify the selected campaigns
        prompt = f"""
        다음은 마케팅 캠페인 선정을 위한 전문가 토론의 요약입니다:
        
        {moderator_summary}
        
        원래 제안된 캠페인 목록:
        {", ".join([f"{i+1}. {campaign['name']}" for i, campaign in enumerate(self.campaign_options)])}
        
        토론 결과를 바탕으로 최종 선정된 캠페인 번호와 이름을 정확히 추출해주세요.
        여러 캠페인이 선정된 경우 모두 포함해주세요.
        
        응답 형식:
        선정된 캠페인: [번호]. [이름], [번호]. [이름], ...
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        selected_text = response.content
        
        # Parse selected campaigns
        self.selected_campaigns = []
        
        for i, campaign in enumerate(self.campaign_options):
            campaign_num = str(i + 1)
            campaign_name = campaign["name"]
            
            if campaign_num in selected_text and campaign_name in selected_text:
                self.selected_campaigns.append(campaign)
        
        # If no campaigns were selected, choose the first one as fallback
        if not self.selected_campaigns and self.campaign_options:
            self.selected_campaigns = [self.campaign_options[0]]
            logger.warning("No campaigns were clearly selected from discussion. Using first option as fallback.")
        
        logger.info(f"Selected {len(self.selected_campaigns)} campaigns")
    
    def _allocate_resources(self):
        """Allocate resources to the selected campaigns."""
        logger.info("Allocating resources to selected campaigns")
        
        # Use LLM to allocate resources
        campaigns_text = "\n\n".join([
            f"캠페인 {i+1}: {campaign['name']}\n"
            f"목표: {campaign.get('objective', 'N/A')}\n"
            f"타겟: {campaign.get('target', 'N/A')}\n"
            f"제품/카테고리: {campaign.get('products', 'N/A')}\n"
            f"채널: {campaign.get('channels', 'N/A')}\n"
            f"예상 ROI: {campaign.get('roi', 'N/A')}\n"
            f"기간: {campaign.get('duration', 'N/A')}"
            for i, campaign in enumerate(self.selected_campaigns)
        ])
        
        prompt = f"""
        다음은 가전 리테일 업체의 최종 선정된 마케팅 캠페인입니다:
        
        {campaigns_text}
        
        총 마케팅 예산은 1억원이며, 다음 리소스를 할당해야 합니다:
        1. 예산 (총 1억원)
        2. 마케팅 인력 (총 10명)
        3. 채널별 집행 비율 (온라인/오프라인)
        4. 실행 일정 (다음 분기 내)
        
        각 캠페인에 대해 위 리소스를 최적으로 할당하고, 그 이유를 설명해주세요.
        캠페인이 여러 개인 경우, 각 캠페인의 중요도와 예상 효과에 따라 적절히 배분해주세요.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        allocation_text = response.content
        
        # Store the resource allocation
        self.resource_allocation = {
            "raw_text": allocation_text,
            "campaigns": []
        }
        
        # Parse allocation for each campaign
        for i, campaign in enumerate(self.selected_campaigns):
            campaign_name = campaign["name"]
            
            # Find sections related to this campaign
            campaign_sections = [
                section for section in allocation_text.split("\n\n")
                if campaign_name in section
            ]
            
            campaign_allocation = {
                "name": campaign_name,
                "budget": "N/A",
                "staff": "N/A",
                "channels": "N/A",
                "timeline": "N/A",
                "raw_text": "\n\n".join(campaign_sections)
            }
            
            # Extract specific allocations
            for section in campaign_sections:
                if "예산" in section:
                    budget_line = [line for line in section.split("\n") if "예산" in line]
                    if budget_line:
                        campaign_allocation["budget"] = budget_line[0].split(":", 1)[1].strip() if ":" in budget_line[0] else budget_line[0]
                
                if "인력" in section:
                    staff_line = [line for line in section.split("\n") if "인력" in line]
                    if staff_line:
                        campaign_allocation["staff"] = staff_line[0].split(":", 1)[1].strip() if ":" in staff_line[0] else staff_line[0]
                
                if "채널" in section:
                    channel_line = [line for line in section.split("\n") if "채널" in line]
                    if channel_line:
                        campaign_allocation["channels"] = channel_line[0].split(":", 1)[1].strip() if ":" in channel_line[0] else channel_line[0]
                
                if "일정" in section or "기간" in section:
                    timeline_line = [line for line in section.split("\n") if "일정" in line or "기간" in line]
                    if timeline_line:
                        campaign_allocation["timeline"] = timeline_line[0].split(":", 1)[1].strip() if ":" in timeline_line[0] else timeline_line[0]
            
            self.resource_allocation["campaigns"].append(campaign_allocation)
        
        logger.info(f"Allocated resources to {len(self.resource_allocation['campaigns'])} campaigns")
    
    def _generate_strategic_advice(self):
        """Generate strategic advice for campaign execution."""
        logger.info("Generating strategic advice")
        
        # Combine all information for context
        campaigns_text = "\n\n".join([
            f"캠페인: {campaign['name']}\n"
            f"목표: {campaign.get('objective', 'N/A')}\n"
            f"타겟: {campaign.get('target', 'N/A')}\n"
            f"제품/카테고리: {campaign.get('products', 'N/A')}\n"
            f"채널: {campaign.get('channels', 'N/A')}\n"
            f"할당 예산: {next((alloc['budget'] for alloc in self.resource_allocation['campaigns'] if alloc['name'] == campaign['name']), 'N/A')}\n"
            f"할당 인력: {next((alloc['staff'] for alloc in self.resource_allocation['campaigns'] if alloc['name'] == campaign['name']), 'N/A')}"
            for campaign in self.selected_campaigns
        ])
        
        # Use LLM to generate strategic advice
        prompt = f"""
        당신은 가전 리테일 업체의 최고 마케팅 책임자(CMO)입니다. 다음 선정된 마케팅 캠페인의 성공적인 실행을 위한 
        전략적 조언을 작성해주세요.
        
        ## 선정된 캠페인 정보
        {campaigns_text}
        
        ## 트렌드 분석 인사이트
        {chr(10).join([f"- {insight}" for insight in self.trend_insights[:3]])}
        
        ## 수요 예측 인사이트
        {chr(10).join([f"- {insight}" for insight in self.forecast_insights[:3]])}
        
        다음 항목에 대한 구체적인 전략적 조언을 작성해주세요:
        1. 실행 우선순위 및 로드맵
        2. 성공적인 캠페인 실행을 위한 핵심 고려사항
        3. 잠재적 리스크 및 대응 방안
        4. 캠페인 성과 측정 방법 및 KPI
        5. 캠페인 최적화 및 조정 전략
        
        각 항목은 구체적이고 실행 가능한 조언이어야 합니다.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        advice_text = response.content
        
        # Parse strategic advice
        sections = advice_text.split("\n\n")
        self.strategic_advice = []
        
        current_section = ""
        for section in sections:
            if section.strip().startswith("1.") or section.strip().startswith("2.") or \
               section.strip().startswith("3.") or section.strip().startswith("4.") or \
               section.strip().startswith("5."):
                if current_section:
                    self.strategic_advice.append(current_section)
                current_section = section
            elif current_section:
                current_section += "\n\n" + section
        
        if current_section:
            self.strategic_advice.append(current_section)
        
        logger.info(f"Generated {len(self.strategic_advice)} strategic advice points")
    
    def _generate_report(self):
        """Generate a PDF report with decision results."""
        logger.info("Generating decision report")
        
        from utils.pdf_generator import ReportPDF
        
        # Create PDF report
        report = ReportPDF()
        report.set_title("가전 리테일 마케팅 의사결정 리포트")
        report.add_date()
        
        # Add executive summary
        summary = f"본 리포트는 트렌드 분석과 수요 예측 결과를 바탕으로 {len(self.selected_campaigns)}개의 최적 마케팅 캠페인을 선정하고, "
        summary += f"리소스 할당 및 실행 전략을 제시합니다. "
        summary += f"선정된 캠페인은 {', '.join([campaign['name'] for campaign in self.selected_campaigns])}이며, "
        summary += "각 캠페인은 데이터 기반 인사이트와 전문가 토론을 통해 선정되었습니다."
        
        report.add_executive_summary(summary)
        
        # Add insights section
        report.add_section("데이터 인사이트")
        report.add_text("트렌드 분석 및 수요 예측에서 도출된 주요 인사이트:")
        
        for insight in self.trend_insights[:3]:
            report.add_text("• " + insight)
        
        for insight in self.forecast_insights[:3]:
            report.add_text("• " + insight)
        
        # Add selected campaigns section
        report.add_section("선정된 마케팅 캠페인")
        
        for i, campaign in enumerate(self.selected_campaigns):
            report.add_subsection(f"{i+1}. {campaign['name']}")
            report.add_text(f"목표: {campaign.get('objective', 'N/A')}")
            report.add_text(f"타겟 고객: {campaign.get('target', 'N/A')}")
            report.add_text(f"주요 제품/카테고리: {campaign.get('products', 'N/A')}")
            report.add_text(f"채널 전략: {campaign.get('channels', 'N/A')}")
            report.add_text(f"예상 ROI: {campaign.get('roi', 'N/A')}")
            report.add_text(f"실행 기간: {campaign.get('duration', 'N/A')}")
        
        # Add resource allocation section
        report.add_section("리소스 할당")
        report.add_text("선정된 캠페인에 대한 리소스 할당 계획:")
        
        for allocation in self.resource_allocation["campaigns"]:
            report.add_subsection(allocation["name"])
            report.add_text(f"예산: {allocation['budget']}")
            report.add_text(f"인력: {allocation['staff']}")
            report.add_text(f"채널 배분: {allocation['channels']}")
            report.add_text(f"실행 일정: {allocation['timeline']}")
        
        # Add strategic advice section
        report.add_section("전략적 조언")
        
        for advice in self.strategic_advice:
            report.add_text(advice)
        
        # Add discussion summary section
        report.add_section("전문가 토론 요약")
        moderator_summary = self.discussion_summary[-1]["content"]
        report.add_text(moderator_summary)
        
        # Generate and save the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(config.REPORTS_DIR, f"decision_report_{timestamp}.pdf")
        report.output(report_path)
        
        # Update state with report path
        self.report_path = report_path
        
        logger.info(f"Decision report generated successfully at {report_path}")
    
    def get_selected_campaigns(self) -> List[Dict[str, Any]]:
        """
        Get the selected campaigns.
        
        Returns:
            List of selected campaign dictionaries
        """
        return self.selected_campaigns
    
    def get_resource_allocation(self) -> Dict[str, Any]:
        """
        Get the resource allocation plan.
        
        Returns:
            Dictionary with resource allocation information
        """
        return self.resource_allocation
    
    def get_strategic_advice(self) -> List[str]:
        """
        Get the strategic advice.
        
        Returns:
            List of strategic advice points
        """
        return self.strategic_advice
    
    def get_discussion_summary(self) -> List[Dict[str, str]]:
        """
        Get the multi-agent discussion summary.
        
        Returns:
            List of discussion entries with role and content
        """
        return self.discussion_summary
    
    def get_report_path(self) -> str:
        """
        Get the path to the generated report.
        
        Returns:
            Path to the PDF report
        """
        return self.report_path
    
    def adjust_campaign_parameters(self, campaign_index: int, updates: Dict[str, str]) -> Dict[str, Any]:
        """
        Adjust parameters for a selected campaign.
        
        Args:
            campaign_index: Index of the campaign to update
            updates: Dictionary with updated parameters
            
        Returns:
            Updated campaign dictionary
        """
        if not self.selected_campaigns:
            raise ValueError("No campaigns have been selected yet. Run the decision process first.")
        
        if campaign_index < 0 or campaign_index >= len(self.selected_campaigns):
            raise ValueError(f"Invalid campaign index: {campaign_index}. Must be between 0 and {len(self.selected_campaigns)-1}")
        
        # Update campaign parameters
        for key, value in updates.items():
            if key in self.selected_campaigns[campaign_index]:
                self.selected_campaigns[campaign_index][key] = value
        
        logger.info(f"Updated parameters for campaign: {self.selected_campaigns[campaign_index]['name']}")
        
        return self.selected_campaigns[campaign_index]
    
    def generate_custom_advice(self, question: str) -> str:
        """
        Generate custom strategic advice based on a specific question.
        
        Args:
            question: Question to analyze
            
        Returns:
            Generated advice text
        """
        if not self.selected_campaigns:
            raise ValueError("No campaigns have been selected yet. Run the decision process first.")
        
        # Prepare prompt with campaign information and the question
        campaigns_text = "\n\n".join([
            f"캠페인: {campaign['name']}\n"
            f"목표: {campaign.get('objective', 'N/A')}\n"
            f"타겟: {campaign.get('target', 'N/A')}"
            for campaign in self.selected_campaigns
        ])
        
        prompt = f"""
        당신은 가전 리테일 업체의 마케팅 전략 전문가입니다. 다음 선정된 마케팅 캠페인과 질문을 바탕으로 
        전략적 조언을 제공해주세요.
        
        ## 선정된 캠페인
        {campaigns_text}
        
        ## 질문
        {question}
        
        위 정보를 바탕으로 질문에 대한 구체적이고 실행 가능한 조언을 제공해주세요.
        """
        
        # Get advice from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

# For direct execution
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run the agent
    agent = DecisionAgent()
    report_path = agent.run()
    
    print(f"Report generated at: {report_path}")
    print(f"Selected {len(agent.get_selected_campaigns())} campaigns")
    print(f"Generated {len(agent.get_strategic_advice())} strategic advice points")
