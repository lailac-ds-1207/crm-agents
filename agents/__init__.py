"""
Agent package for the CRM-Agent system.
This package contains various agent modules for trend analysis, demand forecasting,
decision making, segmentation, and report generation.
"""

# Package metadata
__version__ = '0.1.0'
__author__ = 'CDP AI Agent Team'
__description__ = 'Multi CRM-Agent system for retail CDP analysis and campaign planning'

# Import main components from agent modules
# These imports will be uncommented as the modules are implemented

# Trend Analysis Agent
# from agents.trend_analysis import TrendAnalysisAgent, TrendAnalysisGraph

# Demand Forecast Agent
# from agents.demand_forecast import DemandForecastAgent, ForecastGraph

# Decision Agent
# from agents.decision import DecisionAgent, DecisionGraph

# Segmentation Agent
# from agents.segmentation import SegmentationAgent, SegmentationGraph

# Report Generation Agent
# from agents.reporting import ReportGenerationAgent, ReportGraph

# Agent registry - will be populated as agents are implemented
AGENT_REGISTRY = {
    'trend_analysis': {
        'description': '트렌드 분석 Agent: 카테고리별 분석/시각화를 통해 트렌드 분석 리포트 작성',
        'sub_agents': [
            '데이터수집', '카테고리분석', '채널분석', '고객행동분석', '트렌드시각화', '리포트작성'
        ]
    },
    'demand_forecast': {
        'description': '수요 예측 Agent: 카테고리별 판매량/판매액 예측 ML 코드 작성 및 실행, 결과 리포트화',
        'sub_agents': [
            '데이터준비', '모델선택', '모델학습', '예측실행', '결과분석', '리포트작성'
        ]
    },
    'decision': {
        'description': 'Decision Agent: 트렌드 분석과 수요 예측 리포트를 토대로 강점/약점 파악 및 캠페인 제안',
        'sub_agents': [
            '트렌드전문가', '예측전문가', '마케팅전략가', '리테일전문가', '시즌전문가', '조정자'
        ]
    },
    'segmentation': {
        'description': '세그멘테이션 Agent: 캠페인별 타겟 세그먼트를 자연어로 정의하고 SQL 쿼리 실행',
        'sub_agents': [
            '자연어정의', 'SQL변환', '쿼리검증', '쿼리실행', '세그먼트추출'
        ]
    },
    'reporting': {
        'description': '결과보고서 작성 Agent: 전체 과정의 결과를 요약하고 세그먼트 분석 리포트 작성',
        'sub_agents': [
            '개요', '트렌드요약', '예측요약', '캠페인설명', '세그먼트분석', '기대효과', '실행계획'
        ]
    }
}

def get_agent_info(agent_type=None):
    """
    Get information about available agents.
    
    Args:
        agent_type: Optional agent type to get specific information
        
    Returns:
        Dictionary with agent information
    """
    if agent_type and agent_type in AGENT_REGISTRY:
        return AGENT_REGISTRY[agent_type]
    return AGENT_REGISTRY
