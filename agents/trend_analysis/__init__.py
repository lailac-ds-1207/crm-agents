"""
Trend Analysis Agent package for the CRM-Agent system.
This package contains components for analyzing trends in CDP data,
creating visualizations, and generating reports.
"""

# Package metadata
__version__ = '0.1.0'
__author__ = 'CDP AI Agent Team'
__description__ = 'Trend Analysis Agent for retail CDP data'

# Import main components from submodules
# These will be uncommented as the modules are implemented
# from .agent import TrendAnalysisAgent
# from .graph import TrendAnalysisGraph
# from .data_collector import DataCollector
# from .category_analyzer import CategoryAnalyzer
# from .channel_analyzer import ChannelAnalyzer
# from .customer_analyzer import CustomerAnalyzer
# from .visualizer import TrendVisualizer
# from .report_generator import TrendReportGenerator

# Sub-agent registry
SUB_AGENTS = {
    'data_collector': {
        'name': '데이터수집',
        'description': 'BigQuery에서 관련 테이블 데이터 쿼리 및 전처리',
        'role': 'CDP 데이터를 수집하고 분석에 적합한 형태로 전처리합니다.'
    },
    'category_analyzer': {
        'name': '카테고리분석',
        'description': '제품 카테고리별 판매 트렌드 분석',
        'role': '카테고리별 매출, 성장률, 시즌성 등을 분석합니다.'
    },
    'channel_analyzer': {
        'name': '채널분석',
        'description': '온/오프라인 채널별 실적 분석',
        'role': '온라인과 오프라인 채널의 성과를 비교 분석합니다.'
    },
    'customer_analyzer': {
        'name': '고객행동분석',
        'description': '고객 행동 패턴 및 선호도 분석',
        'role': '고객 세그먼트별 구매 패턴과 선호도를 분석합니다.'
    },
    'visualizer': {
        'name': '트렌드시각화',
        'description': '시각화 자료 생성',
        'role': '분석 결과를 효과적으로 전달하는 시각화 자료를 생성합니다.'
    },
    'report_generator': {
        'name': '리포트작성',
        'description': '최종 PDF 리포트 생성',
        'role': '분석 결과와 인사이트를 종합한 PDF 리포트를 생성합니다.'
    }
}

def get_sub_agent_info(agent_name=None):
    """
    Get information about trend analysis sub-agents.
    
    Args:
        agent_name: Optional sub-agent name to get specific information
        
    Returns:
        Dictionary with sub-agent information
    """
    if agent_name and agent_name in SUB_AGENTS:
        return SUB_AGENTS[agent_name]
    return SUB_AGENTS
