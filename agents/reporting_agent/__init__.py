"""
Reporting Agent package for the CRM-Agent system.
This package contains components for generating comprehensive reports
by combining outputs from trend analysis, demand forecasting, decision making,
and segmentation agents.
"""

# Package metadata
__version__ = '0.1.0'
__author__ = 'CDP AI Agent Team'
__description__ = 'Reporting Agent for retail CDP data'

# Import main components from submodules
# These will be uncommented as the modules are implemented
# from .agent import ReportingAgent
# from .graph import ReportingGraph
# from .report_compiler import ReportCompiler
# from .insight_extractor import InsightExtractor
# from .visualization_selector import VisualizationSelector
# from .executive_summarizer import ExecutiveSummarizer
# from .report_formatter import ReportFormatter
# from .report_distributor import ReportDistributor

# Sub-agent registry
SUB_AGENTS = {
    'compiler': {
        'name': '리포트컴파일러',
        'description': '다양한 에이전트의 결과를 종합',
        'role': '여러 에이전트의 분석 결과와 인사이트를 수집하고 통합합니다.'
    },
    'insight_extractor': {
        'name': '인사이트추출',
        'description': '주요 인사이트 추출 및 우선순위화',
        'role': '다양한 분석 결과에서 핵심 인사이트를 추출하고 우선순위를 정합니다.'
    },
    'visualization_selector': {
        'name': '시각화선택',
        'description': '최적의 시각화 요소 선택',
        'role': '리포트 목적에 맞는 최적의 시각화 요소를 선택합니다.'
    },
    'executive_summarizer': {
        'name': '요약작성',
        'description': '경영진용 요약 작성',
        'role': '경영진이 빠르게 이해할 수 있는 핵심 요약을 작성합니다.'
    },
    'report_formatter': {
        'name': '포맷팅',
        'description': '리포트 디자인 및 포맷팅',
        'role': '리포트의 디자인과 포맷을 전문적으로 구성합니다.'
    },
    'report_distributor': {
        'name': '배포관리',
        'description': '리포트 배포 및 접근 관리',
        'role': '완성된 리포트를 적절한 형식으로 저장하고 배포합니다.'
    }
}

def get_sub_agent_info(agent_name=None):
    """
    Get information about reporting sub-agents.
    
    Args:
        agent_name: Optional sub-agent name to get specific information
        
    Returns:
        Dictionary with sub-agent information
    """
    if agent_name and agent_name in SUB_AGENTS:
        return SUB_AGENTS[agent_name]
    return SUB_AGENTS
