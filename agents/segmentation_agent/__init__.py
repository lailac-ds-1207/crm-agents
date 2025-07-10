"""
Segmentation Agent package for the CRM-Agent system.
This package contains components for analyzing customer data,
creating meaningful segments, and generating segmentation reports.
"""

# Package metadata
__version__ = '0.1.0'
__author__ = 'CDP AI Agent Team'
__description__ = 'Segmentation Agent for retail CDP data'

# Import main components from submodules
# These will be uncommented as the modules are implemented
# from .agent import SegmentationAgent
# from .graph import SegmentationGraph
# from .text2sql import Text2SQLConverter
# from .segment_analyzer import SegmentAnalyzer
# from .segment_validator import SegmentValidator
# from .segment_visualizer import SegmentVisualizer
# from .report_generator import SegmentationReportGenerator

# Sub-agent registry
SUB_AGENTS = {
    'text2sql': {
        'name': '텍스트변환',
        'description': '자연어를 SQL 쿼리로 변환',
        'role': '사용자의 자연어 요청을 BigQuery SQL로 변환합니다.'
    },
    'segment_analyzer': {
        'name': '세그먼트분석',
        'description': '고객 세그먼트 분석 및 특성 추출',
        'role': '생성된 세그먼트의 특성과 행동 패턴을 분석합니다.'
    },
    'segment_validator': {
        'name': '세그먼트검증',
        'description': '세그먼트 유효성 및 품질 검증',
        'role': '생성된 세그먼트의 크기, 안정성, 타겟팅 적합성을 검증합니다.'
    },
    'segment_visualizer': {
        'name': '시각화',
        'description': '세그먼트 시각화 및 차트 생성',
        'role': '세그먼트 분석 결과를 시각적으로 표현합니다.'
    },
    'report_generator': {
        'name': '리포트작성',
        'description': '세그먼테이션 결과 리포트 생성(PDF)',
        'role': '세그먼테이션 분석 결과를 종합한 PDF 리포트를 생성합니다.'
    }
}

def get_sub_agent_info(agent_name=None):
    """
    Get information about segmentation sub-agents.
    
    Args:
        agent_name: Optional sub-agent name to get specific information
        
    Returns:
        Dictionary with sub-agent information
    """
    if agent_name and agent_name in SUB_AGENTS:
        return SUB_AGENTS[agent_name]
    return SUB_AGENTS
