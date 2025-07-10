"""
Decision Agent package for the CRM-Agent system.
This package contains components for making strategic decisions based on
trend analysis and demand forecasting results.
"""

# Package metadata
__version__ = '0.1.0'
__author__ = 'CDP AI Agent Team'
__description__ = 'Decision Agent for retail CDP data'

# Import main components from submodules
# These will be uncommented as the modules are implemented
# from .agent import DecisionAgent
# from .graph import DecisionGraph
# from .evaluator import OptionEvaluator
# from .campaign_selector import CampaignSelector
# from .resource_allocator import ResourceAllocator
# from .strategy_advisor import StrategyAdvisor
# from .discussion_moderator import DiscussionModerator
# from .report_generator import DecisionReportGenerator

# Sub-agent registry
SUB_AGENTS = {
    'evaluator': {
        'name': '옵션평가',
        'description': '마케팅 캠페인 및 전략 옵션 평가',
        'role': '트렌드 분석과 수요 예측 결과를 바탕으로 다양한 마케팅 옵션을 평가합니다.'
    },
    'campaign_selector': {
        'name': '캠페인선택',
        'description': '최적의 마케팅 캠페인 선정',
        'role': '평가된 옵션 중에서 최적의 마케팅 캠페인을 선정합니다.'
    },
    'resource_allocator': {
        'name': '자원할당',
        'description': '예산 및 자원 최적 할당',
        'role': '선정된 캠페인에 예산과 자원을 최적으로 할당합니다.'
    },
    'strategy_advisor': {
        'name': '전략조언',
        'description': '마케팅 전략 및 실행 계획 제안',
        'role': '선정된 캠페인의 실행을 위한 구체적인 전략과 계획을 제안합니다.'
    },
    'discussion_moderator': {
        'name': '토론진행',
        'description': '에이전트 간 토론 진행 및 조율',
        'role': '여러 에이전트 간의 토론을 진행하고 의견을 조율합니다.'
    },
    'report_generator': {
        'name': '리포트작성',
        'description': '의사결정 결과 리포트 생성(PDF)',
        'role': '의사결정 과정과 결과를 종합한 PDF 리포트를 생성합니다.'
    }
}

def get_sub_agent_info(agent_name=None):
    """
    Get information about decision agent sub-agents.
    
    Args:
        agent_name: Optional sub-agent name to get specific information
        
    Returns:
        Dictionary with sub-agent information
    """
    if agent_name and agent_name in SUB_AGENTS:
        return SUB_AGENTS[agent_name]
    return SUB_AGENTS
