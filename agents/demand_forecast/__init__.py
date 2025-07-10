"""
Demand Forecast Agent package for the CRM-Agent system.
This package contains components for forecasting demand in CDP data,
training prediction models, and generating forecast reports.
"""

# Package metadata
__version__ = '0.1.0'
__author__ = 'CDP AI Agent Team'
__description__ = 'Demand Forecast Agent for retail CDP data'

# Import main components from submodules
# These will be uncommented as the modules are implemented
# from .agent import DemandForecastAgent
# from .graph import ForecastGraph
# from .data_preparer import DataPreparer
# from .model_selector import ModelSelector
# from .model_trainer import ModelTrainer
# from .forecaster import Forecaster
# from .result_analyzer import ResultAnalyzer
# from .report_generator import ForecastReportGenerator

# Sub-agent registry
SUB_AGENTS = {
    'data_preparer': {
        'name': '데이터준비',
        'description': '예측 모델링을 위한 시계열 데이터 전처리',
        'role': '예측에 적합한 형태로 시계열 데이터를 준비하고 전처리합니다.'
    },
    'model_selector': {
        'name': '모델선택',
        'description': '최적의 예측 모델 선정(ARIMA, Prophet, LSTM 등)',
        'role': '데이터 특성에 맞는 최적의 예측 모델을 선정합니다.'
    },
    'model_trainer': {
        'name': '모델학습',
        'description': '선택된 모델 학습 및 평가',
        'role': '선정된 예측 모델을 학습시키고 성능을 평가합니다.'
    },
    'forecaster': {
        'name': '예측실행',
        'description': '향후 기간 수요 예측 실행',
        'role': '학습된 모델을 사용하여 미래 기간의 수요를 예측합니다.'
    },
    'result_analyzer': {
        'name': '결과분석',
        'description': '예측 결과 분석 및 인사이트 도출',
        'role': '예측 결과를 분석하고 비즈니스 인사이트를 도출합니다.'
    },
    'report_generator': {
        'name': '리포트작성',
        'description': '예측 결과 리포트 생성(PDF)',
        'role': '예측 결과와 인사이트를 종합한 PDF 리포트를 생성합니다.'
    }
}

def get_sub_agent_info(agent_name=None):
    """
    Get information about demand forecast sub-agents.
    
    Args:
        agent_name: Optional sub-agent name to get specific information
        
    Returns:
        Dictionary with sub-agent information
    """
    if agent_name and agent_name in SUB_AGENTS:
        return SUB_AGENTS[agent_name]
    return SUB_AGENTS
