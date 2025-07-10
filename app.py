
"""
CDP CRM-Agent System Web Application
This is the main Streamlit application for the CDP CRM-Agent system.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time

# Import agent modules
from agents.trend_analysis.agent import TrendAnalysisAgent
from agents.demand_forecast.agent import DemandForecastAgent
from agents.decision_agent.agent import DecisionAgent
from agents.segmentation_agent.agent import SegmentationAgent
from agents.segmentation_agent.text2sql import Text2SQLConverter
from agents.reporting_agent.agent import ReportingAgent

# Import utility modules
from utils.bigquery import BigQueryConnector
from utils.visualization import create_bar_chart, create_pie_chart, save_plotly_fig
import config

# Configure page
st.set_page_config(
    page_title="CDP CRM-Agent System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "bq_connector" not in st.session_state:
    st.session_state.bq_connector = None
if "current_agent" not in st.session_state:
    st.session_state.current_agent = None
if "trend_report_path" not in st.session_state:
    st.session_state.trend_report_path = None
if "forecast_report_path" not in st.session_state:
    st.session_state.forecast_report_path = None
if "decision_report_path" not in st.session_state:
    st.session_state.decision_report_path = None
if "segmentation_report_path" not in st.session_state:
    st.session_state.segmentation_report_path = None
if "comprehensive_report_path" not in st.session_state:
    st.session_state.comprehensive_report_path = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 0.3rem;
    }
    .info-box {
        background-color: #DBEAFE;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.3rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.3rem;
    }
    .agent-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        min-width: 150px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Authentication function
def authenticate():
    st.markdown('<div class="main-header">CDP CRM-Agent System</div>', unsafe_allow_html=True)
    
    with st.form("auth_form"):
        st.subheader("로그인")
        username = st.text_input("사용자 이름")
        password = st.text_input("비밀번호", type="password")
        project_id = st.text_input("GCP 프로젝트 ID", value=config.GCP_PROJECT_ID)
        dataset_id = st.text_input("BigQuery 데이터셋 ID", value=config.BQ_DATASET_ID)
        
        submit = st.form_submit_button("로그인")
        
        if submit:
            # Simple authentication (in production, use a more secure method)
            if username == "admin" and password == "admin":
                with st.spinner("BigQuery 연결 중..."):
                    try:
                        # Initialize BigQuery connector
                        bq_connector = BigQueryConnector(project_id=project_id)
                        
                        # Test connection
                        test_query = f"SELECT COUNT(*) as count FROM `{dataset_id}.customer_master`"
                        result = bq_connector.run_query(test_query)
                        
                        if result is not None and len(result) > 0:
                            st.session_state.authenticated = True
                            st.session_state.bq_connector = bq_connector
                            st.session_state.project_id = project_id
                            st.session_state.dataset_id = dataset_id
                            st.success("로그인 성공! BigQuery 연결 완료.")
                            st.rerun()
                        else:
                            st.error("BigQuery 연결 실패. 프로젝트 ID와 데이터셋 ID를 확인해주세요.")
                    except Exception as e:
                        st.error(f"오류 발생: {str(e)}")
            else:
                st.error("잘못된 사용자 이름 또는 비밀번호입니다.")

# Main application
def main():
    # Sidebar navigation
    st.sidebar.markdown('<div class="sub-header">CDP CRM-Agent</div>', unsafe_allow_html=True)
    
    # User info
    st.sidebar.markdown(f"""
    <div class="info-box">
        <b>프로젝트:</b> {st.session_state.project_id}<br>
        <b>데이터셋:</b> {st.session_state.dataset_id}
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    agent_options = {
        "dashboard": "🏠 대시보드",
        "trend": "📈 트렌드 분석",
        "forecast": "🔮 수요 예측",
        "decision": "🎯 의사결정",
        "segmentation": "👥 고객 세그먼테이션",
        "reporting": "📊 리포팅",
        "settings": "⚙️ 설정"
    }
    
    selected_agent = st.sidebar.radio("메뉴 선택", list(agent_options.values()))
    
    # Set current agent in session state
    for key, value in agent_options.items():
        if value == selected_agent:
            st.session_state.current_agent = key
            break
    
    # Display selected agent page
    if st.session_state.current_agent == "dashboard":
        show_dashboard()
    elif st.session_state.current_agent == "trend":
        show_trend_analysis()
    elif st.session_state.current_agent == "forecast":
        show_demand_forecast()
    elif st.session_state.current_agent == "decision":
        show_decision_agent()
    elif st.session_state.current_agent == "segmentation":
        show_segmentation_agent()
    elif st.session_state.current_agent == "reporting":
        show_reporting_agent()
    elif st.session_state.current_agent == "settings":
        show_settings()
    
    # Logout button
    if st.sidebar.button("로그아웃"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Dashboard page
def show_dashboard():
    st.markdown('<div class="main-header">CDP CRM-Agent 대시보드</div>', unsafe_allow_html=True)
    
    # Display summary of available reports
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">에이전트 현황</div>', unsafe_allow_html=True)
        
        # Check which reports are available
        agents_status = {
            "트렌드 분석": st.session_state.trend_report_path is not None,
            "수요 예측": st.session_state.forecast_report_path is not None,
            "의사결정": st.session_state.decision_report_path is not None,
            "고객 세그먼테이션": st.session_state.segmentation_report_path is not None,
            "종합 리포트": st.session_state.comprehensive_report_path is not None
        }
        
        # Display agent status
        for agent, status in agents_status.items():
            if status:
                st.markdown(f"""
                <div class="success-box">
                    <b>{agent}:</b> 리포트 생성 완료
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                    <b>{agent}:</b> 리포트 미생성
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sub-header">빠른 실행</div>', unsafe_allow_html=True)
        
        if st.button("🚀 모든 에이전트 실행", key="run_all_agents"):
            with st.spinner("모든 에이전트 실행 중..."):
                try:
                    # Run trend analysis
                    trend_agent = TrendAnalysisAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        bq_connector=st.session_state.bq_connector
                    )
                    st.session_state.trend_report_path = trend_agent.run()
                    st.success("트렌드 분석 완료")
                    
                    # Run demand forecast
                    forecast_agent = DemandForecastAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        bq_connector=st.session_state.bq_connector
                    )
                    st.session_state.forecast_report_path = forecast_agent.run()
                    st.success("수요 예측 완료")
                    
                    # Run decision agent
                    decision_agent = DecisionAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        trend_report_path=st.session_state.trend_report_path,
                        forecast_report_path=st.session_state.forecast_report_path,
                        bq_connector=st.session_state.bq_connector
                    )
                    st.session_state.decision_report_path = decision_agent.run()
                    st.success("의사결정 완료")
                    
                    # Run segmentation agent
                    segmentation_agent = SegmentationAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        bq_connector=st.session_state.bq_connector
                    )
                    st.session_state.segmentation_report_path = segmentation_agent.run()
                    st.success("고객 세그먼테이션 완료")
                    
                    # Run reporting agent for comprehensive report
                    reporting_agent = ReportingAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        report_type="comprehensive",
                        trend_report_path=st.session_state.trend_report_path,
                        forecast_report_path=st.session_state.forecast_report_path,
                        decision_report_path=st.session_state.decision_report_path,
                        segmentation_report_path=st.session_state.segmentation_report_path,
                        bq_connector=st.session_state.bq_connector
                    )
                    st.session_state.comprehensive_report_path = reporting_agent.run()
                    st.success("종합 리포트 생성 완료")
                    
                    st.success("모든 에이전트 실행 완료!")
                    
                except Exception as e:
                    st.error(f"에이전트 실행 중 오류 발생: {str(e)}")
        
        if st.button("📊 종합 리포트 생성", key="generate_comprehensive_report"):
            with st.spinner("종합 리포트 생성 중..."):
                try:
                    # Run reporting agent for comprehensive report
                    reporting_agent = ReportingAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        report_type="comprehensive",
                        trend_report_path=st.session_state.trend_report_path,
                        forecast_report_path=st.session_state.forecast_report_path,
                        decision_report_path=st.session_state.decision_report_path,
                        segmentation_report_path=st.session_state.segmentation_report_path,
                        bq_connector=st.session_state.bq_connector
                    )
                    st.session_state.comprehensive_report_path = reporting_agent.run()
                    st.success("종합 리포트 생성 완료!")
                    
                except Exception as e:
                    st.error(f"리포트 생성 중 오류 발생: {str(e)}")
    
    # Display key metrics
    st.markdown('<div class="sub-header">주요 지표</div>', unsafe_allow_html=True)
    
    # Fetch some basic metrics from BigQuery
    try:
        # Customer count
        customer_query = f"SELECT COUNT(*) as count FROM `{st.session_state.dataset_id}.customer_master`"
        customer_result = st.session_state.bq_connector.run_query(customer_query)
        customer_count = customer_result.iloc[0]['count'] if not customer_result.empty else 0
        
        # Transaction count
        transaction_query = f"SELECT COUNT(*) as count FROM `{st.session_state.dataset_id}.offline_transactions`"
        transaction_result = st.session_state.bq_connector.run_query(transaction_query)
        transaction_count = transaction_result.iloc[0]['count'] if not transaction_result.empty else 0
        
        # Total sales
        sales_query = f"SELECT SUM(total_amount) as total FROM `{st.session_state.dataset_id}.offline_transactions`"
        sales_result = st.session_state.bq_connector.run_query(sales_query)
        total_sales = sales_result.iloc[0]['total'] if not sales_result.empty else 0
        
        # Online sessions
        online_query = f"SELECT COUNT(DISTINCT session_id) as count FROM `{st.session_state.dataset_id}.online_behavior`"
        online_result = st.session_state.bq_connector.run_query(online_query)
        online_sessions = online_result.iloc[0]['count'] if not online_result.empty else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{customer_count:,}</div>
                <div class="metric-label">총 고객 수</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{transaction_count:,}</div>
                <div class="metric-label">총 거래 수</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_sales:,.0f}원</div>
                <div class="metric-label">총 매출액</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{online_sessions:,}</div>
                <div class="metric-label">온라인 세션 수</div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"지표 조회 중 오류 발생: {str(e)}")
    
    # Display available reports
    st.markdown('<div class="sub-header">생성된 리포트</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.trend_report_path:
            st.markdown(f"""
            <div class="card">
                <h3>트렌드 분석 리포트</h3>
                <p>생성 시간: {datetime.fromtimestamp(os.path.getctime(st.session_state.trend_report_path)).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists(st.session_state.trend_report_path):
                with open(st.session_state.trend_report_path, "rb") as file:
                    st.download_button(
                        label="리포트 다운로드",
                        data=file,
                        file_name="trend_analysis_report.pdf",
                        mime="application/pdf"
                    )
        
        if st.session_state.forecast_report_path:
            st.markdown(f"""
            <div class="card">
                <h3>수요 예측 리포트</h3>
                <p>생성 시간: {datetime.fromtimestamp(os.path.getctime(st.session_state.forecast_report_path)).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists(st.session_state.forecast_report_path):
                with open(st.session_state.forecast_report_path, "rb") as file:
                    st.download_button(
                        label="리포트 다운로드",
                        data=file,
                        file_name="demand_forecast_report.pdf",
                        mime="application/pdf"
                    )
    
    with col2:
        if st.session_state.decision_report_path:
            st.markdown(f"""
            <div class="card">
                <h3>의사결정 리포트</h3>
                <p>생성 시간: {datetime.fromtimestamp(os.path.getctime(st.session_state.decision_report_path)).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists(st.session_state.decision_report_path):
                with open(st.session_state.decision_report_path, "rb") as file:
                    st.download_button(
                        label="리포트 다운로드",
                        data=file,
                        file_name="decision_report.pdf",
                        mime="application/pdf"
                    )
        
        if st.session_state.segmentation_report_path:
            st.markdown(f"""
            <div class="card">
                <h3>고객 세그먼테이션 리포트</h3>
                <p>생성 시간: {datetime.fromtimestamp(os.path.getctime(st.session_state.segmentation_report_path)).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists(st.session_state.segmentation_report_path):
                with open(st.session_state.segmentation_report_path, "rb") as file:
                    st.download_button(
                        label="리포트 다운로드",
                        data=file,
                        file_name="segmentation_report.pdf",
                        mime="application/pdf"
                    )
    
    if st.session_state.comprehensive_report_path:
        st.markdown(f"""
        <div class="card">
            <h3>종합 리포트</h3>
            <p>생성 시간: {datetime.fromtimestamp(os.path.getctime(st.session_state.comprehensive_report_path)).strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists(st.session_state.comprehensive_report_path):
            with open(st.session_state.comprehensive_report_path, "rb") as file:
                st.download_button(
                    label="종합 리포트 다운로드",
                    data=file,
                    file_name="comprehensive_report.pdf",
                    mime="application/pdf"
                )

# Trend Analysis page
def show_trend_analysis():
    st.markdown('<div class="main-header">트렌드 분석 에이전트</div>', unsafe_allow_html=True)
    
    # Agent description
    st.markdown("""
    트렌드 분석 에이전트는 CDP 데이터를 분석하여 고객 행동, 제품 카테고리, 판매 채널 등의 트렌드를 파악합니다.
    이를 통해 비즈니스 의사결정에 필요한 인사이트를 제공합니다.
    """)
    
    # Configuration options
    st.markdown('<div class="sub-header">분석 설정</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        timeframe_days = st.slider(
            "분석 기간 (일)",
            min_value=30,
            max_value=365,
            value=180,
            step=30,
            help="트렌드 분석을 위한 과거 데이터 기간을 설정합니다."
        )
    
    # Run analysis button
    if st.button("트렌드 분석 실행", key="run_trend_analysis"):
        with st.spinner("트렌드 분석 중..."):
            try:
                # Run trend analysis agent
                trend_agent = TrendAnalysisAgent(
                    project_id=st.session_state.project_id,
                    dataset_id=st.session_state.dataset_id,
                    timeframe_days=timeframe_days,
                    bq_connector=st.session_state.bq_connector
                )
                
                report_path = trend_agent.run()
                st.session_state.trend_report_path = report_path
                
                # Get insights and visualizations
                insights = trend_agent.get_insights()
                visualizations = trend_agent.get_visualizations()
                
                # Display success message
                st.success("트렌드 분석이 완료되었습니다!")
                
                # Display insights
                st.markdown('<div class="sub-header">주요 인사이트</div>', unsafe_allow_html=True)
                
                for insight in insights:
                    st.markdown(f"- {insight}")
                
                # Display visualizations
                st.markdown('<div class="sub-header">시각화</div>', unsafe_allow_html=True)
                
                # Create tabs for different visualization categories
                viz_tabs = st.tabs(["판매 트렌드", "고객 행동", "제품 카테고리", "채널 분석"])
                
                # Filter visualizations by category
                sales_vizs = [v for v in visualizations if "판매" in v.get("title", "") or "매출" in v.get("title", "")]
                customer_vizs = [v for v in visualizations if "고객" in v.get("title", "") or "구매자" in v.get("title", "")]
                product_vizs = [v for v in visualizations if "제품" in v.get("title", "") or "카테고리" in v.get("title", "")]
                channel_vizs = [v for v in visualizations if "채널" in v.get("title", "") or "온라인" in v.get("title", "") or "오프라인" in v.get("title", "")]
                
                # Display visualizations in tabs
                with viz_tabs[0]:  # Sales trends
                    for viz in sales_vizs:
                        if "path" in viz and os.path.exists(viz["path"]):
                            st.image(viz["path"], caption=viz.get("title", ""))
                            st.markdown(viz.get("description", ""))
                
                with viz_tabs[1]:  # Customer behavior
                    for viz in customer_vizs:
                        if "path" in viz and os.path.exists(viz["path"]):
                            st.image(viz["path"], caption=viz.get("title", ""))
                            st.markdown(viz.get("description", ""))
                
                with viz_tabs[2]:  # Product categories
                    for viz in product_vizs:
                        if "path" in viz and os.path.exists(viz["path"]):
                            st.image(viz["path"], caption=viz.get("title", ""))
                            st.markdown(viz.get("description", ""))
                
                with viz_tabs[3]:  # Channel analysis
                    for viz in channel_vizs:
                        if "path" in viz and os.path.exists(viz["path"]):
                            st.image(viz["path"], caption=viz.get("title", ""))
                            st.markdown(viz.get("description", ""))
                
                # Display report download button
                st.markdown('<div class="sub-header">리포트 다운로드</div>', unsafe_allow_html=True)
                
                if os.path.exists(report_path):
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="트렌드 분석 리포트 다운로드",
                            data=file,
                            file_name="trend_analysis_report.pdf",
                            mime="application/pdf"
                        )
                
            except Exception as e:
                st.error(f"트렌드 분석 중 오류 발생: {str(e)}")
    
    # Display existing report if available
    elif st.session_state.trend_report_path and os.path.exists(st.session_state.trend_report_path):
        st.markdown('<div class="sub-header">기존 트렌드 분석 리포트</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            기존 트렌드 분석 리포트가 있습니다. 생성 시간: {datetime.fromtimestamp(os.path.getctime(st.session_state.trend_report_path)).strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
        
        with open(st.session_state.trend_report_path, "rb") as file:
            st.download_button(
                label="기존 리포트 다운로드",
                data=file,
                file_name="trend_analysis_report.pdf",
                mime="application/pdf"
            )
        
        if st.button("새 분석 실행", key="run_new_trend_analysis"):
            st.session_state.trend_report_path = None
            st.rerun()
    
    # Custom trend analysis
    st.markdown('<div class="sub-header">커스텀 트렌드 분석</div>', unsafe_allow_html=True)
    
    custom_question = st.text_input(
        "트렌드에 대한 질문을 입력하세요",
        placeholder="예: 지난 3개월 동안 가장 성장한 제품 카테고리는 무엇인가요?"
    )
    
    if custom_question and st.button("질문 분석", key="analyze_custom_question"):
        with st.spinner("질문 분석 중..."):
            try:
                # Initialize trend agent if not already done
                if not hasattr(st.session_state, 'trend_agent'):
                    st.session_state.trend_agent = TrendAnalysisAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        bq_connector=st.session_state.bq_connector
                    )
                    
                    # Run if no report exists
                    if not st.session_state.trend_report_path:
                        st.session_state.trend_report_path = st.session_state.trend_agent.run()
                
                # Generate custom insight
                custom_insight = st.session_state.trend_agent.generate_custom_insight(custom_question)
                
                # Display result
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**질문:** {custom_question}")
                st.markdown(f"**답변:** {custom_insight}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"질문 분석 중 오류 발생: {str(e)}")

# Demand Forecast page
def show_demand_forecast():
    st.markdown('<div class="main-header">수요 예측 에이전트</div>', unsafe_allow_html=True)
    
    # Agent description
    st.markdown("""
    수요 예측 에이전트는 과거 판매 데이터를 분석하여 미래 수요를 예측합니다.
    시계열 분석과 머신러닝 모델을 활용하여 제품 카테고리별 수요 예측과 성장률을 제공합니다.
    """)
    
    # Configuration options
    st.markdown('<div class="sub-header">예측 설정</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_horizon_weeks = st.slider(
            "예측 기간 (주)",
            min_value=4,
            max_value=52,
            value=12,
            step=4,
            help="미래 몇 주 동안의 수요를 예측할지 설정합니다."
        )
    
    # Run forecast button
    if st.button("수요 예측 실행", key="run_demand_forecast"):
        with st.spinner("수요 예측 중..."):
            try:
                # Run demand forecast agent
                forecast_agent = DemandForecastAgent(
                    project_id=st.session_state.project_id,
                    dataset_id=st.session_state.dataset_id,
                    forecast_horizon_weeks=forecast_horizon_weeks,
                    bq_connector=st.session_state.bq_connector
                )
                
                report_path = forecast_agent.run()
                st.session_state.forecast_report_path = report_path
                
                # Get insights, visualizations, and growth rates
                insights = forecast_agent.get_insights()
                visualizations = forecast_agent.get_visualizations()
                growth_rates = forecast_agent.get_growth_rates()
                seasonality = forecast_agent.get_seasonality()
                
                # Display success message
                st.success("수요 예측이 완료되었습니다!")
                
                # Display growth rates
                st.markdown('<div class="sub-header">예측 성장률</div>', unsafe_allow_html=True)
                
                if growth_rates and "total_growth_rate" in growth_rates:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{growth_rates['total_growth_rate']:.2f}%</div>
                        <div class="metric-label">총 예상 성장률 (향후 {forecast_horizon_weeks}주)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if growth_rates and "category_growth" in growth_rates and not growth_rates["category_growth"].empty:
                    # Display category growth rates
                    category_growth = growth_rates["category_growth"]
                    
                    # Create bar chart for category growth rates
                    fig = px.bar(
                        category_growth,
                        x="category",
                        y="growth_rate",
                        title=f"카테고리별 예상 성장률 (향후 {forecast_horizon_weeks}주)",
                        labels={"category": "카테고리", "growth_rate": "성장률 (%)"},
                        color="growth_rate",
                        color_continuous_scale="RdYlGn",
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display insights
                st.markdown('<div class="sub-header">주요 인사이트</div>', unsafe_allow_html=True)
                
                for insight in insights:
                    st.markdown(f"- {insight}")
                
                # Display visualizations
                st.markdown('<div class="sub-header">예측 시각화</div>', unsafe_allow_html=True)
                
                # Create tabs for different visualization categories
                viz_tabs = st.tabs(["총 매출 예측", "카테고리별 예측", "계절성 분석"])
                
                # Filter visualizations by category
                total_vizs = [v for v in visualizations if "총 매출" in v.get("title", "")]
                category_vizs = [v for v in visualizations if "카테고리" in v.get("title", "")]
                seasonality_vizs = [v for v in visualizations if "계절성" in v.get("title", "") or "요일별" in v.get("title", "") or "월별" in v.get("title", "")]
                
                # Display visualizations in tabs
                with viz_tabs[0]:  # Total sales forecast
                    for viz in total_vizs:
                        if "path" in viz and os.path.exists(viz["path"]):
                            st.image(viz["path"], caption=viz.get("title", ""))
                            st.markdown(viz.get("description", ""))
                
                with viz_tabs[1]:  # Category forecasts
                    for viz in category_vizs:
                        if "path" in viz and os.path.exists(viz["path"]):
                            st.image(viz["path"], caption=viz.get("title", ""))
                            st.markdown(viz.get("description", ""))
                
                with viz_tabs[2]:  # Seasonality analysis
                    for viz in seasonality_vizs:
                        if "path" in viz and os.path.exists(viz["path"]):
                            st.image(viz["path"], caption=viz.get("title", ""))
                            st.markdown(viz.get("description", ""))
                
                # Display report download button
                st.markdown('<div class="sub-header">리포트 다운로드</div>', unsafe_allow_html=True)
                
                if os.path.exists(report_path):
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="수요 예측 리포트 다운로드",
                            data=file,
                            file_name="demand_forecast_report.pdf",
                            mime="application/pdf"
                        )
                
            except Exception as e:
                st.error(f"수요 예측 중 오류 발생: {str(e)}")
    
    # Display existing report if available
    elif st.session_state.forecast_report_path and os.path.exists(st.session_state.forecast_report_path):
        st.markdown('<div class="sub-header">기존 수요 예측 리포트</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            기존 수요 예측 리포트가 있습니다. 생성 시간: {datetime.fromtimestamp(os.path.getctime(st.session_state.forecast_report_path)).strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
        
        with open(st.session_state.forecast_report_path, "rb") as file:
            st.download_button(
                label="기존 리포트 다운로드",
                data=file,
                file_name="demand_forecast_report.pdf",
                mime="application/pdf"
            )
        
        if st.button("새 예측 실행", key="run_new_forecast"):
            st.session_state.forecast_report_path = None
            st.rerun()
    
    # Category-specific forecast
    st.markdown('<div class="sub-header">카테고리별 예측</div>', unsafe_allow_html=True)
    
    # Get available categories
    try:
        category_query = f"""
        SELECT DISTINCT category_level_1
        FROM `{st.session_state.dataset_id}.product_master`
        ORDER BY category_level_1
        """
        categories = st.session_state.bq_connector.run_query(category_query)
        
        if not categories.empty:
            selected_category = st.selectbox(
                "카테고리 선택",
                options=categories['category_level_1'].tolist()
            )
            
            if selected_category and st.button("카테고리 예측 분석", key="analyze_category_forecast"):
                with st.spinner(f"{selected_category} 카테고리 예측 분석 중..."):
                    try:
                        # Initialize forecast agent if not already done
                        if not hasattr(st.session_state, 'forecast_agent'):
                            st.session_state.forecast_agent = DemandForecastAgent(
                                project_id=st.session_state.project_id,
                                dataset_id=st.session_state.dataset_id,
                                bq_connector=st.session_state.bq_connector
                            )
                            
                            # Run if no report exists
                            if not st.session_state.forecast_report_path:
                                st.session_state.forecast_report_path = st.session_state.forecast_agent.run()
                        
                        # Get category forecast
                        category_forecast = st.session_state.forecast_agent.get_category_forecast(selected_category)
                        
                        # Display category forecast
                        st.markdown(f"### {selected_category} 카테고리 예측")
                        
                        # Display growth rate
                        if "growth_rate" in category_forecast and category_forecast["growth_rate"] is not None:
                            growth_color = "green" if category_forecast["growth_rate"] > 0 else "red"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: {growth_color};">{category_forecast["growth_rate"]:.2f}%</div>
                                <div class="metric-label">예상 성장률</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display visualization if available
                        if "visualization" in category_forecast and category_forecast["visualization"] and os.path.exists(category_forecast["visualization"]):
                            st.image(category_forecast["visualization"], caption=f"{selected_category} 카테고리 예측")
                        
                        # Display insights
                        if "insights" in category_forecast and category_forecast["insights"]:
                            st.markdown("### 카테고리 인사이트")
                            st.markdown(category_forecast["insights"])
                        
                    except Exception as e:
                        st.error(f"카테고리 예측 분석 중 오류 발생: {str(e)}")
        
    except Exception as e:
        st.error(f"카테고리 정보 로딩 중 오류 발생: {str(e)}")

# Decision Agent page
def show_decision_agent():
    st.markdown('<div class="main-header">의사결정 에이전트</div>', unsafe_allow_html=True)
    
    # Agent description
    st.markdown("""
    의사결정 에이전트는 트렌드 분석과 수요 예측 결과를 바탕으로 최적의 마케팅 캠페인을 선정하고,
    리소스 할당 및 실행 전략을 제안합니다. 다중 에이전트 토론을 통해 의사결정을 진행합니다.
    """)
    
    # Check if required reports exist
    trend_report_exists = st.session_state.trend_report_path and os.path.exists(st.session_state.trend_report_path)
    forecast_report_exists = st.session_state.forecast_report_path and os.path.exists(st.session_state.forecast_report_path)
    
    if not trend_report_exists or not forecast_report_exists:
        st.warning("의사결정 에이전트를 실행하기 위해서는 트렌드 분석과 수요 예측이 먼저 완료되어야 합니다.")
        
        missing_reports = []
        if not trend_report_exists:
            missing_reports.append("트렌드 분석")
        if not forecast_report_exists:
            missing_reports.append("수요 예측")
        
        st.markdown(f"다음 분석을 먼저 실행해주세요: {', '.join(missing_reports)}")
        
        # Add buttons to run missing reports
        col1, col2 = st.columns(2)
        
        if not trend_report_exists:
            with col1:
                if st.button("트렌드 분석 실행", key="run_trend_for_decision"):
                    st.session_state.current_agent = "trend"
                    st.rerun()
        
        if not forecast_report_exists:
            with col2:
                if st.button("수요 예측 실행", key="run_forecast_for_decision"):
                    st.session_state.current_agent = "forecast"
                    st.rerun()
    else:
        # Run decision agent button
        if st.button("의사결정 프로세스 실행", key="run_decision_process"):
            with st.spinner("의사결정 프로세스 실행 중..."):
                try:
                    # Run decision agent
                    decision_agent = DecisionAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        trend_report_path=st.session_state.trend_report_path,
                        forecast_report_path=st.session_state.forecast_report_path,
                        bq_connector=st.session_state.bq_connector
                    )
                    
                    report_path = decision_agent.run()
                    st.session_state.decision_report_path = report_path
                    
                    # Get selected campaigns, resource allocation, and strategic advice
                    selected_campaigns = decision_agent.get_selected_campaigns()
                    resource_allocation = decision_agent.get_resource_allocation()
                    strategic_advice = decision_agent.get_strategic_advice()
                    discussion_summary = decision_agent.get_discussion_summary()
                    
                    # Display success message
                    st.success("의사결정 프로세스가 완료되었습니다!")
                    
                    # Display selected campaigns
                    st.markdown('<div class="sub-header">선정된 마케팅 캠페인</div>', unsafe_allow_html=True)
                    
                    for i, campaign in enumerate(selected_campaigns):
                        st.markdown(f"""
                        <div class="card">
                            <h3>{i+1}. {campaign.get('name', '캠페인')}</h3>
                            <p><b>목표:</b> {campaign.get('objective', 'N/A')}</p>
                            <p><b>타겟 고객:</b> {campaign.get('target', 'N/A')}</p>
                            <p><b>주요 제품/카테고리:</b> {campaign.get('products', 'N/A')}</p>
                            <p><b>채널 전략:</b> {campaign.get('channels', 'N/A')}</p>
                            <p><b>예상 ROI:</b> {campaign.get('roi', 'N/A')}</p>
                            <p><b>실행 기간:</b> {campaign.get('duration', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display resource allocation
                    st.markdown('<div class="sub-header">리소스 할당</div>', unsafe_allow_html=True)
                    
                    if resource_allocation and "campaigns" in resource_allocation:
                        for allocation in resource_allocation["campaigns"]:
                            st.markdown(f"""
                            <div class="card">
                                <h3>{allocation.get('name', '캠페인')}</h3>
                                <p><b>예산:</b> {allocation.get('budget', 'N/A')}</p>
                                <p><b>인력:</b> {allocation.get('staff', 'N/A')}</p>
                                <p><b>채널 배분:</b> {allocation.get('channels', 'N/A')}</p>
                                <p><b>실행 일정:</b> {allocation.get('timeline', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Display strategic advice
                    st.markdown('<div class="sub-header">전략적 조언</div>', unsafe_allow_html=True)
                    
                    for advice in strategic_advice:
                        st.markdown(f"- {advice}")
                    
                    # Display discussion summary
                    st.markdown('<div class="sub-header">전문가 토론 요약</div>', unsafe_allow_html=True)
                    
                    # Find moderator summary
                    moderator_summary = next((entry for entry in discussion_summary if "요약" in entry["role"]), None)
                    
                    if moderator_summary:
                        st.markdown(moderator_summary["content"])
                    
                    # Option to view full discussion
                    if st.checkbox("전체 토론 내용 보기"):
                        for entry in discussion_summary:
                            st.markdown(f"""
                            <div class="card">
                                <h4>{entry['role']}</h4>
                                <p>{entry['content']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Display report download button
                    st.markdown('<div class="sub-header">리포트 다운로드</div>', unsafe_allow_html=True)
                    
                    if os.path.exists(report_path):
                        with open(report_path, "rb") as file:
                            st.download_button(
                                label="의사결정 리포트 다운로드",
                                data=file,
                                file_name="decision_report.pdf",
                                mime="application/pdf"
                            )
                    
                except Exception as e:
                    st.error(f"의사결정 프로세스 실행 중 오류 발생: {str(e)}")
        
        # Display existing report if available
        elif st.session_state.decision_report_path and os.path.exists(st.session_state.decision_report_path):
            st.markdown('<div class="sub-header">기존 의사결정 리포트</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-box">
                기존 의사결정 리포트가 있습니다. 생성 시간: {datetime.fromtimestamp(os.path.getctime(st.session_state.decision_report_path)).strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
            
            with open(st.session_state.decision_report_path, "rb") as file:
                st.download_button(
                    label="기존 리포트 다운로드",
                    data=file,
                    file_name="decision_report.pdf",
                    mime="application/pdf"
                )
            
            if st.button("새 의사결정 프로세스 실행", key="run_new_decision"):
                st.session_state.decision_report_path = None
                st.rerun()
        
        # Custom strategic advice
        st.markdown('<div class="sub-header">커스텀 전략 조언</div>', unsafe_allow_html=True)
        
        custom_question = st.text_input(
            "전략에 대한 질문을 입력하세요",
            placeholder="예: 신규 고객 유치를 위한 최적의 마케팅 채널은 무엇인가요?"
        )
        
        if custom_question and st.button("전략 조언 요청", key="get_custom_advice"):
            with st.spinner("전략 조언 생성 중..."):
                try:
                    # Initialize decision agent if not already done
                    if not hasattr(st.session_state, 'decision_agent'):
                        st.session_state.decision_agent = DecisionAgent(
                            project_id=st.session_state.project_id,
                            dataset_id=st.session_state.dataset_id,
                            trend_report_path=st.session_state.trend_report_path,
                            forecast_report_path=st.session_state.forecast_report_path,
                            bq_connector=st.session_state.bq_connector
                        )
                        
                        # Run if no report exists
                        if not st.session_state.decision_report_path:
                            st.session_state.decision_report_path = st.session_state.decision_agent.run()
                    
                    # Generate custom advice
                    custom_advice = st.session_state.decision_agent.generate_custom_advice(custom_question)
                    
                    # Display result
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"**질문:** {custom_question}")
                    st.markdown(f"**전략 조언:**\n\n{custom_advice}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"전략 조언 생성 중 오류 발생: {str(e)}")

# Segmentation Agent page
def show_segmentation_agent():
    st.markdown('<div class="main-header">고객 세그먼테이션 에이전트</div>', unsafe_allow_html=True)
    
    # Agent description
    st.markdown("""
    고객 세그먼테이션 에이전트는 CDP 데이터를 분석하여 의미 있는 고객 세그먼트를 생성합니다.
    RFM, 라이프사이클, 채널 선호도, 카테고리 선호도 등 다양한 관점에서 고객을 세분화하고,
    각 세그먼트의 특성과 마케팅 접근법을 제안합니다.
    """)
    
    # Create tabs for different segmentation approaches
    seg_tabs = st.tabs(["기본 세그먼테이션", "Text2SQL 세그먼테이션", "세그먼트 비교"])
    
    # Tab 1: Default Segmentation
    with seg_tabs[0]:
        st.markdown('<div class="sub-header">기본 세그먼테이션</div>', unsafe_allow_html=True)
        
        # Run segmentation button
        if st.button("세그먼테이션 실행", key="run_segmentation"):
            with st.spinner("고객 세그먼테이션 실행 중..."):
                try:
                    # Run segmentation agent
                    segmentation_agent = SegmentationAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        bq_connector=st.session_state.bq_connector
                    )
                    
                    report_path = segmentation_agent.run()
                    st.session_state.segmentation_report_path = report_path
                    
                    # Get segments, analysis, and visualizations
                    segments = segmentation_agent.get_segments()
                    segment_analysis = segmentation_agent.get_segment_analysis()
                    visualizations = segmentation_agent.get_visualizations()
                    
                    # Display success message
                    st.success("고객 세그먼테이션이 완료되었습니다!")
                    
                    # Display segment types
                    st.markdown('<div class="sub-header">세그먼트 유형</div>', unsafe_allow_html=True)
                    
                    segment_types = list(segments.keys())
                    
                    # Create segment type selection
                    selected_segment_type = st.selectbox(
                        "세그먼트 유형 선택",
                        options=segment_types,
                        format_func=lambda x: {
                            "rfm": "RFM 세그먼테이션",
                            "lifecycle": "라이프사이클 세그먼테이션",
                            "channel": "채널 선호도 세그먼테이션",
                            "category": "카테고리 선호도 세그먼테이션"
                        }.get(x, x)
                    )
                    
                    if selected_segment_type:
                        # Display segment distribution visualization
                        segment_viz = next((v for v in visualizations if v["segment_type"] == selected_segment_type and "분포" in v["title"]), None)
                        if segment_viz and "path" in segment_viz and os.path.exists(segment_viz["path"]):
                            st.image(segment_viz["path"], caption=segment_viz.get("title", ""))
                        
                        # Display segment counts
                        if selected_segment_type in segment_analysis and "segment_counts" in segment_analysis[selected_segment_type]:
                            counts = segment_analysis[selected_segment_type]["segment_counts"]
                            
                            # Create DataFrame for display
                            counts_df = pd.DataFrame({
                                "세그먼트": list(counts.keys()),
                                "고객 수": list(counts.values()),
                                "비율 (%)": [count / sum(counts.values()) * 100 for count in counts.values()]
                            })
                            
                            st.dataframe(counts_df, use_container_width=True)
                        
                        # Display segment insights
                        if selected_segment_type in segment_analysis and "insights" in segment_analysis[selected_segment_type]:
                            st.markdown('<div class="sub-header">세그먼트 인사이트</div>', unsafe_allow_html=True)
                            st.markdown(segment_analysis[selected_segment_type]["insights"])
                        
                        # Display additional visualizations
                        additional_vizs = [v for v in visualizations if v["segment_type"] == selected_segment_type and "분포" not in v["title"]]
                        
                        if additional_vizs:
                            st.markdown('<div class="sub-header">세그먼트 시각화</div>', unsafe_allow_html=True)
                            
                            for viz in additional_vizs:
                                if "path" in viz and os.path.exists(viz["path"]):
                                    st.image(viz["path"], caption=viz.get("title", ""))
                                    st.markdown(viz.get("description", ""))
                    
                    # Display report download button
                    st.markdown('<div class="sub-header">리포트 다운로드</div>', unsafe_allow_html=True)
                    
                    if os.path.exists(report_path):
                        with open(report_path, "rb") as file:
                            st.download_button(
                                label="세그먼테이션 리포트 다운로드",
                                data=file,
                                file_name="segmentation_report.pdf",
                                mime="application/pdf"
                            )
                    
                except Exception as e:
                    st.error(f"세그먼테이션 실행 중 오류 발생: {str(e)}")
        
        # Display existing report if available
        elif st.session_state.segmentation_report_path and os.path.exists(st.session_state.segmentation_report_path):
            st.markdown('<div class="sub-header">기존 세그먼테이션 리포트</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-box">
                기존 세그먼테이션 리포트가 있습니다. 생성 시간: {datetime.fromtimestamp(os.path.getctime(st.session_state.segmentation_report_path)).strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
            
            with open(st.session_state.segmentation_report_path, "rb") as file:
                st.download_button(
                    label="기존 리포트 다운로드",
                    data=file,
                    file_name="segmentation_report.pdf",
                    mime="application/pdf"
                )
            
            if st.button("새 세그먼테이션 실행", key="run_new_segmentation"):
                st.session_state.segmentation_report_path = None
                st.rerun()
    
    # Tab 2: Text2SQL Segmentation
    with seg_tabs[1]:
        st.markdown('<div class="sub-header">Text2SQL 세그먼테이션</div>', unsafe_allow_html=True)
        
        st.markdown("""
        자연어로 세그먼테이션 요청을 입력하면, Text2SQL 변환기가 이를 SQL 쿼리로 변환하여
        원하는 고객 세그먼트를 생성합니다.
        """)
        
        # Text input for segmentation request
        segmentation_request = st.text_area(
            "세그먼테이션 요청 (자연어)",
            placeholder="예: 최근 3개월 내에 TV 카테고리 제품을 구매한 30-40대 여성 고객을 찾아주세요.",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("SQL 생성", key="generate_sql"):
                if segmentation_request:
                    with st.spinner("SQL 쿼리 생성 중..."):
                        try:
                            # Initialize Text2SQL converter
                            text2sql = Text2SQLConverter(
                                project_id=st.session_state.project_id,
                                dataset_id=st.session_state.dataset_id,
                                bq_connector=st.session_state.bq_connector
                            )
                            
                            # Convert text to SQL
                            sql_query = text2sql.convert(segmentation_request)
                            
                            # Store in session state
                            st.session_state.current_sql_query = sql_query
                            
                            # Get explanation
                            explanation = text2sql.explain_query(sql_query)
                            st.session_state.current_sql_explanation = explanation
                            
                            # Display SQL query
                            st.markdown('<div class="sub-header">생성된 SQL 쿼리</div>', unsafe_allow_html=True)
                            st.code(sql_query, language="sql")
                            
                            # Display explanation
                            st.markdown('<div class="sub-header">쿼리 설명</div>', unsafe_allow_html=True)
                            st.markdown(explanation)
                            
                        except Exception as e:
                            st.error(f"SQL 생성 중 오류 발생: {str(e)}")
                else:
                    st.warning("세그먼테이션 요청을 입력해주세요.")
        
        with col2:
            if st.button("세그먼트 생성", key="create_segment"):
                if hasattr(st.session_state, 'current_sql_query') and st.session_state.current_sql_query:
                    with st.spinner("세그먼트 생성 중..."):
                        try:
                            # Initialize segmentation agent
                            if not hasattr(st.session_state, 'segmentation_agent'):
                                st.session_state.segmentation_agent = SegmentationAgent(
                                    project_id=st.session_state.project_id,
                                    dataset_id=st.session_state.dataset_id,
                                    bq_connector=st.session_state.bq_connector
                                )
                            
                            # Create custom segment
                            segment_name = "custom"
                            segment_data = st.session_state.segmentation_agent.create_custom_segment(
                                segment_name,
                                st.session_state.current_sql_query
                            )
                            
                            # Display segment data
                            st.markdown('<div class="sub-header">생성된 세그먼트</div>', unsafe_allow_html=True)
                            st.markdown(f"**고객 수:** {len(segment_data)}")
                            
                            # Display sample data
                            st.markdown('<div class="sub-header">샘플 데이터</div>', unsafe_allow_html=True)
                            st.dataframe(segment_data.head(10), use_container_width=True)
                            
                            # Store segment data in session state
                            st.session_state.current_segment_data = segment_data
                            
                        except Exception as e:
                            st.error(f"세그먼트 생성 중 오류 발생: {str(e)}")
                else:
                    st.warning("먼저 SQL을 생성해주세요.")
        
        # Display segment data if available
        if hasattr(st.session_state, 'current_segment_data') and not st.session_state.current_segment_data.empty:
            st.markdown('<div class="sub-header">세그먼트 분석</div>', unsafe_allow_html=True)
            
            # Basic statistics
            st.markdown("### 기본 통계")
            
            # Select numeric columns for statistics
            numeric_cols = st.session_state.current_segment_data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                stats_df = st.session_state.current_segment_data[numeric_cols].describe().T
                st.dataframe(stats_df, use_container_width=True)
            
            # Download segment data as CSV
            st.download_button(
                label="세그먼트 데이터 CSV 다운로드",
                data=st.session_state.current_segment_data.to_csv(index=False).encode('utf-8'),
                file_name="custom_segment.csv",
                mime="text/csv"
            )
    
    # Tab 3: Segment Comparison
    with seg_tabs[2]:
        st.markdown('<div class="sub-header">세그먼트 비교</div>', unsafe_allow_html=True)
        
        st.markdown("""
        서로 다른 세그먼테이션 방법으로 생성된 세그먼트들을 비교하여
        고객 그룹 간의 관계와 중첩을 분석합니다.
        """)
        
        # Check if segmentation has been run
        if not st.session_state.segmentation_report_path:
            st.warning("세그먼트 비교를 위해 먼저 기본 세그먼테이션을 실행해주세요.")
            
            if st.button("세그먼테이션 실행하기", key="run_segmentation_for_comparison"):
                st.session_state.current_agent = "segmentation"
                st.rerun()
        else:
            # Initialize segmentation agent if not already done
            if not hasattr(st.session_state, 'segmentation_agent'):
                try:
                    st.session_state.segmentation_agent = SegmentationAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        bq_connector=st.session_state.bq_connector
                    )
                except Exception as e:
                    st.error(f"세그먼테이션 에이전트 초기화 중 오류 발생: {str(e)}")
            
            # Get available segment types
            try:
                segments = st.session_state.segmentation_agent.get_segments()
                segment_types = list(segments.keys())
                
                # Create segment type selection
                col1, col2 = st.columns(2)
                
                with col1:
                    segment_type1 = st.selectbox(
                        "첫 번째 세그먼트 유형",
                        options=segment_types,
                        format_func=lambda x: {
                            "rfm": "RFM 세그먼테이션",
                            "lifecycle": "라이프사이클 세그먼테이션",
                            "channel": "채널 선호도 세그먼테이션",
                            "category": "카테고리 선호도 세그먼테이션",
                            "custom": "커스텀 세그먼테이션"
                        }.get(x, x)
                    )
                
                with col2:
                    # Filter out the first selection
                    remaining_types = [t for t in segment_types if t != segment_type1]
                    segment_type2 = st.selectbox(
                        "두 번째 세그먼트 유형",
                        options=remaining_types,
                        format_func=lambda x: {
                            "rfm": "RFM 세그먼테이션",
                            "lifecycle": "라이프사이클 세그먼테이션",
                            "channel": "채널 선호도 세그먼테이션",
                            "category": "카테고리 선호도 세그먼테이션",
                            "custom": "커스텀 세그먼테이션"
                        }.get(x, x)
                    )
                
                if st.button("세그먼트 비교", key="compare_segments"):
                    with st.spinner("세그먼트 비교 중..."):
                        try:
                            # Compare segments
                            comparison_result = st.session_state.segmentation_agent.compare_segments(
                                segment_type1,
                                segment_type2
                            )
                            
                            # Display comparison visualization
                            st.markdown('<div class="sub-header">세그먼트 비교 결과</div>', unsafe_allow_html=True)
                            
                            if "visualization_path" in comparison_result and os.path.exists(comparison_result["visualization_path"]):
                                st.image(comparison_result["visualization_path"], caption=f"{segment_type1} vs {segment_type2} 세그먼트 비교")
                            
                            # Display insights
                            if "insights" in comparison_result:
                                st.markdown('<div class="sub-header">비교 인사이트</div>', unsafe_allow_html=True)
                                st.markdown(comparison_result["insights"])
                            
                            # Display crosstab if available
                            if "crosstab" in comparison_result and not comparison_result["crosstab"].empty:
                                st.markdown('<div class="sub-header">세그먼트 교차표 (행 기준 %)</div>', unsafe_allow_html=True)
                                st.dataframe(comparison_result["crosstab"], use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"세그먼트 비교 중 오류 발생: {str(e)}")
                
            except Exception as e:
                st.error(f"세그먼트 정보 로딩 중 오류 발생: {str(e)}")

# Reporting Agent page
def show_reporting_agent():
    st.markdown('<div class="main-header">리포팅 에이전트</div>', unsafe_allow_html=True)
    
    # Agent description
    st.markdown("""
    리포팅 에이전트는 다양한 분석 결과를 종합하여 목적에 맞는 리포트를 생성합니다.
    트렌드 분석, 수요 예측, 의사결정, 세그먼테이션 결과를 통합하여
    경영진용, 마케팅용, 운영용 등 다양한 리포트를 제공합니다.
    """)
    
    # Report type selection
    report_type = st.selectbox(
        "리포트 유형",
        options=["comprehensive", "executive", "marketing", "operations"],
        format_func=lambda x: {
            "comprehensive": "종합 리포트",
            "executive": "경영진용 요약 리포트",
            "marketing": "마케팅 전략 리포트",
            "operations": "운영 계획 리포트"
        }.get(x, x)
    )
    
    # Check which reports are available
    trend_report_exists = st.session_state.trend_report_path and os.path.exists(st.session_state.trend_report_path)
    forecast_report_exists = st.session_state.forecast_report_path and os.path.exists(st.session_state.forecast_report_path)
    decision_report_exists = st.session_state.decision_report_path and os.path.exists(st.session_state.decision_report_path)
    segmentation_report_exists = st.session_state.segmentation_report_path and os.path.exists(st.session_state.segmentation_report_path)
    
    # Display available reports
    st.markdown('<div class="sub-header">사용 가능한 리포트</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="{'success-box' if trend_report_exists else 'warning-box'}">
            <b>트렌드 분석:</b> {'사용 가능' if trend_report_exists else '미생성'}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="{'success-box' if forecast_report_exists else 'warning-box'}">
            <b>수요 예측:</b> {'사용 가능' if forecast_report_exists else '미생성'}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="{'success-box' if decision_report_exists else 'warning-box'}">
            <b>의사결정:</b> {'사용 가능' if decision_report_exists else '미생성'}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="{'success-box' if segmentation_report_exists else 'warning-box'}">
            <b>고객 세그먼테이션:</b> {'사용 가능' if segmentation_report_exists else '미생성'}
        </div>
        """, unsafe_allow_html=True)
    
    # Required reports based on report type
    required_reports = {
        "comprehensive": ["trend", "forecast", "decision", "segmentation"],
        "executive": ["trend", "forecast", "decision"],
        "marketing": ["trend", "decision", "segmentation"],
        "operations": ["forecast", "trend"]
    }
    
    # Check if required reports are available
    missing_reports = []
    for report in required_reports[report_type]:
        if report == "trend" and not trend_report_exists:
            missing_reports.append("트렌드 분석")
        elif report == "forecast" and not forecast_report_exists:
            missing_reports.append("수요 예측")
        elif report == "decision" and not decision_report_exists:
            missing_reports.append("의사결정")
        elif report == "segmentation" and not segmentation_report_exists:
            missing_reports.append("고객 세그먼테이션")
    
    # Warning if required reports are missing
    if missing_reports:
        st.warning(f"선택한 리포트 유형({report_type})을 생성하기 위해 다음 분석이 필요합니다: {', '.join(missing_reports)}")
        
        # Add buttons to run missing reports
        st.markdown("### 필요한 분석 실행")
        
        missing_report_cols = st.columns(len(missing_reports))
        
        for i, report in enumerate(missing_reports):
            with missing_report_cols[i]:
                if report == "트렌드 분석":
                    if st.button("트렌드 분석 실행", key="run_trend_for_report"):
                        st.session_state.current_agent = "trend"
                        st.rerun()
                elif report == "수요 예측":
                    if st.button("수요 예측 실행", key="run_forecast_for_report"):
                        st.session_state.current_agent = "forecast"
                        st.rerun()
                elif report == "의사결정":
                    if st.button("의사결정 실행", key="run_decision_for_report"):
                        st.session_state.current_agent = "decision"
                        st.rerun()
                elif report == "고객 세그먼테이션":
                    if st.button("세그먼테이션 실행", key="run_segmentation_for_report"):
                        st.session_state.current_agent = "segmentation"
                        st.rerun()
    
    # Run reporting agent button
    if not missing_reports and st.button(f"{report_type} 리포트 생성", key="generate_report"):
        with st.spinner(f"{report_type} 리포트 생성 중..."):
            try:
                # Run reporting agent
                reporting_agent = ReportingAgent(
                    project_id=st.session_state.project_id,
                    dataset_id=st.session_state.dataset_id,
                    report_type=report_type,
                    trend_report_path=st.session_state.trend_report_path,
                    forecast_report_path=st.session_state.forecast_report_path,
                    decision_report_path=st.session_state.decision_report_path,
                    segmentation_report_path=st.session_state.segmentation_report_path,
                    bq_connector=st.session_state.bq_connector
                )
                
                report_path = reporting_agent.run()
                
                # Store report path based on type
                if report_type == "comprehensive":
                    st.session_state.comprehensive_report_path = report_path
                elif report_type == "executive":
                    st.session_state.executive_report_path = report_path
                elif report_type == "marketing":
                    st.session_state.marketing_report_path = report_path
                elif report_type == "operations":
                    st.session_state.operations_report_path = report_path
                
                # Get executive summary and insights
                executive_summary = reporting_agent.get_executive_summary()
                combined_insights = reporting_agent.get_combined_insights()
                
                # Display success message
                st.success(f"{report_type} 리포트가 생성되었습니다!")
                
                # Display executive summary
                st.markdown('<div class="sub-header">요약</div>', unsafe_allow_html=True)
                st.markdown(executive_summary)
                
                # Display insights
                st.markdown('<div class="sub-header">주요 인사이트</div>', unsafe_allow_html=True)
                
                for insight in combined_insights:
                    st.markdown(f"- {insight}")
                
                # Display report download button
                st.markdown('<div class="sub-header">리포트 다운로드</div>', unsafe_allow_html=True)
                
                if os.path.exists(report_path):
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label=f"{report_type} 리포트 다운로드",
                            data=file,
                            file_name=f"{report_type}_report.pdf",
                            mime="application/pdf"
                        )
                
            except Exception as e:
                st.error(f"리포트 생성 중 오류 발생: {str(e)}")
    
    # Custom report configuration
    st.markdown('<div class="sub-header">커스텀 리포트 설정</div>', unsafe_allow_html=True)
    
    st.markdown("""
    특정 키워드나 주제에 초점을 맞춘 커스텀 리포트를 생성할 수 있습니다.
    원하는 필터와 설정을 선택하세요.
    """)
    
    # Custom report filters
    custom_filters = st.multiselect(
        "인사이트 필터 (선택한 키워드가 포함된 인사이트만 포함)",
        options=["고객", "제품", "카테고리", "채널", "매출", "성장", "트렌드", "예측", "세그먼트", "마케팅", "캠페인"]
    )
    
    custom_viz_filters = st.multiselect(
        "시각화 필터 (선택한 키워드가 포함된 시각화만 포함)",
        options=["고객", "제품", "카테고리", "채널", "매출", "성장", "트렌드", "예측", "세그먼트", "마케팅", "캠페인"]
    )
    
    custom_summary = st.text_area(
        "커스텀 요약 (비워두면 자동 생성)",
        placeholder="리포트에 포함할 커스텀 요약을 입력하세요.",
        height=100
    )
    
    # Run custom report button
    if st.button("커스텀 리포트 생성", key="generate_custom_report"):
        if missing_reports:
            st.warning(f"커스텀 리포트를 생성하기 위해 다음 분석이 필요합니다: {', '.join(missing_reports)}")
        else:
            with st.spinner("커스텀 리포트 생성 중..."):
                try:
                    # Create custom report configuration
                    report_config = {
                        "report_type": report_type
                    }
                    
                    if custom_filters:
                        report_config["insight_filters"] = custom_filters
                    
                    if custom_viz_filters:
                        report_config["visualization_filters"] = custom_viz_filters
                    
                    if custom_summary:
                        report_config["custom_summary"] = custom_summary
                    
                    # Run reporting agent with custom configuration
                    reporting_agent = ReportingAgent(
                        project_id=st.session_state.project_id,
                        dataset_id=st.session_state.dataset_id,
                        report_type=report_type,
                        trend_report_path=st.session_state.trend_report_path,
                        forecast_report_path=st.session_state.forecast_report_path,
                        decision_report_path=st.session_state.decision_report_path,
                        segmentation_report_path=st.session_state.segmentation_report_path,
                        bq_connector=st.session_state.bq_connector
                    )
                    
                    report_path = reporting_agent.generate_custom_report(report_config)
                    
                    # Display success message
                    st.success("커스텀 리포트가 생성되었습니다!")
                    
                    # Display report download button
                    if os.path.exists(report_path):
                        with open(report_path, "rb") as file:
                            st.download_button(
                                label="커스텀 리포트 다운로드",
                                data=file,
                                file_name="custom_report.pdf",
                                mime="application/pdf"
                            )
                    
                except Exception as e:
                    st.error(f"커스텀 리포트 생성 중 오류 발생: {str(e)}")

# Settings page
def show_settings():
    st.markdown('<div class="main-header">설정</div>', unsafe_allow_html=True)
    
    # System settings
    st.markdown('<div class="sub-header">시스템 설정</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        project_id = st.text_input("GCP 프로젝트 ID", value=st.session_state.project_id)
        dataset_id = st.text_input("BigQuery 데이터셋 ID", value=st.session_state.dataset_id)
    
    with col2:
        debug_mode = st.checkbox("디버그 모드", value=config.DEBUG_MODE)
        reports_dir = st.text_input("리포트 저장 경로", value=config.REPORTS_DIR)
    
    # Agent settings
    st.markdown('<div class="sub-header">에이전트 설정</div>', unsafe_allow_html=True)
    
    agent_tabs = st.tabs(["트렌드 분석", "수요 예측", "의사결정", "세그먼테이션", "리포팅"])
    
    with agent_tabs[0]:  # Trend Analysis
        trend_timeframe = st.slider(
            "기본 분석 기간 (일)",
            min_value=30,
            max_value=365,
            value=config.TREND_ANALYSIS_TIMEFRAME_DAYS,
            step=30
        )
    
    with agent_tabs[1]:  # Demand Forecast
        forecast_horizon = st.slider(
            "기본 예측 기간 (주)",
            min_value=4,
            max_value=52,
            value=config.FORECAST_HORIZON_WEEKS,
            step=4
        )
    
    with agent_tabs[2]:  # Decision Agent
        pass  # No specific settings for now
    
    with agent_tabs[3]:  # Segmentation Agent
        pass  # No specific settings for now
    
    with agent_tabs[4]:  # Reporting Agent
        default_report_type = st.selectbox(
            "기본 리포트 유형",
            options=["comprehensive", "executive", "marketing", "operations"],
            index=0,
            format_func=lambda x: {
                "comprehensive": "종합 리포트",
                "executive": "경영진용 요약 리포트",
                "marketing": "마케팅 전략 리포트",
                "operations": "운영 계획 리포트"
            }.get(x, x)
        )
    
    # LLM settings
    st.markdown('<div class="sub-header">LLM 설정</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gemini_model = st.selectbox(
            "Gemini 모델",
            options=["gemini-1.0-pro", "gemini-1.0-pro-vision", "gemini-1.5-pro"],
            index=0
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=config.GEMINI_TEMPERATURE, step=0.1)
    
    with col2:
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=config.GEMINI_TOP_P, step=0.1)
        max_output_tokens = st.slider("최대 출력 토큰", min_value=1024, max_value=8192, value=config.GEMINI_MAX_OUTPUT_TOKENS, step=1024)
    
    # Save settings button
    if st.button("설정 저장", key="save_settings"):
        try:
            # Update session state
            st.session_state.project_id = project_id
            st.session_state.dataset_id = dataset_id
            
            # Update config values (in a real app, this would write to a config file)
            config.GCP_PROJECT_ID = project_id
            config.BQ_DATASET_ID = dataset_id
            config.DEBUG_MODE = debug_mode
            config.REPORTS_DIR = reports_dir
            config.TREND_ANALYSIS_TIMEFRAME_DAYS = trend_timeframe
            config.FORECAST_HORIZON_WEEKS = forecast_horizon
            config.GEMINI_MODEL = gemini_model
            config.GEMINI_TEMPERATURE = temperature
            config.GEMINI_TOP_P = top_p
            config.GEMINI_MAX_OUTPUT_TOKENS = max_output_tokens
            
            # Reinitialize BigQuery connector
            st.session_state.bq_connector = BigQueryConnector(project_id=project_id)
            
            st.success("설정이 저장되었습니다.")
            
        except Exception as e:
            st.error(f"설정 저장 중 오류 발생: {str(e)}")
    
    # Reset all data button
    st.markdown('<div class="sub-header">데이터 초기화</div>', unsafe_allow_html=True)
    
    if st.button("모든 데이터 초기화", key="reset_all_data"):
        if st.session_state.authenticated:
            # Keep authentication but reset all other data
            authenticated = True
            project_id = st.session_state.project_id
            dataset_id = st.session_state.dataset_id
            bq_connector = st.session_state.bq_connector
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.session_state.authenticated = authenticated
            st.session_state.project_id = project_id
            st.session_state.dataset_id = dataset_id
            st.session_state.bq_connector = bq_connector
            st.session_state.current_agent = "dashboard"
            
            st.success("모든 데이터가 초기화되었습니다.")
            st.rerun()

# Main execution
if __name__ == "__main__":
    if not st.session_state.authenticated:
        authenticate()
    else:
        main()
