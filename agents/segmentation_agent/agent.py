"""
Segmentation Agent for the CRM-Agent system.
This module defines the main class and interface for the customer segmentation functionality.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

import config
from utils.bigquery import BigQueryConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SegmentationAgent:
    """
    Agent for analyzing customer data and creating meaningful segments.
    
    This agent orchestrates the process of converting natural language to SQL,
    executing queries, analyzing segments, validating results, creating
    visualizations, and generating comprehensive reports.
    """
    
    def __init__(
        self,
        project_id: str = None,
        dataset_id: str = None,
        bq_connector: BigQueryConnector = None
    ):
        """
        Initialize the Segmentation Agent.
        
        Args:
            project_id: Google Cloud project ID (if None, uses the value from config)
            dataset_id: BigQuery dataset ID (if None, uses the value from config)
            bq_connector: Optional existing BigQuery connector to reuse
        """
        self.project_id = project_id or config.GCP_PROJECT_ID
        self.dataset_id = dataset_id or config.BQ_DATASET_ID
        
        # Initialize BigQuery connector if not provided
        self.bq_connector = bq_connector or BigQueryConnector(project_id=self.project_id)
        
        # Initialize results storage
        self.segments = {}
        self.segment_analysis = {}
        self.visualizations = []
        self.report_path = None
        self.queries = {}
        self.segment_validation = {}
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_TEMPERATURE,
            top_p=config.GEMINI_TOP_P,
            top_k=config.GEMINI_TOP_K,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
            google_api_key=config.GEMINI_API_KEY,
        )
        
        logger.info(f"Segmentation Agent initialized for dataset {self.dataset_id}")
    
    def run(self, segmentation_request: str = None) -> str:
        """
        Run the segmentation workflow.
        
        Args:
            segmentation_request: Natural language request for segmentation (optional)
            
        Returns:
            Path to the generated report
        """
        logger.info("Starting segmentation workflow")
        
        try:
            # Step 1: Convert natural language to SQL (if request provided)
            if segmentation_request:
                sql_query = self.text_to_sql(segmentation_request)
                segment_data = self.execute_query(sql_query)
                self.segments["custom"] = segment_data
            else:
                # Run default segmentation if no specific request
                self._run_default_segmentation()
            
            # Step 2: Analyze segments
            self._analyze_segments()
            
            # Step 3: Validate segments
            self._validate_segments()
            
            # Step 4: Create visualizations
            self._create_visualizations()
            
            # Step 5: Generate report
            self._generate_report()
            
            logger.info(f"Segmentation completed successfully. Report: {self.report_path}")
            return self.report_path
            
        except Exception as e:
            logger.error(f"Error running segmentation: {str(e)}")
            raise
    
    def text_to_sql(self, text_request: str) -> str:
        """
        Convert natural language request to SQL query.
        
        Args:
            text_request: Natural language segmentation request
            
        Returns:
            SQL query string
        """
        logger.info(f"Converting text to SQL: {text_request}")
        
        # Get table schema information
        customer_schema = self.bq_connector.get_table_schema(f"{self.dataset_id}.customer_master")
        transaction_schema = self.bq_connector.get_table_schema(f"{self.dataset_id}.offline_transactions")
        online_schema = self.bq_connector.get_table_schema(f"{self.dataset_id}.online_behavior")
        product_schema = self.bq_connector.get_table_schema(f"{self.dataset_id}.product_master")
        
        # Format schema information for the prompt
        schema_info = f"""
        ## 테이블 스키마 정보
        
        1. customer_master 테이블:
        {self._format_schema(customer_schema)}
        
        2. offline_transactions 테이블:
        {self._format_schema(transaction_schema)}
        
        3. online_behavior 테이블:
        {self._format_schema(online_schema)}
        
        4. product_master 테이블:
        {self._format_schema(product_schema)}
        """
        
        # Create prompt for SQL generation
        prompt = f"""
        당신은 가전 리테일 업체의 CDP 시스템에서 고객 세그먼테이션을 위한 SQL 쿼리를 생성하는 전문가입니다.
        다음 테이블 스키마 정보와 자연어 요청을 바탕으로 BigQuery SQL 쿼리를 작성해주세요.
        
        {schema_info}
        
        ## 자연어 요청
        {text_request}
        
        다음 가이드라인을 따라주세요:
        1. 쿼리는 BigQuery SQL 문법을 사용해야 합니다.
        2. 테이블 이름에는 데이터셋 ID를 포함해야 합니다 (예: `{self.dataset_id}.customer_master`).
        3. 쿼리 결과는 고객 ID(customer_id)와 관련 속성을 포함해야 합니다.
        4. 가능하면 세그먼트 이름이나 레이블을 포함하는 열을 생성해주세요.
        5. 쿼리 설명이나 주석 없이 SQL 쿼리만 제공해주세요.
        6. yyyy-mm-dd 형태의 날짜나, timestamp처럼 보이는 컬럼들은 현재 STRING 타입입니다. 불필요한 PARSE_DATE를 하지 마세요.
        """
        
        # Get SQL from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip()
        
        # Clean up the SQL query (remove markdown formatting if present)
        if sql_query.startswith("```sql"):
            sql_query = sql_query.split("```sql")[1]
        if sql_query.endswith("```"):
            sql_query = sql_query.split("```")[0]
        
        sql_query = sql_query.strip()
        
        print(sql_query)
        
        # Store the query
        self.queries[text_request] = sql_query
        
        logger.info(f"Generated SQL query: {sql_query[:100]}...")
        return sql_query
    
    def _format_schema(self, schema: List[Dict[str, Any]]) -> str:
        """Format table schema for the prompt."""
        if not schema:
            return "스키마 정보가 없습니다."
        
        formatted = []
        for field in schema:
            formatted.append(f"- {field.get('name', 'unknown')}: {field.get('type', 'unknown')} - {field.get('description', '설명 없음')}")
        
        return "\n".join(formatted)
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            DataFrame with query results
        """
        logger.info("Executing SQL query")
        
        # Validate query before execution
        self._validate_query(sql_query)
        
        # Execute query
        result = self.bq_connector.run_query(sql_query)
        
        logger.info(f"Query returned {len(result)} rows")
        return result
    
    def _validate_query(self, sql_query: str) -> None:
        """
        Validate SQL query for safety and correctness.
        
        Args:
            sql_query: SQL query to validate
            
        Raises:
            ValueError: If query is invalid or potentially harmful
        """
        # Check for data modification statements
        modification_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        for keyword in modification_keywords:
            if keyword in sql_query.upper():
                raise ValueError(f"Query contains forbidden keyword: {keyword}")
        
        # Ensure query includes customer_id
        if "customer_id" not in sql_query.lower():
            raise ValueError("Query must include customer_id column")
        
        # Additional validation could be added here
        
        logger.info("SQL query validation passed")
    
    def _run_default_segmentation(self) -> None:
        """Run default segmentation strategies."""
        logger.info("Running default segmentation")
        
        # 1. RFM Segmentation
        rfm_query = f"""
        WITH rfm_data AS (
            SELECT
                c.customer_id,
                DATETIME_DIFF(CURRENT_DATETIME(), MAX(PARSE_DATETIME('%Y-%m-%d %H:%M:%S', t.transaction_date)), DAY) as recency,
                COUNT(DISTINCT t.transaction_id) as frequency,
                SUM(t.total_amount) as monetary
            FROM
                `{self.dataset_id}.customer_master` c
            LEFT JOIN
                `{self.dataset_id}.offline_transactions` t
            ON
                c.customer_id = t.customer_id
            GROUP BY
                c.customer_id
        ),
        rfm_scores AS (
            SELECT
                customer_id,
                recency,
                frequency,
                monetary,
                CASE
                    WHEN recency <= 30 THEN 3
                    WHEN recency <= 90 THEN 2
                    ELSE 1
                END as r_score,
                CASE
                    WHEN frequency >= 10 THEN 3
                    WHEN frequency >= 5 THEN 2
                    ELSE 1
                END as f_score,
                CASE
                    WHEN monetary >= 1000000 THEN 3
                    WHEN monetary >= 500000 THEN 2
                    ELSE 1
                END as m_score
            FROM
                rfm_data
        )
        SELECT
            rs.*,
            CONCAT(
                CASE WHEN r_score = 3 THEN 'H' WHEN r_score = 2 THEN 'M' ELSE 'L' END,
                CASE WHEN f_score = 3 THEN 'H' WHEN f_score = 2 THEN 'M' ELSE 'L' END,
                CASE WHEN m_score = 3 THEN 'H' WHEN m_score = 2 THEN 'M' ELSE 'L' END
            ) as rfm_segment,
            CASE
                WHEN r_score = 3 AND f_score = 3 AND m_score = 3 THEN '최우수 고객'
                WHEN r_score >= 2 AND f_score >= 2 AND m_score >= 2 THEN '우수 고객'
                WHEN r_score = 1 AND f_score >= 2 AND m_score >= 2 THEN '휴면 우수 고객'
                WHEN r_score = 3 AND f_score = 1 AND m_score = 1 THEN '신규 고객'
                WHEN r_score = 1 AND f_score = 1 AND m_score = 1 THEN '이탈 위험 고객'
                ELSE '일반 고객'
            END as segment_name
        FROM
            rfm_scores rs
        """
        
        rfm_segments = self.execute_query(rfm_query)
        self.segments["rfm"] = rfm_segments
        
        # 2. Lifecycle Segmentation
        lifecycle_query = f"""
        WITH customer_activity AS (
            SELECT
                c.customer_id,
                c.registration_date,
                DATETIME_DIFF(CURRENT_DATETIME(), PARSE_DATETIME('%Y-%m-%d', c.registration_date), DAY) as days_since_registration,
                DATE_DIFF(CURRENT_DATE(), PARSE_DATE('%Y-%m-%d', c.registration_date), DAY) as days_since_registration,
                DATETIME_DIFF(CURRENT_DATETIME(), MAX(PARSE_DATETIME('%Y-%m-%d %H:%M:%S', t.transaction_date)), DAY) as days_since_last_purchase,
                DATE_DIFF(CURRENT_DATE(), DATE(MAX(PARSE_DATETIME('%Y-%m-%d %H:%M:%S', t.transaction_date))), DAY) as days_since_last_purchase,
                COUNT(DISTINCT ob.session_id) as online_sessions
            FROM
                `{self.dataset_id}.customer_master` c
            LEFT JOIN
                `{self.dataset_id}.offline_transactions` t
            ON
                c.customer_id = t.customer_id
            LEFT JOIN
                `{self.dataset_id}.online_behavior` ob
            ON
                c.customer_id = ob.customer_id
            GROUP BY
                c.customer_id, c.registration_date
        )
        SELECT
            ca.*,
            CASE
                WHEN days_since_registration <= 30 AND purchase_count > 0 THEN '신규 활성 고객'
                WHEN days_since_registration <= 30 AND purchase_count = 0 THEN '신규 미구매 고객'
                WHEN days_since_last_purchase <= 30 THEN '활성 고객'
                WHEN days_since_last_purchase <= 90 THEN '준활성 고객'
                WHEN days_since_last_purchase <= 180 THEN '휴면 위험 고객'
                ELSE '휴면 고객'
            END as lifecycle_segment
        FROM
            customer_activity ca
        """
        
        lifecycle_segments = self.execute_query(lifecycle_query)
        self.segments["lifecycle"] = lifecycle_segments
        
        # 3. Channel Preference Segmentation
        channel_query = f"""
        WITH channel_data AS (
            SELECT
                c.customer_id,
                COUNT(DISTINCT t.transaction_id) as offline_transactions,
                SUM(t.total_amount) as offline_amount,
                COUNT(DISTINCT ob.session_id) as online_sessions,
                COUNT(DISTINCT CASE WHEN ob.event_type = 'purchase' THEN ob.session_id ELSE NULL END) as online_purchases
            FROM
                `{self.dataset_id}.customer_master` c
            LEFT JOIN
                `{self.dataset_id}.offline_transactions` t
            ON
                c.customer_id = t.customer_id
            LEFT JOIN
                `{self.dataset_id}.online_behavior` ob
            ON
                c.customer_id = ob.customer_id
            GROUP BY
                c.customer_id
        )
        SELECT
            cd.*,
            CASE
                WHEN offline_transactions > 0 AND online_sessions = 0 THEN '오프라인 전용'
                WHEN offline_transactions = 0 AND online_sessions > 0 THEN '온라인 전용'
                WHEN offline_transactions > 0 AND online_sessions > 0 THEN '옴니채널'
                ELSE '미활동'
            END as channel_segment
        FROM
            channel_data cd
        """
        
        channel_segments = self.execute_query(channel_query)
        self.segments["channel"] = channel_segments
        
        # 4. Product Category Preference
        category_query = f"""
        WITH category_preferences AS (
            SELECT
                t.customer_id,
                p.category_level_1,
                COUNT(DISTINCT t.transaction_id) as purchase_count,
                SUM(t.total_amount) as total_spent
            FROM
                `{self.dataset_id}.offline_transactions` t
            JOIN
                `{self.dataset_id}.product_master` p
            ON
                t.product_id = p.product_id
            GROUP BY
                t.customer_id, p.category_level_1
        ),
        customer_total AS (
            SELECT
                customer_id,
                SUM(total_spent) as customer_total_spent
            FROM
                category_preferences
            GROUP BY
                customer_id
        ),
        category_ranks AS (
            SELECT
                cp.customer_id,
                cp.category_level_1,
                cp.purchase_count,
                cp.total_spent,
                cp.total_spent / ct.customer_total_spent as category_share,
                ROW_NUMBER() OVER (PARTITION BY cp.customer_id ORDER BY cp.total_spent DESC) as category_rank
            FROM
                category_preferences cp
            JOIN
                customer_total ct
            ON
                cp.customer_id = ct.customer_id
        )
        SELECT
            cr.customer_id,
            MAX(CASE WHEN category_rank = 1 THEN category_level_1 ELSE NULL END) as top_category,
            MAX(CASE WHEN category_rank = 1 THEN total_spent ELSE NULL END) as top_category_spent,
            MAX(CASE WHEN category_rank = 1 THEN category_share ELSE NULL END) as top_category_share,
            COUNT(DISTINCT category_level_1) as category_diversity,
            CASE
                WHEN COUNT(DISTINCT category_level_1) = 1 THEN '단일 카테고리'
                WHEN MAX(CASE WHEN category_rank = 1 THEN category_share ELSE NULL END) >= 0.7 THEN '주요 카테고리 집중'
                WHEN COUNT(DISTINCT category_level_1) >= 4 THEN '다양한 카테고리'
                ELSE '일반 구매자'
            END as category_segment
        FROM
            category_ranks cr
        GROUP BY
            cr.customer_id
        """
        
        category_segments = self.execute_query(category_query)
        self.segments["category"] = category_segments
        
        logger.info(f"Created {len(self.segments)} default segment types")
    
    def _analyze_segments(self) -> None:
        """Analyze segments to extract insights."""
        logger.info("Analyzing segments")
        
        for segment_type, segment_data in self.segments.items():
            logger.info(f"Analyzing {segment_type} segments")
            
            # Initialize segment analysis
            self.segment_analysis[segment_type] = {
                "segment_counts": {},
                "segment_profiles": {},
                "insights": []
            }
            
            # Get segment column name
            segment_col = None
            if segment_type == "rfm":
                segment_col = "segment_name"
            elif segment_type == "lifecycle":
                segment_col = "lifecycle_segment"
            elif segment_type == "channel":
                segment_col = "channel_segment"
            elif segment_type == "category":
                segment_col = "category_segment"
            elif segment_type == "custom":
                # Try to find a column that might contain segment information
                potential_cols = [col for col in segment_data.columns if "segment" in col.lower()]
                if potential_cols:
                    segment_col = potential_cols[0]
            
            # Skip if no segment column found
            if not segment_col or segment_col not in segment_data.columns:
                logger.warning(f"No segment column found for {segment_type}")
                continue
            
            # Calculate segment counts
            segment_counts = segment_data[segment_col].value_counts().to_dict()
            self.segment_analysis[segment_type]["segment_counts"] = segment_counts
            
            # Create segment profiles
            for segment_name, count in segment_counts.items():
                segment_subset = segment_data[segment_data[segment_col] == segment_name]
                
                # Calculate basic statistics for numeric columns
                profile = {
                    "count": count,
                    "percentage": (count / len(segment_data)) * 100,
                    "metrics": {}
                }
                
                for col in segment_data.columns:
                    if col == segment_col:
                        continue
                        
                    if pd.api.types.is_numeric_dtype(segment_data[col]):
                        profile["metrics"][col] = {
                            "mean": segment_subset[col].mean(),
                            "median": segment_subset[col].median(),
                            "min": segment_subset[col].min(),
                            "max": segment_subset[col].max()
                        }
                
                self.segment_analysis[segment_type]["segment_profiles"][segment_name] = profile
            
            # Generate insights using LLM
            self._generate_segment_insights(segment_type, segment_data, segment_col)
    
    def _generate_segment_insights(self, segment_type: str, segment_data: pd.DataFrame, segment_col: str) -> None:
        """Generate insights for segments using LLM."""
        # Prepare segment statistics for the prompt
        segment_stats = []
        for segment_name, profile in self.segment_analysis[segment_type]["segment_profiles"].items():
            stats = f"세그먼트: {segment_name}\n"
            stats += f"고객 수: {profile['count']} ({profile['percentage']:.1f}%)\n"
            
            # Add key metrics
            if "metrics" in profile:
                for metric, values in profile["metrics"].items():
                    if metric in ["recency", "frequency", "monetary", "days_since_last_purchase", 
                                "purchase_count", "offline_transactions", "online_sessions"]:
                        stats += f"{metric}: 평균 {values['mean']:.1f}, 중앙값 {values['median']:.1f}\n"
            
            segment_stats.append(stats)
        
        segment_stats_text = "\n\n".join(segment_stats)
        
        # Create prompt for insight generation
        prompt = f"""
        당신은 가전 리테일 업체의 고객 세그먼테이션 전문가입니다. 다음 '{segment_type}' 세그먼트 분석 결과를 
        검토하고 주요 인사이트와 마케팅 제안을 작성해주세요.
        
        ## 세그먼트 통계
        {segment_stats_text}
        
        다음 항목에 대한 인사이트를 작성해주세요:
        1. 각 세그먼트의 주요 특징
        2. 세그먼트별 마케팅 접근 방법
        3. 세그먼트 간 비교 분석
        4. 세그먼트 활용을 위한 제안
        
        각 인사이트는 데이터에 기반하고 실행 가능해야 합니다.
        """
        
        # Get insights from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        insights = response.content
        
        # Store insights
        self.segment_analysis[segment_type]["insights"] = insights
        
        logger.info(f"Generated insights for {segment_type} segments")
    
    def _validate_segments(self) -> None:
        """Validate segments for quality and usefulness."""
        logger.info("Validating segments")
        
        for segment_type, segment_data in self.segments.items():
            logger.info(f"Validating {segment_type} segments")
            
            # Initialize validation results
            self.segment_validation[segment_type] = {
                "quality_metrics": {},
                "warnings": [],
                "recommendations": []
            }
            
            # Get segment column name
            segment_col = None
            if segment_type == "rfm":
                segment_col = "segment_name"
            elif segment_type == "lifecycle":
                segment_col = "lifecycle_segment"
            elif segment_type == "channel":
                segment_col = "channel_segment"
            elif segment_type == "category":
                segment_col = "category_segment"
            elif segment_type == "custom":
                # Try to find a column that might contain segment information
                potential_cols = [col for col in segment_data.columns if "segment" in col.lower()]
                if potential_cols:
                    segment_col = potential_cols[0]
            
            # Skip if no segment column found
            if not segment_col or segment_col not in segment_data.columns:
                logger.warning(f"No segment column found for {segment_type}")
                continue
            
            # Calculate segment counts
            segment_counts = segment_data[segment_col].value_counts()
            total_customers = len(segment_data)
            
            # Calculate quality metrics
            quality_metrics = {
                "segment_count": len(segment_counts),
                "min_segment_size": segment_counts.min(),
                "max_segment_size": segment_counts.max(),
                "min_segment_percentage": (segment_counts.min() / total_customers) * 100,
                "max_segment_percentage": (segment_counts.max() / total_customers) * 100,
                "size_disparity": segment_counts.max() / segment_counts.min() if segment_counts.min() > 0 else float('inf')
            }
            
            self.segment_validation[segment_type]["quality_metrics"] = quality_metrics
            
            # Check for potential issues
            warnings = []
            
            # Check for very small segments
            if quality_metrics["min_segment_percentage"] < 5:
                warnings.append(f"일부 세그먼트가 전체의 5% 미만입니다 (최소: {quality_metrics['min_segment_percentage']:.1f}%).")
            
            # Check for very large segments
            if quality_metrics["max_segment_percentage"] > 50:
                warnings.append(f"일부 세그먼트가 전체의 50% 이상입니다 (최대: {quality_metrics['max_segment_percentage']:.1f}%).")
            
            # Check for high size disparity
            if quality_metrics["size_disparity"] > 10:
                warnings.append(f"세그먼트 크기 차이가 큽니다 (최대/최소: {quality_metrics['size_disparity']:.1f}).")
            
            # Check for too many segments
            if quality_metrics["segment_count"] > 10:
                warnings.append(f"세그먼트 수가 많습니다 ({quality_metrics['segment_count']}개). 너무 많은 세그먼트는 활용이 어려울 수 있습니다.")
            
            # Check for too few segments
            if quality_metrics["segment_count"] < 3:
                warnings.append(f"세그먼트 수가 적습니다 ({quality_metrics['segment_count']}개). 더 세분화된 세그먼트가 필요할 수 있습니다.")
            
            self.segment_validation[segment_type]["warnings"] = warnings
            
            # Generate recommendations based on warnings
            recommendations = []
            
            if any("5% 미만" in w for w in warnings):
                recommendations.append("작은 세그먼트를 통합하거나 특별 관리 대상으로 지정하는 것을 고려하세요.")
            
            if any("50% 이상" in w for w in warnings):
                recommendations.append("큰 세그먼트를 더 세분화하여 타겟팅 효과를 높이는 것을 고려하세요.")
            
            if any("크기 차이가 큽니다" in w for w in warnings):
                recommendations.append("세그먼트 기준을 조정하여 크기 균형을 개선하는 것을 고려하세요.")
            
            if any("세그먼트 수가 많습니다" in w for w in warnings):
                recommendations.append("유사한 세그먼트를 통합하여 관리 효율성을 높이는 것을 고려하세요.")
            
            if any("세그먼트 수가 적습니다" in w for w in warnings):
                recommendations.append("추가 변수를 도입하여 세그먼트를 더 세분화하는 것을 고려하세요.")
            
            self.segment_validation[segment_type]["recommendations"] = recommendations
            
            logger.info(f"Validated {segment_type} segments: {len(warnings)} warnings, {len(recommendations)} recommendations")
    
    def _create_visualizations(self) -> None:
        """Create visualizations for segments."""
        logger.info("Creating segment visualizations")
        
        from utils.visualization import create_pie_chart, create_bar_chart, save_plotly_fig
        
        for segment_type, segment_data in self.segments.items():
            logger.info(f"Creating visualizations for {segment_type} segments")
            
            # Get segment column name
            segment_col = None
            if segment_type == "rfm":
                segment_col = "segment_name"
            elif segment_type == "lifecycle":
                segment_col = "lifecycle_segment"
            elif segment_type == "channel":
                segment_col = "channel_segment"
            elif segment_type == "category":
                segment_col = "category_segment"
            elif segment_type == "custom":
                # Try to find a column that might contain segment information
                potential_cols = [col for col in segment_data.columns if "segment" in col.lower()]
                if potential_cols:
                    segment_col = potential_cols[0]
            
            # Skip if no segment column found
            if not segment_col or segment_col not in segment_data.columns:
                logger.warning(f"No segment column found for {segment_type}")
                continue
            
            # Create pie chart for segment distribution
            segment_counts = segment_data[segment_col].value_counts()
            
            fig_pie = create_pie_chart(
                values=segment_counts.values,
                labels=segment_counts.index,
                title=f"{segment_type.upper()} 세그먼트 분포",
                use_plotly=True
            )
            
            pie_path = save_plotly_fig(fig_pie, f"{segment_type}_segment_distribution.png")
            
            self.visualizations.append({
                "path": pie_path,
                "title": f"{segment_type.upper()} 세그먼트 분포",
                "description": f"{segment_type} 세그먼테이션 결과의 고객 분포를 보여줍니다.",
                "segment_type": segment_type
            })
            
            # Create additional visualizations based on segment type
            if segment_type == "rfm":
                # Create bar charts for R, F, M metrics by segment
                if all(col in segment_data.columns for col in ["segment_name", "recency", "frequency", "monetary"]):
                    # Recency by segment
                    recency_by_segment = segment_data.groupby("segment_name")["recency"].mean().sort_values()
                    
                    fig_recency = create_bar_chart(
                        x=recency_by_segment.index,
                        y=recency_by_segment.values,
                        title="세그먼트별 평균 최근성(Recency)",
                        xlabel="세그먼트",
                        ylabel="평균 최근성(일)",
                        use_plotly=True
                    )
                    
                    recency_path = save_plotly_fig(fig_recency, "rfm_recency_by_segment.png")
                    
                    self.visualizations.append({
                        "path": recency_path,
                        "title": "세그먼트별 평균 최근성(Recency)",
                        "description": "각 RFM 세그먼트의 평균 최근성(마지막 구매 후 경과일)을 보여줍니다.",
                        "segment_type": segment_type
                    })
                    
                    # Frequency by segment
                    frequency_by_segment = segment_data.groupby("segment_name")["frequency"].mean().sort_values(ascending=False)
                    
                    fig_frequency = create_bar_chart(
                        x=frequency_by_segment.index,
                        y=frequency_by_segment.values,
                        title="세그먼트별 평균 구매빈도(Frequency)",
                        xlabel="세그먼트",
                        ylabel="평균 구매빈도",
                        use_plotly=True
                    )
                    
                    frequency_path = save_plotly_fig(fig_frequency, "rfm_frequency_by_segment.png")
                    
                    self.visualizations.append({
                        "path": frequency_path,
                        "title": "세그먼트별 평균 구매빈도(Frequency)",
                        "description": "각 RFM 세그먼트의 평균 구매빈도를 보여줍니다.",
                        "segment_type": segment_type
                    })
                    
                    # Monetary by segment
                    monetary_by_segment = segment_data.groupby("segment_name")["monetary"].mean().sort_values(ascending=False)
                    
                    fig_monetary = create_bar_chart(
                        x=monetary_by_segment.index,
                        y=monetary_by_segment.values,
                        title="세그먼트별 평균 구매금액(Monetary)",
                        xlabel="세그먼트",
                        ylabel="평균 구매금액(원)",
                        use_plotly=True
                    )
                    
                    monetary_path = save_plotly_fig(fig_monetary, "rfm_monetary_by_segment.png")
                    
                    self.visualizations.append({
                        "path": monetary_path,
                        "title": "세그먼트별 평균 구매금액(Monetary)",
                        "description": "각 RFM 세그먼트의 평균 구매금액을 보여줍니다.",
                        "segment_type": segment_type
                    })
            
            elif segment_type == "lifecycle":
                # Create bar chart for online sessions by lifecycle segment
                if all(col in segment_data.columns for col in ["lifecycle_segment", "online_sessions"]):
                    online_by_segment = segment_data.groupby("lifecycle_segment")["online_sessions"].mean().sort_values(ascending=False)
                    
                    fig_online = create_bar_chart(
                        x=online_by_segment.index,
                        y=online_by_segment.values,
                        title="라이프사이클 세그먼트별 평균 온라인 세션 수",
                        xlabel="세그먼트",
                        ylabel="평균 온라인 세션 수",
                        use_plotly=True
                    )
                    
                    online_path = save_plotly_fig(fig_online, "lifecycle_online_sessions.png")
                    
                    self.visualizations.append({
                        "path": online_path,
                        "title": "라이프사이클 세그먼트별 평균 온라인 세션 수",
                        "description": "각 라이프사이클 세그먼트의 평균 온라인 활동 수준을 보여줍니다.",
                        "segment_type": segment_type
                    })
            
            elif segment_type == "channel":
                # Create bar chart for online vs offline activity
                if all(col in segment_data.columns for col in ["channel_segment", "offline_transactions", "online_sessions"]):
                    # Calculate average metrics by segment
                    channel_metrics = segment_data.groupby("channel_segment").agg({
                        "offline_transactions": "mean",
                        "online_sessions": "mean"
                    }).reset_index()
                    
                    # Create grouped bar chart
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=channel_metrics["channel_segment"],
                        y=channel_metrics["offline_transactions"],
                        name="평균 오프라인 거래 수",
                        marker_color='indianred'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=channel_metrics["channel_segment"],
                        y=channel_metrics["online_sessions"],
                        name="평균 온라인 세션 수",
                        marker_color='lightsalmon'
                    ))
                    
                    fig.update_layout(
                        title="채널 세그먼트별 온/오프라인 활동",
                        xaxis_title="채널 세그먼트",
                        yaxis_title="평균 활동 수",
                        barmode='group',
                        bargap=0.15,
                        bargroupgap=0.1
                    )
                    
                    channel_path = save_plotly_fig(fig, "channel_activity_comparison.png")
                    
                    self.visualizations.append({
                        "path": channel_path,
                        "title": "채널 세그먼트별 온/오프라인 활동",
                        "description": "각 채널 세그먼트의 온라인 및 오프라인 활동 수준을 비교합니다.",
                        "segment_type": segment_type
                    })
            
            elif segment_type == "category":
                # Create bar chart for category diversity
                if all(col in segment_data.columns for col in ["category_segment", "category_diversity"]):
                    diversity_by_segment = segment_data.groupby("category_segment")["category_diversity"].mean().sort_values(ascending=False)
                    
                    fig_diversity = create_bar_chart(
                        x=diversity_by_segment.index,
                        y=diversity_by_segment.values,
                        title="세그먼트별 평균 카테고리 다양성",
                        xlabel="세그먼트",
                        ylabel="평균 구매 카테고리 수",
                        use_plotly=True
                    )
                    
                    diversity_path = save_plotly_fig(fig_diversity, "category_diversity.png")
                    
                    self.visualizations.append({
                        "path": diversity_path,
                        "title": "세그먼트별 평균 카테고리 다양성",
                        "description": "각 카테고리 세그먼트의 평균 구매 카테고리 다양성을 보여줍니다.",
                        "segment_type": segment_type
                    })
        
        logger.info(f"Created {len(self.visualizations)} visualizations")
    
    def _generate_report(self) -> None:
        """Generate a PDF report with segmentation results."""
        logger.info("Generating segmentation report")
        
        from utils.pdf_generator import ReportPDF
        
        # Create PDF report
        report = ReportPDF()
        report.set_title("가전 리테일 CDP 고객 세그먼테이션 리포트")
        report.add_date()
        
        # Add executive summary
        summary = f"본 리포트는 가전 리테일 CDP 데이터를 활용한 {len(self.segments)}가지 세그먼테이션 분석 결과를 제시합니다. "
        summary += "RFM, 라이프사이클, 채널 선호도, 카테고리 선호도 등 다양한 관점에서 고객을 세분화하여 "
        summary += "타겟 마케팅과 개인화 전략 수립을 위한 인사이트를 제공합니다."
        
        report.add_executive_summary(summary)
        
        # Add sections for each segment type
        for segment_type in self.segments.keys():
            segment_name_map = {
                "rfm": "RFM 세그먼테이션",
                "lifecycle": "라이프사이클 세그먼테이션",
                "channel": "채널 선호도 세그먼테이션",
                "category": "카테고리 선호도 세그먼테이션",
                "custom": "커스텀 세그먼테이션"
            }
            
            section_title = segment_name_map.get(segment_type, f"{segment_type} 세그먼테이션")
            report.add_section(section_title)
            
            # Add segment distribution visualization
            segment_viz = next((v for v in self.visualizations if v["segment_type"] == segment_type and "분포" in v["title"]), None)
            if segment_viz:
                report.add_image(segment_viz["path"], width=160, caption=segment_viz["title"])
                report.add_text(segment_viz["description"])
            
            # Add segment counts
            if segment_type in self.segment_analysis and "segment_counts" in self.segment_analysis[segment_type]:
                counts = self.segment_analysis[segment_type]["segment_counts"]
                report.add_subsection("세그먼트 분포")
                
                count_text = ""
                for segment_name, count in counts.items():
                    percentage = (count / sum(counts.values())) * 100
                    count_text += f"• {segment_name}: {count}명 ({percentage:.1f}%)\n"
                
                report.add_text(count_text)
            
            # Add segment insights
            if segment_type in self.segment_analysis and "insights" in self.segment_analysis[segment_type]:
                insights = self.segment_analysis[segment_type]["insights"]
                report.add_subsection("세그먼트 인사이트")
                report.add_text(insights)
            
            # Add additional visualizations
            segment_vizs = [v for v in self.visualizations if v["segment_type"] == segment_type and "분포" not in v["title"]]
            for viz in segment_vizs:
                report.add_image(viz["path"], width=160, caption=viz["title"])
                report.add_text(viz["description"])
            
            # Add validation warnings and recommendations
            if segment_type in self.segment_validation:
                warnings = self.segment_validation[segment_type].get("warnings", [])
                recommendations = self.segment_validation[segment_type].get("recommendations", [])
                
                if warnings or recommendations:
                    report.add_subsection("세그먼트 품질 및 제안")
                    
                    if warnings:
                        report.add_text("주의사항:")
                        for warning in warnings:
                            report.add_text(f"• {warning}")
                    
                    if recommendations:
                        report.add_text("개선 제안:")
                        for recommendation in recommendations:
                            report.add_text(f"• {recommendation}")
        
        # Add overall recommendations section
        report.add_section("종합 제안 및 활용 방안")
        
        # Use LLM to generate overall recommendations
        all_insights = []
        for segment_type in self.segments.keys():
            if segment_type in self.segment_analysis and "insights" in self.segment_analysis[segment_type]:
                all_insights.append(f"## {segment_type} 세그먼트 인사이트\n{self.segment_analysis[segment_type]['insights']}")
        
        all_insights_text = "\n\n".join(all_insights)
        
        prompt = f"""
        당신은 가전 리테일 업체의 고객 세그먼테이션 전문가입니다. 다음 세그먼트 분석 인사이트를 종합하여
        전체적인 활용 방안과 마케팅 전략 제안을 작성해주세요.
        
        {all_insights_text}
        
        다음 항목에 대한 종합적인 제안을 작성해주세요:
        1. 세그먼트 간 연계 활용 방안
        2. 우선 타겟팅이 필요한 세그먼트
        3. 세그먼트별 차별화된 마케팅 접근법
        4. 세그먼테이션 결과의 비즈니스 활용 방안
        5. 향후 세그먼테이션 개선 방향
        
        각 제안은 구체적이고 실행 가능해야 합니다.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        overall_recommendations = response.content
        
        report.add_text(overall_recommendations)
        
        # Generate and save the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(config.REPORTS_DIR, f"segmentation_report_{timestamp}.pdf")
        report.output(report_path)
        
        # Update state with report path
        self.report_path = report_path
        
        logger.info(f"Segmentation report generated successfully at {report_path}")
    
    def get_segments(self, segment_type: str = None) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Get the generated segments.
        
        Args:
            segment_type: Optional segment type to retrieve specific segments
            
        Returns:
            Dictionary with all segments or specific segment DataFrame
        """
        if segment_type:
            if segment_type in self.segments:
                return self.segments[segment_type]
            else:
                raise ValueError(f"Segment type '{segment_type}' not found")
        return self.segments
    
    def get_segment_analysis(self, segment_type: str = None) -> Union[Dict[str, Any], Any]:
        """
        Get the segment analysis results.
        
        Args:
            segment_type: Optional segment type to retrieve specific analysis
            
        Returns:
            Dictionary with all analyses or specific analysis
        """
        if segment_type:
            if segment_type in self.segment_analysis:
                return self.segment_analysis[segment_type]
            else:
                raise ValueError(f"Segment analysis for '{segment_type}' not found")
        return self.segment_analysis
    
    def get_visualizations(self) -> List[Dict[str, Any]]:
        """
        Get the visualizations generated from the segmentation.
        
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
    
    def create_custom_segment(self, segment_name: str, query: str) -> pd.DataFrame:
        """
        Create a custom segment using SQL query.
        
        Args:
            segment_name: Name for the custom segment
            query: SQL query to define the segment
            
        Returns:
            DataFrame with the custom segment
        """
        logger.info(f"Creating custom segment: {segment_name}")
        
        # Validate query
        self._validate_query(query)
        
        # Execute query
        segment_data = self.execute_query(query)
        
        # Store segment
        self.segments[segment_name] = segment_data
        
        # Analyze the new segment
        segment_col = None
        potential_cols = [col for col in segment_data.columns if "segment" in col.lower()]
        if potential_cols:
            segment_col = potential_cols[0]
            self._analyze_segments()
            self._validate_segments()
            self._create_visualizations()
        
        logger.info(f"Created custom segment with {len(segment_data)} customers")
        return segment_data
    
    def compare_segments(self, segment_type1: str, segment_type2: str) -> Dict[str, Any]:
        """
        Compare two different segment types.
        
        Args:
            segment_type1: First segment type
            segment_type2: Second segment type
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing segments: {segment_type1} vs {segment_type2}")
        
        if segment_type1 not in self.segments:
            raise ValueError(f"Segment type '{segment_type1}' not found")
        
        if segment_type2 not in self.segments:
            raise ValueError(f"Segment type '{segment_type2}' not found")
        
        # Get segment data
        segment_data1 = self.segments[segment_type1]
        segment_data2 = self.segments[segment_type2]
        
        # Get segment column names
        segment_col1 = None
        segment_col2 = None
        
        for segment_type, col_name in [
            (segment_type1, "segment_name"),
            (segment_type1, "lifecycle_segment"),
            (segment_type1, "channel_segment"),
            (segment_type1, "category_segment")
        ]:
            if col_name in self.segments[segment_type].columns:
                segment_col1 = col_name
                break
        
        for segment_type, col_name in [
            (segment_type2, "segment_name"),
            (segment_type2, "lifecycle_segment"),
            (segment_type2, "channel_segment"),
            (segment_type2, "category_segment")
        ]:
            if col_name in self.segments[segment_type].columns:
                segment_col2 = col_name
                break
        
        # If segment columns not found, try to find them
        if not segment_col1:
            potential_cols = [col for col in segment_data1.columns if "segment" in col.lower()]
            if potential_cols:
                segment_col1 = potential_cols[0]
        
        if not segment_col2:
            potential_cols = [col for col in segment_data2.columns if "segment" in col.lower()]
            if potential_cols:
                segment_col2 = potential_cols[0]
        
        # If still not found, return error
        if not segment_col1 or not segment_col2:
            raise ValueError("Could not identify segment columns for comparison")
        
        # Merge segments on customer_id
        if "customer_id" in segment_data1.columns and "customer_id" in segment_data2.columns:
            merged_data = pd.merge(
                segment_data1[["customer_id", segment_col1]],
                segment_data2[["customer_id", segment_col2]],
                on="customer_id",
                how="inner"
            )
            
            # Create cross-tabulation
            crosstab = pd.crosstab(
                merged_data[segment_col1],
                merged_data[segment_col2],
                normalize="index"
            ) * 100
            
            # Create heatmap visualization
            from utils.visualization import create_heatmap, save_plotly_fig
            
            fig = create_heatmap(
                data_matrix=crosstab.values,
                x_labels=crosstab.columns,
                y_labels=crosstab.index,
                title=f"{segment_type1} vs {segment_type2} 세그먼트 비교",
                xlabel=f"{segment_type2} 세그먼트",
                ylabel=f"{segment_type1} 세그먼트",
                use_plotly=True
            )
            
            heatmap_path = save_plotly_fig(fig, f"{segment_type1}_vs_{segment_type2}_heatmap.png")
            
            self.visualizations.append({
                "path": heatmap_path,
                "title": f"{segment_type1} vs {segment_type2} 세그먼트 비교",
                "description": f"{segment_type1}과 {segment_type2} 세그먼트 간의 관계를 보여줍니다. 값은 행 기준 백분율입니다.",
                "segment_type": "comparison"
            })
            
            # Generate insights using LLM
            prompt = f"""
            당신은 가전 리테일 업체의 고객 세그먼테이션 전문가입니다. 다음 두 세그먼테이션 간의 
            교차 분석 결과를 검토하고 주요 인사이트를 작성해주세요.
            
            ## 교차 분석 결과
            {crosstab.to_string()}
            
            위 데이터는 {segment_type1} 세그먼트(행)와 {segment_type2} 세그먼트(열) 간의 관계를 보여줍니다.
            값은 행 기준 백분율입니다.
            
            두 세그먼테이션 간의 관계, 주요 패턴, 마케팅 활용 방안에 대한 인사이트를 작성해주세요.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            comparison_insights = response.content
            
            return {
                "crosstab": crosstab,
                "visualization_path": heatmap_path,
                "insights": comparison_insights,
                "segment_types": (segment_type1, segment_type2),
                "segment_columns": (segment_col1, segment_col2)
            }
        else:
            raise ValueError("Both segment types must contain customer_id column for comparison")

# For direct execution
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run the agent
    agent = SegmentationAgent()
    report_path = agent.run()
    
    print(f"Report generated at: {report_path}")
    print(f"Generated {len(agent.get_segments())} segment types")
    print(f"Created {len(agent.get_visualizations())} visualizations")
