"""
Segmentation Agent for the CRM-Agent system.
This module defines the main class and interface for the customer segmentation functionality.
"""
import os
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Set
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
        
        # Store valid table and column metadata
        self.valid_tables = []
        self.valid_columns = {}
        self.column_types = {}
        
        # Cache schema metadata on initialization
        self._cache_schema_metadata()
        
        logger.info(f"Segmentation Agent initialized for dataset {self.dataset_id}")
    
    def _cache_schema_metadata(self) -> None:
        """
        Cache detailed schema metadata from BigQuery tables.
        This helps prevent hallucinations by storing the actual table and column names.
        """
        logger.info("Caching schema metadata from BigQuery")
        
        # Define the tables we want to work with (from config)
        table_ids = [
            config.BQ_CUSTOMER_TABLE,
            config.BQ_PRODUCT_TABLE,
            config.BQ_OFFLINE_TRANSACTIONS_TABLE,
            config.BQ_ONLINE_BEHAVIOR_TABLE
        ]
        
        # Store valid table names (without dataset prefix)
        self.valid_tables = table_ids.copy()
        
        # Initialize column collections
        self.valid_columns = {table: set() for table in table_ids}
        self.column_types = {}
        
        # Get schema for each table and cache column information
        for table_id in table_ids:
            try:
                schema = self.bq_connector.get_table_schema(table_id)
                
                # Store column names and types for this table
                for field in schema:
                    column_name = field.get('name')
                    column_type = field.get('type')
                    
                    if column_name and column_type:
                        self.valid_columns[table_id].add(column_name)
                        # Store as "table.column": type
                        self.column_types[f"{table_id}.{column_name}"] = column_type
                        
                logger.info(f"Cached schema for {table_id}: {len(self.valid_columns[table_id])} columns")
                
            except Exception as e:
                logger.error(f"Error caching schema for table {table_id}: {str(e)}")
        
        # Create a mapping of common aliases to full table names
        self.table_aliases = {
            "c": config.BQ_CUSTOMER_TABLE,
            "p": config.BQ_PRODUCT_TABLE,
            "t": config.BQ_OFFLINE_TRANSACTIONS_TABLE,
            "ob": config.BQ_ONLINE_BEHAVIOR_TABLE,
            "customer": config.BQ_CUSTOMER_TABLE,
            "product": config.BQ_PRODUCT_TABLE,
            "transaction": config.BQ_OFFLINE_TRANSACTIONS_TABLE,
            "online": config.BQ_ONLINE_BEHAVIOR_TABLE
        }
    
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
        customer_schema = self.bq_connector.get_table_schema(config.BQ_CUSTOMER_TABLE)
        transaction_schema = self.bq_connector.get_table_schema(config.BQ_OFFLINE_TRANSACTIONS_TABLE)
        online_schema = self.bq_connector.get_table_schema(config.BQ_ONLINE_BEHAVIOR_TABLE)
        product_schema = self.bq_connector.get_table_schema(config.BQ_PRODUCT_TABLE)
        
        # Format schema information for the prompt
        schema_info = f"""
        ## 테이블 스키마 정보
        
        1. customer_master 테이블 (고객 정보):
        {self._format_schema(customer_schema)}
        
        2. offline_transactions 테이블 (오프라인 거래 정보):
        {self._format_schema(transaction_schema)}
        
        3. online_behavior 테이블 (온라인 행동 정보):
        {self._format_schema(online_schema)}
        
        4. product_master 테이블 (제품 정보):
        {self._format_schema(product_schema)}
        
        ## 테이블 관계
        - customer_master.customer_id = offline_transactions.customer_id
        - customer_master.customer_id = online_behavior.customer_id
        - offline_transactions.product_id = product_master.product_id
        
        ## 중요: 사용 가능한 테이블과 컬럼
        - 위에 명시된 4개의 테이블과 컬럼만 사용할 수 있습니다.
        - 존재하지 않는 테이블이나 컬럼을 참조하면 쿼리가 실패합니다.
        - 각 테이블의 별칭은 다음과 같이 사용하세요: customer_master -> c, product_master -> p, offline_transactions -> t, online_behavior -> ob
        """
        
        # Add example templates to help guide the model
        example_templates = """
        ## 예시 템플릿
        
        1. RFM 세그먼테이션:
        ```sql
        WITH rfm_data AS (
            SELECT
                c.customer_id,
                DATE_DIFF(CURRENT_DATE(), MAX(PARSE_DATE('%Y-%m-%d', t.transaction_date)), DAY) as recency,
                COUNT(DISTINCT t.transaction_id) as frequency,
                SUM(t.total_amount) as monetary
            FROM
                `{dataset_id}.customer_master` c
            LEFT JOIN
                `{dataset_id}.offline_transactions` t
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
            ) as rfm_segment
        FROM
            rfm_scores rs
        ```
        
        2. 라이프사이클 세그먼테이션:
        ```sql
        SELECT
            c.customer_id,
            c.registration_date,
            DATE_DIFF(CURRENT_DATE(), PARSE_DATE('%Y-%m-%d', c.registration_date), DAY) as days_since_registration,
            MAX(PARSE_DATE('%Y-%m-%d', t.transaction_date)) as last_purchase_date,
            DATE_DIFF(CURRENT_DATE(), MAX(PARSE_DATE('%Y-%m-%d', t.transaction_date)), DAY) as days_since_last_purchase,
            COUNT(DISTINCT t.transaction_id) as purchase_count,
            CASE
                WHEN DATE_DIFF(CURRENT_DATE(), PARSE_DATE('%Y-%m-%d', c.registration_date), DAY) <= 30 THEN '신규 고객'
                WHEN DATE_DIFF(CURRENT_DATE(), MAX(PARSE_DATE('%Y-%m-%d', t.transaction_date)), DAY) <= 30 THEN '활성 고객'
                WHEN DATE_DIFF(CURRENT_DATE(), MAX(PARSE_DATE('%Y-%m-%d', t.transaction_date)), DAY) <= 90 THEN '준활성 고객'
                ELSE '휴면 고객'
            END as lifecycle_segment
        FROM
            `{dataset_id}.customer_master` c
        LEFT JOIN
            `{dataset_id}.offline_transactions` t
        ON
            c.customer_id = t.customer_id
        GROUP BY
            c.customer_id, c.registration_date
        ```
        
        3. 온라인 행동 분석:
        ```sql
        SELECT
            c.customer_id,
            COUNT(DISTINCT ob.session_id) as session_count,
            MAX(PARSE_DATETIME('%Y-%m-%d %H:%M:%S', ob.event_timestamp)) as last_online_activity,
            DATETIME_DIFF(CURRENT_DATETIME(), MAX(PARSE_DATETIME('%Y-%m-%d %H:%M:%S', ob.event_timestamp)), DAY) as days_since_last_activity,
            COUNT(CASE WHEN ob.event_type = 'view_product' THEN 1 END) as product_views,
            COUNT(CASE WHEN ob.event_type = 'add_to_cart' THEN 1 END) as add_to_carts,
            COUNT(CASE WHEN ob.event_type = 'purchase' THEN 1 END) as online_purchases
        FROM
            `{dataset_id}.customer_master` c
        JOIN
            `{dataset_id}.online_behavior` ob
        ON
            c.customer_id = ob.customer_id
        GROUP BY
            c.customer_id
        ```
        """
        
        # Create prompt for SQL generation
        prompt = f"""
        당신은 가전 리테일 업체의 CDP 시스템에서 고객 세그먼테이션을 위한 SQL 쿼리를 생성하는 전문가입니다.
        다음 테이블 스키마 정보와 자연어 요청을 바탕으로 BigQuery SQL 쿼리를 작성해주세요.
        
        {schema_info}
        
        {example_templates}
        
        ## 자연어 요청
        {text_request}
        
        ## 중요 날짜 처리 가이드라인
        1. 모든 날짜/타임스탬프 관련 컬럼은 STRING 타입으로 저장되어 있습니다.
        2. 'YYYY-MM-DD' 형식의 날짜 문자열은 PARSE_DATE('%Y-%m-%d', column_name)을 사용하세요.
        3. 'YYYY-MM-DD HH:MM:SS' 형식의 타임스탬프 문자열은 PARSE_DATETIME('%Y-%m-%d %H:%M:%S', column_name)을 사용하세요.
        4. 날짜 비교는 DATE_DIFF() 함수를 사용하고, 타임스탬프 비교는 DATETIME_DIFF() 함수를 사용하세요.
        5. 현재 날짜는 CURRENT_DATE()를, 현재 타임스탬프는 CURRENT_DATETIME()을 사용하세요.
        
        ## GROUP BY 가이드라인
        1. GROUP BY 절을 사용할 때는 집계 함수(COUNT, SUM, MAX, MIN, AVG 등)로 감싸지 않은 모든 SELECT 컬럼을 GROUP BY에 포함해야 합니다.
        2. 날짜 컬럼을 사용할 때는 특히 주의하세요. 예를 들어 t.transaction_date를 SELECT에서 사용한다면:
           - GROUP BY에 t.transaction_date를 포함하거나
           - MAX(t.transaction_date)나 MIN(t.transaction_date) 같은 집계 함수로 감싸야 합니다.
        3. 날짜 계산(DATE_DIFF 등)을 사용할 때도 원본 날짜 컬럼을 GROUP BY에 포함하거나 집계 함수로 감싸야 합니다.
        
        ## 할루시네이션 방지 가이드라인
        1. 스키마에 명시된 테이블과 컬럼만 사용하세요. 존재하지 않는 테이블이나 컬럼을 참조하지 마세요.
        2. 사용 가능한 테이블: customer_master, product_master, offline_transactions, online_behavior
        3. 각 테이블에 있는 컬럼만 참조하세요. 예를 들어, customer_master 테이블에 없는 컬럼을 customer_master에서 조회하지 마세요.
        4. 테이블 별칭을 사용할 때는 일관성을 유지하세요: c (customer_master), p (product_master), t (offline_transactions), ob (online_behavior)
        
        ## 추가 가이드라인
        1. 쿼리는 BigQuery SQL 문법을 사용해야 합니다.
        2. 테이블 이름에는 데이터셋 ID를 포함해야 합니다 (예: `{self.dataset_id}.customer_master`).
        3. 쿼리 결과는 고객 ID(customer_id)와 관련 속성을 포함해야 합니다.
        4. 가능하면 세그먼트 이름이나 레이블을 포함하는 열을 생성해주세요.
        5. 쿼리 설명이나 주석 없이 SQL 쿼리만 제공해주세요.
        6. 쿼리는 실행 가능하고 효율적이어야 합니다.
        """
        
        # Get SQL from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip()
        
        # Clean up the SQL query (remove markdown formatting if present)
        if sql_query.startswith("```sql"):
            sql_query = sql_query.split("```sql")[1]
        elif sql_query.startswith("```"):
            sql_query = sql_query.split("```")[1]
        
        if sql_query.endswith("```"):
            sql_query = sql_query.split("```")[0]
        
        sql_query = sql_query.strip()
        
        # Apply date handling fixes
        sql_query = self._fix_date_handling(sql_query)
        
        # Fix GROUP BY issues in the generated SQL
        sql_query = self._fix_group_by_issues(sql_query)
        
        # Validate and fix hallucinated table and column references
        hallucination_issues = self._validate_table_column_references(sql_query)
        if hallucination_issues:
            logger.warning(f"Detected hallucinated references: {hallucination_issues}")
            sql_query = self._fix_hallucinated_references(sql_query)
            
            # If serious hallucination issues remain, regenerate the query
            remaining_issues = self._validate_table_column_references(sql_query)
            if remaining_issues:
                logger.warning(f"Hallucination issues remain after fixing: {remaining_issues}")
                sql_query = self._regenerate_query(text_request, 
                                                 f"쿼리에 존재하지 않는 테이블 또는 컬럼이 포함되어 있습니다: {', '.join(remaining_issues)}")
        
        # Store the query
        self.queries[text_request] = sql_query
        
        logger.info(f"Generated SQL query: {sql_query[:100]}...")
        return sql_query
    
    def _format_schema(self, schema: List[Dict[str, Any]]) -> str:
        """Format table schema for the prompt with enhanced type information."""
        if not schema:
            return "스키마 정보가 없습니다."
        
        formatted = []
        for field in schema:
            name = field.get('name', 'unknown')
            field_type = field.get('type', 'unknown')
            description = field.get('description', '설명 없음')
            
            # Add more detailed type information for date/timestamp fields
            type_info = field_type
            if name in ["transaction_date", "registration_date", "birth_date", "join_date"]:
                type_info = f"{field_type} (형식: YYYY-MM-DD)"
            elif name in ["event_timestamp", "login_timestamp", "last_activity_timestamp"]:
                type_info = f"{field_type} (형식: YYYY-MM-DD HH:MM:SS)"
                
            formatted.append(f"- {name}: {type_info} - {description}")
        
        return "\n".join(formatted)
    
    def _validate_table_column_references(self, sql_query: str) -> List[str]:
        """
        Validate table and column references in the SQL query.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            List of hallucinated table or column references
        """
        hallucinated_references = []
        
        # Extract all table references (including aliases)
        table_pattern = r'(?:FROM|JOIN)\s+`?' + self.dataset_id + r'\.([a-zA-Z0-9_]+)`?(?:\s+(?:AS\s+)?([a-zA-Z0-9_]+))?'
        table_matches = re.finditer(table_pattern, sql_query, re.IGNORECASE)
        
        # Track tables and their aliases in this query
        query_tables = {}
        query_aliases = {}
        
        for match in table_matches:
            table_name = match.group(1)
            alias = match.group(2) if match.group(2) else table_name
            
            # Check if the table exists
            if table_name not in self.valid_tables:
                hallucinated_references.append(f"테이블 '{table_name}'")
            else:
                query_tables[alias] = table_name
                query_aliases[table_name] = alias
        
        # Extract column references with table aliases
        column_pattern = r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)'
        column_matches = re.finditer(column_pattern, sql_query)
        
        for match in column_matches:
            alias = match.group(1)
            column = match.group(2)
            
            # Skip if it's not a table alias (might be a subquery alias)
            if alias not in query_tables and alias not in self.table_aliases:
                continue
                
            # Get the actual table name from the alias
            table_name = query_tables.get(alias, self.table_aliases.get(alias))
            
            # If we can't resolve the table, skip
            if not table_name:
                continue
                
            # Check if the column exists in this table
            if column not in self.valid_columns.get(table_name, set()):
                hallucinated_references.append(f"컬럼 '{alias}.{column}'")
        
        return hallucinated_references
    
    def _fix_hallucinated_references(self, sql_query: str) -> str:
        """
        Fix hallucinated table and column references in the SQL query.
        
        Args:
            sql_query: SQL query with potential hallucinations
            
        Returns:
            Fixed SQL query
        """
        logger.info("Fixing hallucinated references in SQL query")
        
        # 1. Fix table references
        # Map of common hallucinated table names to actual table names
        table_corrections = {
            "customers": "customer_master",
            "products": "product_master",
            "transactions": "offline_transactions",
            "online_transactions": "online_behavior",
            "online_activities": "online_behavior",
            "customer": "customer_master",
            "product": "product_master",
            "transaction": "offline_transactions",
            "online": "online_behavior"
        }
        
        # Replace incorrect table names
        for wrong, correct in table_corrections.items():
            # Only replace if it's not already correct and is actually used as a table name
            if wrong != correct and re.search(rf'(?:FROM|JOIN)\s+`?{self.dataset_id}\.{wrong}`?', sql_query, re.IGNORECASE):
                sql_query = re.sub(
                    rf'(?:FROM|JOIN)\s+`?{self.dataset_id}\.{wrong}`?', 
                    f' FROM `{self.dataset_id}.{correct}`', 
                    sql_query, 
                    flags=re.IGNORECASE
                )
        
        # 2. Fix column references
        # Map of common hallucinated column names to actual column names per table
        column_corrections = {
            "customer_master": {
                "birth_day": "birth_date",
                "registration_day": "registration_date",
                "signup_date": "registration_date",
                "gender_code": "gender",
                "age_group": "age_range"
            },
            "offline_transactions": {
                "purchase_date": "transaction_date",
                "amount": "total_amount",
                "transaction_amount": "total_amount",
                "purchase_amount": "total_amount"
            },
            "online_behavior": {
                "timestamp": "event_timestamp",
                "activity_timestamp": "event_timestamp",
                "session": "session_id",
                "activity_type": "event_type"
            },
            "product_master": {
                "category": "category_level_1",
                "subcategory": "category_level_2",
                "product_category": "category_level_1",
                "product_subcategory": "category_level_2"
            }
        }
        
        # Extract all table aliases from the query
        alias_pattern = r'(?:FROM|JOIN)\s+`?' + self.dataset_id + r'\.([a-zA-Z0-9_]+)`?(?:\s+(?:AS\s+)?([a-zA-Z0-9_]+))?'
        alias_matches = re.finditer(alias_pattern, sql_query, re.IGNORECASE)
        
        table_aliases = {}
        for match in alias_matches:
            table_name = match.group(1)
            alias = match.group(2) if match.group(2) else table_name
            table_aliases[alias] = table_name
            
            # Also handle corrected table names
            for wrong, correct in table_corrections.items():
                if table_name == wrong:
                    table_aliases[alias] = correct
        
        # Replace incorrect column names
        for alias, table in table_aliases.items():
            if table in column_corrections:
                for wrong, correct in column_corrections[table].items():
                    # Pattern to match the wrong column with this alias
                    pattern = rf'{alias}\.{wrong}\b'
                    if re.search(pattern, sql_query, re.IGNORECASE):
                        sql_query = re.sub(pattern, f'{alias}.{correct}', sql_query, flags=re.IGNORECASE)
        
        logger.info("Hallucinated reference fixes applied to SQL query")
        return sql_query
    
    def _regenerate_query(self, text_request: str, error_message: str) -> str:
        """
        Regenerate SQL query with additional guidance.
        
        Args:
            text_request: Original natural language request
            error_message: Error message to guide regeneration
            
        Returns:
            Regenerated SQL query
        """
        logger.info(f"Regenerating query with guidance: {error_message}")
        
        # Get table schema information
        customer_schema = self.bq_connector.get_table_schema(config.BQ_CUSTOMER_TABLE)
        transaction_schema = self.bq_connector.get_table_schema(config.BQ_OFFLINE_TRANSACTIONS_TABLE)
        online_schema = self.bq_connector.get_table_schema(config.BQ_ONLINE_BEHAVIOR_TABLE)
        product_schema = self.bq_connector.get_table_schema(config.BQ_PRODUCT_TABLE)
        
        # Format schema information for the prompt
        schema_info = f"""
        ## 테이블 스키마 정보
        
        1. customer_master 테이블 (고객 정보):
        {self._format_schema(customer_schema)}
        
        2. offline_transactions 테이블 (오프라인 거래 정보):
        {self._format_schema(transaction_schema)}
        
        3. online_behavior 테이블 (온라인 행동 정보):
        {self._format_schema(online_schema)}
        
        4. product_master 테이블 (제품 정보):
        {self._format_schema(product_schema)}
        
        ## 테이블 관계
        - customer_master.customer_id = offline_transactions.customer_id
        - customer_master.customer_id = online_behavior.customer_id
        - offline_transactions.product_id = product_master.product_id
        
        ## 중요: 사용 가능한 테이블과 컬럼
        - 위에 명시된 4개의 테이블과 컬럼만 사용할 수 있습니다.
        - 존재하지 않는 테이블이나 컬럼을 참조하면 쿼리가 실패합니다.
        - 각 테이블의 별칭은 다음과 같이 사용하세요: customer_master -> c, product_master -> p, offline_transactions -> t, online_behavior -> ob
        """
        
        # Create prompt with error guidance
        prompt = f"""
        당신은 가전 리테일 업체의 CDP 시스템에서 고객 세그먼테이션을 위한 SQL 쿼리를 생성하는 전문가입니다.
        다음 테이블 스키마 정보와 자연어 요청을 바탕으로 BigQuery SQL 쿼리를 작성해주세요.
        
        {schema_info}
        
        ## 자연어 요청
        {text_request}
        
        ## 이전 쿼리의 문제점
        {error_message}
        
        ## 중요 날짜 처리 가이드라인
        1. 모든 날짜/타임스탬프 관련 컬럼은 STRING 타입으로 저장되어 있습니다.
        2. 'YYYY-MM-DD' 형식의 날짜 문자열은 PARSE_DATE('%Y-%m-%d', column_name)을 사용하세요.
        3. 'YYYY-MM-DD HH:MM:SS' 형식의 타임스탬프 문자열은 PARSE_DATETIME('%Y-%m-%d %H:%M:%S', column_name)을 사용하세요.
        4. 날짜 비교는 DATE_DIFF() 함수를 사용하고, 타임스탬프 비교는 DATETIME_DIFF() 함수를 사용하세요.
        
        ## GROUP BY 가이드라인
        1. GROUP BY 절을 사용할 때는 집계 함수(COUNT, SUM, MAX, MIN, AVG 등)로 감싸지 않은 모든 SELECT 컬럼을 GROUP BY에 포함해야 합니다.
        2. 날짜 컬럼을 사용할 때는 특히 주의하세요. 예를 들어 t.transaction_date를 SELECT에서 사용한다면:
           - GROUP BY에 t.transaction_date를 포함하거나
           - MAX(t.transaction_date)나 MIN(t.transaction_date) 같은 집계 함수로 감싸야 합니다.
        
        ## 할루시네이션 방지 가이드라인
        1. 스키마에 명시된 테이블과 컬럼만 사용하세요. 존재하지 않는 테이블이나 컬럼을 참조하지 마세요.
        2. 사용 가능한 테이블: customer_master, product_master, offline_transactions, online_behavior
        3. 각 테이블에 있는 컬럼만 참조하세요. 예를 들어, customer_master 테이블에 없는 컬럼을 customer_master에서 조회하지 마세요.
        4. 테이블 별칭을 사용할 때는 일관성을 유지하세요: c (customer_master), p (product_master), t (offline_transactions), ob (online_behavior)
        
        ## 추가 가이드라인
        1. 쿼리는 BigQuery SQL 문법을 사용해야 합니다.
        2. 테이블 이름에는 데이터셋 ID를 포함해야 합니다 (예: `{self.dataset_id}.customer_master`).
        3. 쿼리 결과는 반드시 고객 ID(customer_id)와 관련 속성을 포함해야 합니다.
        4. 가능하면 세그먼트 이름이나 레이블을 포함하는 열을 생성해주세요.
        5. 쿼리 설명이나 주석 없이 SQL 쿼리만 제공해주세요.
        6. INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE와 같은 데이터 수정 문은 사용하지 마세요.
        7. 쿼리는 실행 가능하고 효율적이어야 합니다.
        """
        
        # Get SQL from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip()
        
        # Clean up the SQL query
        if sql_query.startswith("```sql"):
            sql_query = sql_query.split("```sql")[1]
        elif sql_query.startswith("```"):
            sql_query = sql_query.split("```")[1]
        
        if sql_query.endswith("```"):
            sql_query = sql_query.split("```")[0]
        
        sql_query = sql_query.strip()
        
        # Fix date handling in the regenerated SQL
        sql_query = self._fix_date_handling(sql_query)
        
        # Fix GROUP BY issues in the regenerated SQL
        sql_query = self._fix_group_by_issues(sql_query)
        
        # Fix hallucinated references
        sql_query = self._fix_hallucinated_references(sql_query)
        
        return sql_query
    
    def _fix_date_handling(self, sql_query: str) -> str:
        """
        Fix date handling in the generated SQL query.
        
        Args:
            sql_query: Generated SQL query
            
        Returns:
            SQL query with fixed date handling
        """
        logger.info("Fixing date handling in SQL query")
        
        # Fix patterns for simple date strings (YYYY-MM-DD)
        # Pattern: PARSE_DATETIME('%Y-%m-%d', date_column) -> PARSE_DATE('%Y-%m-%d', date_column)
        simple_date_pattern = r"PARSE_DATETIME\('%Y-%m-%d',\s*([^)]+)\)"
        sql_query = re.sub(simple_date_pattern, r"PARSE_DATE('%Y-%m-%d', \1)", sql_query)
        
        # Fix patterns for date columns with time component in column name but actually just dates
        date_columns = ["transaction_date", "registration_date", "birth_date", "join_date"]
        for col in date_columns:
            # Pattern: PARSE_DATETIME('%Y-%m-%d %H:%M:%S', date_column) -> PARSE_DATE('%Y-%m-%d', date_column)
            pattern = fr"PARSE_DATETIME\('%Y-%m-%d %H:%M:%S',\s*([^)]*{col}[^)]*)\)"
            sql_query = re.sub(pattern, fr"PARSE_DATE('%Y-%m-%d', \1)", sql_query)
        
        # Fix patterns for timestamp strings (YYYY-MM-DD HH:MM:SS)
        timestamp_columns = ["event_timestamp", "login_timestamp", "last_activity_timestamp"]
        for col in timestamp_columns:
            # Ensure PARSE_DATETIME is used for actual timestamp columns
            pattern = fr"PARSE_DATE\('%Y-%m-%d',\s*([^)]*{col}[^)]*)\)"
            sql_query = re.sub(pattern, fr"PARSE_DATETIME('%Y-%m-%d %H:%M:%S', \1)", sql_query)
        
        # Fix DATETIME_DIFF vs DATE_DIFF usage
        # Replace DATE_DIFF with DATETIME_DIFF when comparing DATETIME values
        datetime_diff_pattern = r"DATE_DIFF\(CURRENT_DATE\(\),\s*PARSE_DATETIME\("
        sql_query = re.sub(datetime_diff_pattern, r"DATETIME_DIFF(CURRENT_DATETIME(), PARSE_DATETIME(", sql_query)
        
        # Replace DATETIME_DIFF with DATE_DIFF when comparing DATE values
        date_diff_pattern = r"DATETIME_DIFF\(CURRENT_DATETIME\(\),\s*PARSE_DATE\("
        sql_query = re.sub(date_diff_pattern, r"DATE_DIFF(CURRENT_DATE(), PARSE_DATE(", sql_query)
        
        # Fix comparison between CURRENT_DATETIME and dates
        current_datetime_pattern = r"PARSE_DATE\('%Y-%m-%d',\s*([^)]+)\)\s*([<>=]+)\s*CURRENT_DATETIME\(\)"
        sql_query = re.sub(current_datetime_pattern, r"PARSE_DATE('%Y-%m-%d', \1) \2 CURRENT_DATE()", sql_query)
        
        # Fix comparison between CURRENT_DATE and timestamps
        current_date_pattern = r"PARSE_DATETIME\('%Y-%m-%d %H:%M:%S',\s*([^)]+)\)\s*([<>=]+)\s*CURRENT_DATE\(\)"
        sql_query = re.sub(current_date_pattern, r"PARSE_DATETIME('%Y-%m-%d %H:%M:%S', \1) \2 CURRENT_DATETIME()", sql_query)
        
        logger.info("Date handling fixes applied to SQL query")
        return sql_query
    
    def _fix_group_by_issues(self, sql_query: str) -> str:
        """
        Fix GROUP BY issues in the generated SQL query.
        
        Args:
            sql_query: Generated SQL query
            
        Returns:
            SQL query with fixed GROUP BY issues
        """
        logger.info("Fixing GROUP BY issues in SQL query")
        
        # If there's no GROUP BY, nothing to fix
        if "GROUP BY" not in sql_query.upper():
            return sql_query
        
        # Extract the main parts of the query
        try:
            # Split query into parts before and after GROUP BY
            before_group_by, after_group_by = re.split(r'GROUP\s+BY', sql_query, flags=re.IGNORECASE, maxsplit=1)
            
            # Extract SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)(?:FROM|$)', before_group_by, re.IGNORECASE | re.DOTALL)
            if not select_match:
                logger.warning("Could not parse SELECT clause for GROUP BY fixing")
                return sql_query
                
            select_clause = select_match.group(1).strip()
            
            # Extract GROUP BY columns
            group_by_clause = after_group_by.strip()
            # Handle cases where there's more SQL after GROUP BY (HAVING, ORDER BY, etc.)
            group_by_end = re.search(r'(HAVING|ORDER\s+BY|LIMIT|;|$)', group_by_clause, re.IGNORECASE)
            if group_by_end:
                group_by_end_pos = group_by_end.start()
                rest_of_query = group_by_clause[group_by_end_pos:]
                group_by_clause = group_by_clause[:group_by_end_pos].strip()
            else:
                rest_of_query = ""
            
            # Parse SELECT columns
            select_columns = []
            current_column = ""
            paren_level = 0
            
            for char in select_clause:
                if char == ',' and paren_level == 0:
                    select_columns.append(current_column.strip())
                    current_column = ""
                else:
                    current_column += char
                    if char == '(':
                        paren_level += 1
                    elif char == ')':
                        paren_level -= 1
            
            if current_column.strip():
                select_columns.append(current_column.strip())
            
            # Parse GROUP BY columns
            group_by_columns = [col.strip() for col in group_by_clause.split(',')]
            
            # Check each SELECT column to see if it needs to be in GROUP BY
            columns_to_add = []
            date_columns_to_fix = []
            
            for col in select_columns:
                # Skip columns that are already aggregated
                if re.search(r'(COUNT|SUM|AVG|MAX|MIN|ARRAY_AGG|STRING_AGG)\s*\(', col, re.IGNORECASE):
                    continue
                
                # Skip columns that are aliases to aggregated expressions
                if ' as ' in col.lower() and re.search(r'(COUNT|SUM|AVG|MAX|MIN|ARRAY_AGG|STRING_AGG)\s*\(', 
                                                     col.lower().split(' as ')[0], re.IGNORECASE):
                    continue
                
                # Extract column name (without alias)
                col_name = col.split(' as ')[0].strip() if ' as ' in col.lower() else col
                
                # Check if column is in GROUP BY
                if not any(col_name == group_col for group_col in group_by_columns):
                    # Special handling for date columns - wrap in MAX() instead of adding to GROUP BY
                    date_column_match = re.search(r'([a-zA-Z0-9_.]+\.(transaction_date|registration_date|birth_date|join_date|event_timestamp|login_timestamp|last_activity_timestamp))', col_name)
                    if date_column_match:
                        date_columns_to_fix.append((col, date_column_match.group(1)))
                    else:
                        # For other columns, add to GROUP BY
                        columns_to_add.append(col_name)
            
            # Fix the query if needed
            if columns_to_add or date_columns_to_fix:
                # Fix date columns by wrapping in MAX()
                modified_select_clause = select_clause
                for col, date_col in date_columns_to_fix:
                    # Replace the date column with MAX(date_column)
                    # Only replace exact matches to avoid partial replacements
                    pattern = r'(?<!\w)' + re.escape(date_col) + r'(?!\w)'
                    modified_select_clause = re.sub(pattern, f"MAX({date_col})", modified_select_clause)
                
                # Rebuild the query with modified SELECT and extended GROUP BY
                modified_before_group_by = before_group_by.replace(select_clause, modified_select_clause)
                
                if columns_to_add:
                    extended_group_by = group_by_clause + ", " + ", ".join(columns_to_add)
                    modified_query = f"{modified_before_group_by} GROUP BY {extended_group_by} {rest_of_query}"
                else:
                    modified_query = f"{modified_before_group_by} GROUP BY {group_by_clause} {rest_of_query}"
                
                logger.info("Fixed GROUP BY issues in query")
                return modified_query
            
            # No issues to fix
            return sql_query
            
        except Exception as e:
            logger.error(f"Error fixing GROUP BY issues: {str(e)}")
            # Return original query if there was an error in the fixing process
            return sql_query
    
    def _check_date_handling_issues(self, sql_query: str) -> str:
        """
        Check for potential date handling issues in the SQL query.
        
        Args:
            sql_query: SQL query to check
            
        Returns:
            String describing issues found, or empty string if no issues
        """
        issues = []
        
        # Check for PARSE_DATETIME used with date columns
        date_columns = ["transaction_date", "registration_date", "birth_date", "join_date"]
        for col in date_columns:
            if f"PARSE_DATETIME('%Y-%m-%d %H:%M:%S', {col})" in sql_query:
                issues.append(f"{col}은 날짜 형식(YYYY-MM-DD)이므로 PARSE_DATE를 사용해야 합니다.")
        
        # Check for PARSE_DATE used with timestamp columns
        timestamp_columns = ["event_timestamp", "login_timestamp", "last_activity_timestamp"]
        for col in timestamp_columns:
            if f"PARSE_DATE('%Y-%m-%d', {col})" in sql_query:
                issues.append(f"{col}은 타임스탬프 형식(YYYY-MM-DD HH:MM:SS)이므로 PARSE_DATETIME을 사용해야 합니다.")
        
        # Check for mismatched date comparisons
        if "PARSE_DATE" in sql_query and "CURRENT_DATETIME()" in sql_query:
            if re.search(r"PARSE_DATE\([^)]+\)\s*[<>=]+\s*CURRENT_DATETIME\(\)", sql_query):
                issues.append("PARSE_DATE 결과를 CURRENT_DATETIME()과 비교하고 있습니다. CURRENT_DATE()를 사용하세요.")
        
        if "PARSE_DATETIME" in sql_query and "CURRENT_DATE()" in sql_query:
            if re.search(r"PARSE_DATETIME\([^)]+\)\s*[<>=]+\s*CURRENT_DATE\(\)", sql_query):
                issues.append("PARSE_DATETIME 결과를 CURRENT_DATE()와 비교하고 있습니다. CURRENT_DATETIME()을 사용하세요.")
        
        # Check for mismatched diff functions
        if "DATE_DIFF" in sql_query and "PARSE_DATETIME" in sql_query:
            if re.search(r"DATE_DIFF\([^,]+,\s*PARSE_DATETIME\(", sql_query):
                issues.append("PARSE_DATETIME 결과에 DATE_DIFF를 사용하고 있습니다. DATETIME_DIFF를 사용하세요.")
        
        if "DATETIME_DIFF" in sql_query and "PARSE_DATE" in sql_query:
            if re.search(r"DATETIME_DIFF\([^,]+,\s*PARSE_DATE\(", sql_query):
                issues.append("PARSE_DATE 결과에 DATETIME_DIFF를 사용하고 있습니다. DATE_DIFF를 사용하세요.")
        
        return "; ".join(issues)
    
    def _check_group_by_issues(self, sql_query: str) -> str:
        """
        Check for potential GROUP BY issues in the SQL query.
        
        Args:
            sql_query: SQL query to check
            
        Returns:
            String describing issues found, or empty string if no issues
        """
        issues = []
        
        # If there's no GROUP BY, nothing to check
        if "GROUP BY" not in sql_query.upper():
            return ""
        
        try:
            # Split query into parts before and after GROUP BY
            before_group_by, after_group_by = re.split(r'GROUP\s+BY', sql_query, flags=re.IGNORECASE, maxsplit=1)
            
            # Extract SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)(?:FROM|$)', before_group_by, re.IGNORECASE | re.DOTALL)
            if not select_match:
                return "SELECT 절을 파싱할 수 없습니다."
                
            select_clause = select_match.group(1).strip()
            
            # Extract GROUP BY columns
            group_by_clause = after_group_by.strip()
            # Handle cases where there's more SQL after GROUP BY
            group_by_end = re.search(r'(HAVING|ORDER\s+BY|LIMIT|;|$)', group_by_clause, re.IGNORECASE)
            if group_by_end:
                group_by_clause = group_by_clause[:group_by_end.start()].strip()
            
            # Parse SELECT columns
            select_columns = []
            current_column = ""
            paren_level = 0
            
            for char in select_clause:
                if char == ',' and paren_level == 0:
                    select_columns.append(current_column.strip())
                    current_column = ""
                else:
                    current_column += char
                    if char == '(':
                        paren_level += 1
                    elif char == ')':
                        paren_level -= 1
            
            if current_column.strip():
                select_columns.append(current_column.strip())
            
            # Parse GROUP BY columns
            group_by_columns = [col.strip() for col in group_by_clause.split(',')]
            
            # Check each SELECT column to see if it needs to be in GROUP BY
            for col in select_columns:
                # Skip columns that are already aggregated
                if re.search(r'(COUNT|SUM|AVG|MAX|MIN|ARRAY_AGG|STRING_AGG)\s*\(', col, re.IGNORECASE):
                    continue
                
                # Skip columns that are aliases to aggregated expressions
                if ' as ' in col.lower() and re.search(r'(COUNT|SUM|AVG|MAX|MIN|ARRAY_AGG|STRING_AGG)\s*\(', 
                                                     col.lower().split(' as ')[0], re.IGNORECASE):
                    continue
                
                # Extract column name (without alias)
                col_name = col.split(' as ')[0].strip() if ' as ' in col.lower() else col
                
                # Check if column is in GROUP BY
                if not any(col_name == group_col for group_col in group_by_columns):
                    # Special check for date columns
                    date_column_match = re.search(r'([a-zA-Z0-9_.]+\.(transaction_date|registration_date|birth_date|join_date|event_timestamp|login_timestamp|last_activity_timestamp))', col_name)
                    if date_column_match:
                        issues.append(f"{date_column_match.group(1)}은(는) GROUP BY에 포함되어 있지 않고 집계 함수로 감싸져 있지도 않습니다. MAX() 함수로 감싸거나 GROUP BY에 추가하세요.")
                    else:
                        issues.append(f"{col_name}은(는) GROUP BY에 포함되어 있지 않고 집계 함수로 감싸져 있지도 않습니다.")
            
            return "; ".join(issues)
            
        except Exception as e:
            return f"GROUP BY 이슈 확인 중 오류 발생: {str(e)}"
    
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
        
        # Check for date handling issues
        date_issues = self._check_date_handling_issues(sql_query)
        if date_issues:
            logger.warning(f"Date handling issues found in query: {date_issues}")
            # Note: We're not raising an error here, just logging a warning,
            # since the _fix_date_handling method should have addressed these issues
        
        # Check for potential GROUP BY issues
        group_by_issues = self._check_group_by_issues(sql_query)
        if group_by_issues:
            logger.warning(f"GROUP BY issues found in query: {group_by_issues}")
            # Note: We're not raising an error here, just logging a warning,
            # since the _fix_group_by_issues method should have addressed these issues
        
        # Check for hallucinated references
        hallucination_issues = self._validate_table_column_references(sql_query)
        if hallucination_issues:
            logger.warning(f"Hallucination issues found in query: {', '.join(hallucination_issues)}")
            # Note: We're not raising an error here, just logging a warning,
            # since the _fix_hallucinated_references method should have addressed these issues
        
        logger.info("SQL query validation passed")
    
    def _run_default_segmentation(self) -> None:
        """Run default segmentation strategies."""
        logger.info("Running default segmentation")
        
        # 1. RFM Segmentation
        rfm_query = f"""
        WITH rfm_data AS (
            SELECT
                c.customer_id,
                DATE_DIFF(CURRENT_DATE(), MAX(PARSE_DATE('%Y-%m-%d', t.transaction_date)), DAY) as recency,
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
        
        # Apply date handling fixes to default queries
        rfm_query = self._fix_date_handling(rfm_query)
        rfm_segments = self.execute_query(rfm_query)
        self.segments["rfm"] = rfm_segments
        
        # 2. Lifecycle Segmentation
        lifecycle_query = f"""
        WITH customer_activity AS (
            SELECT
                c.customer_id,
                c.registration_date,
                DATE_DIFF(CURRENT_DATE(), PARSE_DATE('%Y-%m-%d', c.registration_date), DAY) as days_since_registration,
                MAX(PARSE_DATE('%Y-%m-%d', t.transaction_date)) as last_purchase_date,
                DATE_DIFF(CURRENT_DATE(), MAX(PARSE_DATE('%Y-%m-%d', t.transaction_date)), DAY) as days_since_last_purchase,
                COUNT(DISTINCT t.transaction_id) as purchase_count,
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
        
        lifecycle_query = self._fix_date_handling(lifecycle_query)
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
                # 커스텀 세그먼트의 경우, 흔히 쓰는 컬럼명을 우선 사용하고
                # 없으면 'segment' 라는 이름의 컬럼을 시도
                if "custom_segment" in segment_data.columns:
                    segment_col = "custom_segment"
                else:
                    potential_cols = [col for col in segment_data.columns if "segment" in col.lower()]
                    segment_col = potential_cols[0] if potential_cols else None
            else:
                # 최후의 보루 – 일반적인 이름
                segment_col = "segment"
            
            # 실제 컬럼 존재 여부 확인
            if segment_col and segment_col in segment_data.columns:
                # -------------------
                # Pie chart (분포)
                # -------------------
                try:
                    fig = create_pie_chart(
                        segment_data,
                        segment_col,
                        title=f"{segment_type.capitalize()} 세그먼트 분포"
                    )
                    chart_path = save_plotly_fig(fig, f"{segment_type}_pie_chart")
                    self.visualizations.append(
                        {
                            "segment_type": segment_type,
                            "chart_type": "pie",
                            "path": chart_path,
                            "title": f"{segment_type.capitalize()} 세그먼트 분포"
                        }
                    )
                except Exception as e:
                    logger.warning(f"Pie chart 생성 실패 ({segment_type}): {e}")
                
                # -------------------
                # Bar chart (금액/값)
                # -------------------
                value_col = None
                if "monetary" in segment_data.columns:
                    value_col = "monetary"
                elif "total_amount" in segment_data.columns:
                    value_col = "total_amount"
                
                if value_col:
                    try:
                        fig = create_bar_chart(
                            segment_data,
                            x_col=segment_col,
                            y_col=value_col,
                            title=f"{segment_type.capitalize()} 세그먼트별 {value_col}"
                        )
                        chart_path = save_plotly_fig(fig, f"{segment_type}_bar_chart")
                        self.visualizations.append(
                            {
                                "segment_type": segment_type,
                                "chart_type": "bar",
                                "path": chart_path,
                                "title": f"{segment_type.capitalize()} 세그먼트별 {value_col}"
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Bar chart 생성 실패 ({segment_type}): {e}")
                
                logger.info(f"Created visualizations for {segment_type} segments")
            else:
                logger.warning(
                    f"Could not create visualization for {segment_type}: "
                    f"segment column '{segment_col}' not found in data"
                )

    def _generate_report(self) -> None:
        """Generate a simple HTML report that summarizes all segmentation outputs."""
        logger.info("Generating segmentation report")

        # Lazy imports (report 생성 시에만 필요)
        from datetime import datetime

        # 1) 리포트 저장 경로 준비
        report_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = os.path.join(report_dir, f"segmentation_report_{timestamp}.html")

        def _escape_html(txt: str) -> str:
            """간단한 HTML escape."""
            return (
                txt.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
            )

        # 2) HTML 시작
        html_parts: List[str] = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Segmentation Report</title>",
            "<style>",
            "body{font-family:Arial,Helvetica,sans-serif;margin:20px;}",
            "h1,h2,h3{color:#2c3e50;}",
            "table{border-collapse:collapse;width:100%;margin:10px 0;}",
            "th,td{border:1px solid #ddd;padding:6px;text-align:left;}",
            "th{background:#f4f4f4;}",
            ".warning{color:#e74c3c;}",
            ".recommendation{color:#27ae60;}",
            ".viz img{max-width:100%;height:auto;}",
            "</style></head><body>",
            f"<h1>고객 세그먼테이션 리포트</h1>",
            f"<p>생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        # 3) 세그먼트별 정보 정리
        for segment_type, segment_data in self.segments.items():
            html_parts.append(f"<h2>{segment_type.upper()} 세그먼테이션</h2>")

            # (a) 사용된 쿼리
            query_txt = self.queries.get(segment_type)
            if query_txt:
                html_parts.append("<details><summary>사용된 쿼리 보기</summary><pre>")
                html_parts.append(_escape_html(query_txt))
                html_parts.append("</pre></details>")

            # (b) 분포 테이블
            counts = self.segment_analysis.get(segment_type, {}).get("segment_counts", {})
            if counts:
                html_parts.append("<h3>세그먼트 분포</h3>")
                html_parts.append("<table><tr><th>세그먼트</th><th>고객 수</th><th>비율</th></tr>")
                total_cnt = sum(counts.values()) or 1
                for name, cnt in counts.items():
                    pct = cnt / total_cnt * 100
                    html_parts.append(
                        f"<tr><td>{_escape_html(str(name))}</td><td>{cnt}</td><td>{pct:.1f}%</td></tr>"
                    )
                html_parts.append("</table>")

            # (c) 인사이트
            insights = self.segment_analysis.get(segment_type, {}).get("insights")
            if insights:
                html_parts.append("<h3>세그먼트 인사이트</h3>")
                html_parts.append(
                    "<p>" + _escape_html(insights).replace("\n", "<br>") + "</p>"
                )

            # (d) 검증 결과
            validation = self.segment_validation.get(segment_type, {})
            warns = validation.get("warnings", [])
            recs = validation.get("recommendations", [])
            if warns:
                html_parts.append("<h3>주의사항</h3><ul class='warning'>")
                html_parts.extend([f"<li>{_escape_html(w)}</li>" for w in warns])
                html_parts.append("</ul>")
            if recs:
                html_parts.append("<h3>개선 제안</h3><ul class='recommendation'>")
                html_parts.extend([f"<li>{_escape_html(r)}</li>" for r in recs])
                html_parts.append("</ul>")

            # (e) 시각화
            viz_list = [v for v in self.visualizations if v["segment_type"] == segment_type]
            if viz_list:
                html_parts.append("<div class='viz'>")
                for v in viz_list:
                    rel_path = os.path.relpath(v["path"], report_dir)
                    html_parts.append(
                        f"<figure><img src='{_escape_html(rel_path)}' "
                        f"alt='{_escape_html(v['title'])}'><figcaption>{_escape_html(v['title'])}</figcaption></figure>"
                    )
                html_parts.append("</div>")

        # 4) HTML 종료
        html_parts.append("</body></html>")

        # 5) 파일 저장
        with open(self.report_path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(html_parts))

        logger.info(f"Segmentation report generated: {self.report_path}")
