"""
Text2SQL converter for the Segmentation Agent.
This module provides functionality to convert natural language requests
into SQL queries for customer segmentation.
"""
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union

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

class Text2SQLConverter:
    """
    Converts natural language requests into SQL queries for customer segmentation.
    
    This class handles the process of taking a user's natural language request
    for customer segmentation and converting it into a valid BigQuery SQL query
    that can be executed against the CDP database.
    """
    

    ...

    def convert(self, text_request: str) -> str:
        ...
        # 기존 흐름 유지 후, 마지막에 날짜 문자열 처리 추가



    
    def __init__(
        self,
        project_id: str = None,
        dataset_id: str = None,
        bq_connector: BigQueryConnector = None
    ):
        """
        Initialize the Text2SQL converter.
        
        Args:
            project_id: Google Cloud project ID (if None, uses the value from config)
            dataset_id: BigQuery dataset ID (if None, uses the value from config)
            bq_connector: Optional existing BigQuery connector to reuse
        """
        self.project_id = project_id or config.GCP_PROJECT_ID
        self.dataset_id = dataset_id or config.BQ_DATASET_ID
        
        # Initialize BigQuery connector if not provided
        self.bq_connector = bq_connector or BigQueryConnector(project_id=self.project_id)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_TEMPERATURE,
            top_p=config.GEMINI_TOP_P,
            top_k=config.GEMINI_TOP_K,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
            google_api_key=config.GEMINI_API_KEY,
        )
        
        # Cache for schema information
        self.schema_cache = {}
        
        logger.info(f"Text2SQL converter initialized for dataset {self.dataset_id}")
    
    def convert(self, text_request: str) -> str:
        """
        Convert natural language request to SQL query.
        
        Args:
            text_request: Natural language segmentation request
            
        Returns:
            SQL query string
        """
        logger.info(f"Converting text to SQL: {text_request}")
        
        # Get schema information
        schema_info = self._get_schema_info()
        
        # Create prompt for SQL generation
        prompt = self._create_conversion_prompt(text_request, schema_info)
        
        # Get SQL from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip()
        
        # Clean up the SQL query (remove markdown formatting if present)
        sql_query = self._clean_sql_query(sql_query)
        
        # Validate and refine the query
        sql_query = self._validate_and_refine_query(sql_query, text_request)
        
        logger.info(f"Generated SQL query: {sql_query[:100]}...")
        
        # self.fix_parse_datetime_for_short_dates(sql_query)
        print('########################')
        # if 'PARSE_DATETIME' in sql_query:
        # sql_query = self._wrap_date_literals(sql_query)
        sql_query= sql_query.replace("PARSE_DATETIME('%Y-%m-%d %H:%M:%S', t.transaction_date)",
                          "PARSE_DATE('%Y-%m-%d', t.transaction_date)")
        
        return sql_query
    def _get_schema_info(self) -> str:
        """
        Get database schema information for the prompt.
        
        Returns:
            Formatted schema information string
        """
        # Check if schema is already cached
        if self.schema_cache:
            return self.schema_cache.get("formatted_schema", "")
        
        # Get table schema information
        try:
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
            
            ## 테이블 관계
            - customer_master.customer_id = offline_transactions.customer_id
            - customer_master.customer_id = online_behavior.customer_id
            - offline_transactions.product_id = product_master.product_id
            """
            
            # Cache the schema
            self.schema_cache["formatted_schema"] = schema_info
            self.schema_cache["tables"] = {
                "customer_master": customer_schema,
                "offline_transactions": transaction_schema,
                "online_behavior": online_schema,
                "product_master": product_schema
            }
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting schema information: {str(e)}")
            return "스키마 정보를 가져오는 중 오류가 발생했습니다."
    
    def _format_schema(self, schema: List[Dict[str, Any]]) -> str:
        """
        Format table schema for the prompt.
        
        Args:
            schema: List of field dictionaries from BigQuery
            
        Returns:
            Formatted schema string
        """
        if not schema:
            return "스키마 정보가 없습니다."
        
        formatted = []
        for field in schema:
            formatted.append(f"- {field.get('name', 'unknown')}: {field.get('type', 'unknown')} - {field.get('description', '설명 없음')}")
        
        return "\n".join(formatted)
    
    def _create_conversion_prompt(self, text_request: str, schema_info: str) -> str:
        """
        Create prompt for SQL generation.
        
        Args:
            text_request: Natural language request
            schema_info: Formatted schema information
            
        Returns:
            Prompt string for LLM
        """
        # Add example templates to help guide the model
        example_templates = """
        ## 예시 템플릿
        
        1. RFM 세그먼테이션:
        ```sql
        WITH rfm_data AS (
            SELECT
                c.customer_id,
                DATETIME_DIFF(CURRENT_DATETIME(), MAX(PARSE_DATETIME('%Y-%m-%d %H:%M:%S', t.transaction_date)), DAY) as recency,
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
            DATETIME_DIFF(CURRENT_DATETIME(), PARSE_DATETIME('%Y-%m-%d', c.registration_date), DAY) as days_since_registration,
            MAX(PARSE_DATETIME('%Y-%m-%d %H:%M:%S', t.transaction_date)) as last_purchase_date,
            DATETIME_DIFF(CURRENT_DATETIME(), MAX(PARSE_DATETIME('%Y-%m-%d %H:%M:%S', t.transaction_date)), DAY) as days_since_last_purchase,
            COUNT(DISTINCT t.transaction_id) as purchase_count,
            CASE
                WHEN DATETIME_DIFF(CURRENT_DATETIME(), PARSE_DATETIME('%Y-%m-%d', c.registration_date), DAY) <= 30 THEN '신규 고객'
                WHEN DATETIME_DIFF(CURRENT_DATETIME(), MAX(PARSE_DATETIME('%Y-%m-%d %H:%M:%S', t.transaction_date)), DAY) <= 30 THEN '활성 고객'
                WHEN DATETIME_DIFF(CURRENT_DATETIME(), MAX(PARSE_DATETIME('%Y-%m-%d %H:%M:%S', t.transaction_date)), DAY) <= 90 THEN '준활성 고객'
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
        """
        
        # Create the prompt
        prompt = f"""
        당신은 가전 리테일 업체의 CDP 시스템에서 고객 세그먼테이션을 위한 SQL 쿼리를 생성하는 전문가입니다.
        다음 테이블 스키마 정보와 자연어 요청을 바탕으로 BigQuery SQL 쿼리를 작성해주세요.
        
        {schema_info}
        
        {example_templates}
        
        ## 자연어 요청
        {text_request}
        
        다음 가이드라인을 따라주세요:
        1. 쿼리는 BigQuery SQL 문법을 사용해야 합니다.
        2. 테이블 이름에는 데이터셋 ID를 포함해야 합니다 (예: `{self.dataset_id}.customer_master`).
        3. 쿼리 결과는 고객 ID(customer_id)와 관련 속성을 포함해야 합니다.
        4. 가능하면 세그먼트 이름이나 레이블을 포함하는 열을 생성해주세요.
        5. 쿼리 설명이나 주석 없이 SQL 쿼리만 제공해주세요.
        6. 날짜 형식 변환이 필요한 경우 PARSE_DATETIME 또는 PARSE_DATE 함수를 사용하세요.
        7. 시간 간격 계산이 필요한 경우 DATETIME_DIFF 함수를 사용하세요.
        8. 쿼리는 실행 가능하고 효율적이어야 합니다.
        """
        
        return prompt
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """
        Clean up the SQL query by removing markdown formatting.
        
        Args:
            sql_query: Raw SQL query from LLM
            
        Returns:
            Cleaned SQL query
        """
        # Remove markdown code block formatting if present
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1]
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1]
        
        # Remove trailing markdown markers
        if "```" in sql_query:
            sql_query = sql_query.split("```")[0]
        
        # Remove any explanatory text before or after the query
        lines = sql_query.strip().split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Check if this line looks like SQL
            if (in_sql or 
                stripped.startswith("SELECT") or 
                stripped.startswith("WITH") or
                stripped.startswith("CREATE") or
                stripped.startswith("--")):
                in_sql = True
                sql_lines.append(line)
        
        # If we didn't find any SQL lines, return the original (might be a one-liner)
        if not sql_lines:
            return sql_query.strip()
        
        return '\n'.join(sql_lines).strip()
    
    def _validate_and_refine_query(self, sql_query: str, text_request: str) -> str:
        """
        Validate and refine the generated SQL query.
        
        Args:
            sql_query: Generated SQL query
            text_request: Original natural language request
            
        Returns:
            Validated and refined SQL query
        """
        # Check for data modification statements
        modification_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        for keyword in modification_keywords:
            if f" {keyword} " in f" {sql_query.upper()} ":
                logger.warning(f"Query contains forbidden keyword: {keyword}")
                sql_query = self._regenerate_query(text_request, f"쿼리에 금지된 키워드가 포함되어 있습니다: {keyword}")
                break
        
        # Ensure query includes customer_id
        if "customer_id" not in sql_query.lower() and "SELECT" in sql_query.upper():
            logger.warning("Query does not include customer_id column")
            sql_query = self._regenerate_query(text_request, "쿼리에 customer_id 컬럼이 포함되어야 합니다.")
        
        # Ensure dataset_id is used in table references
        if self.dataset_id not in sql_query and ("FROM" in sql_query.upper() or "JOIN" in sql_query.upper()):
            logger.warning(f"Query does not include dataset_id: {self.dataset_id}")
            sql_query = self._regenerate_query(text_request, f"테이블 이름에 데이터셋 ID({self.dataset_id})를 포함해야 합니다.")
        
        # Replace template placeholders if present
        sql_query = sql_query.replace("{dataset_id}", self.dataset_id)
        
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
        
        # Get schema information
        schema_info = self._get_schema_info()
        
        # Create prompt with error guidance
        prompt = f"""
        당신은 가전 리테일 업체의 CDP 시스템에서 고객 세그먼테이션을 위한 SQL 쿼리를 생성하는 전문가입니다.
        다음 테이블 스키마 정보와 자연어 요청을 바탕으로 BigQuery SQL 쿼리를 작성해주세요.
        
        {schema_info}
        
        ## 자연어 요청
        {text_request}
        
        ## 이전 쿼리의 문제점
        {error_message}
        
        다음 가이드라인을 따라주세요:
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
        sql_query = self._clean_sql_query(sql_query)
        
        return sql_query
    
    def explain_query(self, sql_query: str) -> str:
        """
        Generate a natural language explanation of the SQL query.
        
        Args:
            sql_query: SQL query to explain
            
        Returns:
            Natural language explanation
        """
        logger.info("Generating SQL query explanation")
        
        # Create prompt for explanation
        prompt = f"""
        당신은 가전 리테일 업체의 CDP 시스템에서 고객 세그먼테이션을 위한 SQL 쿼리를 설명하는 전문가입니다.
        다음 SQL 쿼리를 비기술적인 마케팅 담당자도 이해할 수 있도록 설명해주세요.
        
        ```sql
        {sql_query}
        ```
        
        다음 내용을 포함해서 설명해주세요:
        1. 이 쿼리가 어떤 고객 세그먼트를 찾는지
        2. 세그먼트의 주요 특성은 무엇인지
        3. 어떤 데이터를 사용하는지
        4. 세그먼트가 어떻게 정의되는지
        
        전문 용어를 최소화하고 쉬운 언어로 설명해주세요.
        """
        
        # Get explanation from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        explanation = response.content.strip()
        
        return explanation
    
    def suggest_segmentation_approaches(self, business_goal: str) -> List[Dict[str, str]]:
        """
        Suggest segmentation approaches based on business goal.
        
        Args:
            business_goal: Business goal or objective
            
        Returns:
            List of suggested segmentation approaches
        """
        logger.info(f"Suggesting segmentation approaches for goal: {business_goal}")
        
        # Create prompt for suggestions
        prompt = f"""
        당신은 가전 리테일 업체의 고객 세그먼테이션 전문가입니다.
        다음 비즈니스 목표에 적합한 고객 세그먼테이션 접근법을 3-5가지 제안해주세요.
        
        ## 비즈니스 목표
        {business_goal}
        
        각 접근법에 대해 다음 형식으로 작성해주세요:
        1. 세그먼테이션 이름: [이름]
        2. 설명: [설명]
        3. 필요한 데이터: [데이터]
        4. SQL 쿼리 예시: [쿼리]
        
        각 접근법은 구체적이고 실행 가능해야 합니다.
        SQL 쿼리는 BigQuery SQL 문법을 사용하고, 테이블 이름에는 {self.dataset_id} 데이터셋 ID를 포함해야 합니다.
        """
        
        # Get suggestions from LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        suggestions_text = response.content
        
        # Parse suggestions
        suggestions = []
        current_suggestion = {}
        
        # Split by numbered sections
        sections = re.split(r'\n\d+\.', suggestions_text)
        
        for section in sections[1:]:  # Skip the first empty section
            lines = section.strip().split('\n')
            suggestion = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith("세그먼테이션 이름:"):
                    suggestion["name"] = line.split(":", 1)[1].strip()
                elif line.startswith("설명:"):
                    suggestion["description"] = line.split(":", 1)[1].strip()
                elif line.startswith("필요한 데이터:"):
                    suggestion["required_data"] = line.split(":", 1)[1].strip()
                elif line.startswith("SQL 쿼리 예시:"):
                    # The SQL query might span multiple lines
                    sql_start_idx = lines.index(line)
                    sql_lines = []
                    
                    for sql_line in lines[sql_start_idx+1:]:
                        if sql_line.strip() and not sql_line.startswith("세그먼테이션 이름:"):
                            sql_lines.append(sql_line)
                    
                    suggestion["sql_example"] = "\n".join(sql_lines)
            
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    def validate_query_execution(self, sql_query: str) -> Tuple[bool, str]:
        """
        Validate if a query can be executed safely.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        logger.info("Validating query execution safety")
        
        # Check for data modification statements
        modification_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        for keyword in modification_keywords:
            if f" {keyword} " in f" {sql_query.upper()} ":
                return False, f"쿼리에 금지된 키워드가 포함되어 있습니다: {keyword}"
        
        # Ensure query includes customer_id
        if "customer_id" not in sql_query.lower() and "SELECT" in sql_query.upper():
            return False, "쿼리에 customer_id 컬럼이 포함되어야 합니다."
        
        # Ensure dataset_id is used in table references
        if self.dataset_id not in sql_query and ("FROM" in sql_query.upper() or "JOIN" in sql_query.upper()):
            return False, f"테이블 이름에 데이터셋 ID({self.dataset_id})를 포함해야 합니다."
        
        # Check for proper table references
        tables = ["customer_master", "offline_transactions", "online_behavior", "product_master"]
        found_tables = False
        
        for table in tables:
            if f"{self.dataset_id}.{table}" in sql_query:
                found_tables = True
                break
        
        if not found_tables:
            return False, f"쿼리에 유효한 테이블 참조가 없습니다. ({', '.join([f'{self.dataset_id}.{t}' for t in tables])})"
        
        # Check for basic SQL syntax
        if "SELECT" not in sql_query.upper():
            return False, "쿼리에 SELECT 문이 없습니다."
        
        if "FROM" not in sql_query.upper():
            return False, "쿼리에 FROM 절이 없습니다."
        
        # All checks passed
        return True, "유효한 쿼리입니다."

# For direct testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create converter
    converter = Text2SQLConverter()
    
    # Test conversion
    text_request = "최근 3개월 내에 TV 카테고리 제품을 구매한 30-40대 여성 고객을 찾아주세요."
    sql_query = converter.convert(text_request)
    
    print("Generated SQL Query:")
    print(sql_query)
    
    # Test explanation
    explanation = converter.explain_query(sql_query)
    print("\nQuery Explanation:")
    print(explanation)
