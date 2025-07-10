"""
BigQuery connector utility for the CRM-Agent system.
Provides functions to interact with Google BigQuery and execute queries.
"""
import os
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
import logging
from datetime import datetime, timedelta

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BigQueryConnector:
    """Connector class for Google BigQuery operations."""
    
    def __init__(self, project_id: str = None, credentials_path: str = None):
        """
        Initialize BigQuery client.
        
        Args:
            project_id: Google Cloud project ID. If None, uses the value from config.
            credentials_path: Path to service account credentials file. If None, uses the value from config.
        """
        self.project_id = project_id or config.GCP_PROJECT_ID
        
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        try:
            self.client = bigquery.Client(project=self.project_id)
            logger.info(f"Connected to BigQuery project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to connect to BigQuery: {str(e)}")
            raise
    
    def get_dataset_info(self, dataset_id: str = None) -> Dict[str, Any]:
        """
        Get information about a BigQuery dataset.
        
        Args:
            dataset_id: Dataset ID. If None, uses the value from config.
            
        Returns:
            Dictionary with dataset information
        """
        dataset_id = dataset_id or config.BQ_DATASET_ID
        dataset_ref = f"{self.project_id}.{dataset_id}"
        
        try:
            dataset = self.client.get_dataset(dataset_ref)
            return {
                "dataset_id": dataset.dataset_id,
                "project": dataset.project,
                "created": dataset.created,
                "modified": dataset.modified,
                "description": dataset.description,
                "location": dataset.location,
                "labels": dataset.labels,
            }
        except GoogleAPIError as e:
            logger.error(f"Failed to get dataset info for {dataset_id}: {str(e)}")
            raise
    
    def list_tables(self, dataset_id: str = None) -> List[Dict[str, Any]]:
        """
        List all tables in a dataset.
        
        Args:
            dataset_id: Dataset ID. If None, uses the value from config.
            
        Returns:
            List of dictionaries with table information
        """
        dataset_id = dataset_id or config.BQ_DATASET_ID
        dataset_ref = f"{self.project_id}.{dataset_id}"
        
        try:
            tables = list(self.client.list_tables(dataset_ref))
            return [
                {
                    "table_id": table.table_id,
                    "full_table_id": f"{table.project}.{table.dataset_id}.{table.table_id}",
                    "created": table.created,
                    "modified": table.modified,
                    "num_rows": self.client.get_table(table).num_rows,
                    "size_bytes": self.client.get_table(table).num_bytes,
                }
                for table in tables
            ]
        except GoogleAPIError as e:
            logger.error(f"Failed to list tables in dataset {dataset_id}: {str(e)}")
            raise
    
    def get_table_schema(self, table_id: str, dataset_id: str = None) -> List[Dict[str, str]]:
        """
        Get schema information for a table.
        
        Args:
            table_id: Table ID
            dataset_id: Dataset ID. If None, uses the value from config.
            
        Returns:
            List of dictionaries with column information
        """
        dataset_id = dataset_id or config.BQ_DATASET_ID
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        
        try:
            table = self.client.get_table(table_ref)
            return [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description,
                }
                for field in table.schema
            ]
        except GoogleAPIError as e:
            logger.error(f"Failed to get schema for table {table_id}: {str(e)}")
            raise
    
    def get_table_sample(self, table_id: str, dataset_id: str = None, limit: int = 10) -> pd.DataFrame:
        """
        Get a sample of data from a table.
        
        Args:
            table_id: Table ID
            dataset_id: Dataset ID. If None, uses the value from config.
            limit: Maximum number of rows to return
            
        Returns:
            Pandas DataFrame with sample data
        """
        dataset_id = dataset_id or config.BQ_DATASET_ID
        query = f"""
        SELECT * FROM `{self.project_id}.{dataset_id}.{table_id}`
        LIMIT {limit}
        """
        
        return self.run_query(query)
    
    def run_query(self, query: str) -> pd.DataFrame:
        """
        Run a SQL query and return results as a DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            Pandas DataFrame with query results
        """
        try:
            logger.debug(f"Running query: {query}")
            query_job = self.client.query(query)
            results = query_job.result()
            return results.to_dataframe()
        except GoogleAPIError as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            raise
    
    def get_table_stats(self, table_id: str, dataset_id: str = None) -> Dict[str, Any]:
        """
        Get statistics about a table.
        
        Args:
            table_id: Table ID
            dataset_id: Dataset ID. If None, uses the value from config.
            
        Returns:
            Dictionary with table statistics
        """
        dataset_id = dataset_id or config.BQ_DATASET_ID
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        
        try:
            table = self.client.get_table(table_ref)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM `{table_ref}`"
            count_result = self.run_query(count_query)
            row_count = count_result.iloc[0]['count']
            
            return {
                "table_id": table.table_id,
                "row_count": row_count,
                "size_bytes": table.num_bytes,
                "created": table.created,
                "modified": table.modified,
                "schema_fields": len(table.schema),
            }
        except GoogleAPIError as e:
            logger.error(f"Failed to get stats for table {table_id}: {str(e)}")
            raise

    # CDP-specific utility functions
    
    def get_customer_data(self, customer_ids: List[str] = None, limit: int = None) -> pd.DataFrame:
        """
        Get customer data from the customer master table.
        
        Args:
            customer_ids: Optional list of customer IDs to filter by
            limit: Maximum number of rows to return
            
        Returns:
            Pandas DataFrame with customer data
        """
        query = f"""
        SELECT * FROM `{self.project_id}.{config.BQ_DATASET_ID}.{config.BQ_CUSTOMER_TABLE}`
        """
        
        if customer_ids:
            customer_ids_str = ", ".join([f"'{cid}'" for cid in customer_ids])
            query += f" WHERE customer_id IN ({customer_ids_str})"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.run_query(query)
    
    def get_product_data(self, product_ids: List[str] = None, categories: List[str] = None, limit: int = None) -> pd.DataFrame:
        """
        Get product data from the product master table.
        
        Args:
            product_ids: Optional list of product IDs to filter by
            categories: Optional list of category_level_1 values to filter by
            limit: Maximum number of rows to return
            
        Returns:
            Pandas DataFrame with product data
        """
        query = f"""
        SELECT * FROM `{self.project_id}.{config.BQ_DATASET_ID}.{config.BQ_PRODUCT_TABLE}`
        """
        
        conditions = []
        if product_ids:
            product_ids_str = ", ".join([f"'{pid}'" for pid in product_ids])
            conditions.append(f"product_id IN ({product_ids_str})")
        
        if categories:
            categories_str = ", ".join([f"'{cat}'" for cat in categories])
            conditions.append(f"category_level_1 IN ({categories_str})")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.run_query(query)
    
    def get_transactions_data(
        self, 
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        customer_ids: List[str] = None,
        product_ids: List[str] = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Get transaction data from the offline transactions table.
        
        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            customer_ids: Optional list of customer IDs to filter by
            product_ids: Optional list of product IDs to filter by
            limit: Maximum number of rows to return
            
        Returns:
            Pandas DataFrame with transaction data
        """
        query = f"""
        SELECT * FROM `{self.project_id}.{config.BQ_DATASET_ID}.{config.BQ_OFFLINE_TRANSACTIONS_TABLE}`
        """
        
        conditions = []
        
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            conditions.append(f"transaction_date >= '{start_date}'")
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            conditions.append(f"transaction_date <= '{end_date}'")
        
        if customer_ids:
            customer_ids_str = ", ".join([f"'{cid}'" for cid in customer_ids])
            conditions.append(f"customer_id IN ({customer_ids_str})")
        
        if product_ids:
            product_ids_str = ", ".join([f"'{pid}'" for pid in product_ids])
            conditions.append(f"product_id IN ({product_ids_str})")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.run_query(query)
    
    def get_online_behavior_data(
        self,
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        customer_ids: List[str] = None,
        product_ids: List[str] = None,
        event_types: List[str] = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Get online behavior data from the online behavior table.
        
        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            customer_ids: Optional list of customer IDs to filter by
            product_ids: Optional list of product IDs to filter by
            event_types: Optional list of event types to filter by
            limit: Maximum number of rows to return
            
        Returns:
            Pandas DataFrame with online behavior data
        """
        query = f"""
        SELECT * FROM `{self.project_id}.{config.BQ_DATASET_ID}.{config.BQ_ONLINE_BEHAVIOR_TABLE}`
        """
        
        conditions = []
        
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            conditions.append(f"DATE(PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', event_timestamp)) >= '{start_date}'")
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            conditions.append(f"DATE(PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', event_timestamp)) <= '{end_date}'")
        
        if customer_ids:
            customer_ids_str = ", ".join([f"'{cid}'" for cid in customer_ids])
            conditions.append(f"customer_id IN ({customer_ids_str})")
        
        if product_ids:
            product_ids_str = ", ".join([f"'{pid}'" for pid in product_ids])
            conditions.append(f"product_id IN ({product_ids_str})")
        
        if event_types:
            event_types_str = ", ".join([f"'{evt}'" for evt in event_types])
            conditions.append(f"event_type IN ({event_types_str})")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.run_query(query)
    
    def execute_custom_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Execute a custom parameterized query.
        
        Args:
            query: SQL query with optional parameters
            params: Dictionary of parameter values
            
        Returns:
            Pandas DataFrame with query results
        """
        if params:
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(name, "STRING", value)
                    for name, value in params.items()
                ]
            )
            query_job = self.client.query(query, job_config=job_config)
        else:
            query_job = self.client.query(query)
        
        results = query_job.result()
        return results.to_dataframe()
    
    def validate_query(self, query: str) -> bool:
        """
        Validate a SQL query without executing it.
        
        Args:
            query: SQL query string
            
        Returns:
            Boolean indicating if the query is valid
        """
        try:
            # Use the dry run option to validate without executing
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            query_job = self.client.query(query, job_config=job_config)
            return True
        except Exception as e:
            logger.warning(f"Query validation failed: {str(e)}")
            return False
    
    def check_data_exists(self, query: str) -> bool:
        """
        Check if a query would return any results.
        
        Args:
            query: SQL query string
            
        Returns:
            Boolean indicating if the query would return data
        """
        try:
            # Modify the query to just count results
            count_query = f"SELECT COUNT(*) as count FROM ({query}) AS subquery"
            result = self.run_query(count_query)
            return result.iloc[0]['count'] > 0
        except Exception as e:
            logger.warning(f"Data existence check failed: {str(e)}")
            return False
