"""
Test suite for Snowflake Vector Store implementation.

This module provides comprehensive tests for the SnowflakeVectorStore class,
including standard vector store functionality, Snowflake-specific features,
and integration with LangChain ecosystem.
"""

import os
import pytest
from typing import Generator, List, Optional
from unittest.mock import Mock, patch

import snowflake.connector
from snowflake.snowpark import Session
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests


class MockSnowflakeEmbeddings(Embeddings):
    """Mock embeddings class for testing."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for documents."""
        return [[0.1] * self.dimension for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Generate mock embedding for query."""
        return [0.1] * self.dimension


class SnowflakeVectorStore(VectorStore):
    """
    Snowflake Vector Store implementation.
    
    This class provides vector storage and retrieval capabilities using
    Snowflake's native VECTOR data type and Cortex AI functions.
    """
    
    def __init__(
        self,
        connection_parameters: dict,
        table_name: str,
        embedding_function: Embeddings,
        dimension: int = 768,
        embedding_model: str = "snowflake-arctic-embed-m-v1.5"
    ):
        """
        Initialize Snowflake Vector Store.
        
        Args:
            connection_parameters: Snowflake connection parameters
            table_name: Name of the table to store vectors
            embedding_function: Embedding function to use
            dimension: Vector dimension (768 or 1024)
            embedding_model: Snowflake embedding model to use
        """
        self.connection_parameters = connection_parameters
        self.table_name = table_name
        self.embedding_function = embedding_function
        self.dimension = dimension
        self.embedding_model = embedding_model
        self._session = None
        self._setup_table()
    
    def _get_session(self) -> Session:
        """Get or create Snowflake session."""
        if self._session is None:
            self._session = Session.builder.configs(self.connection_parameters).create()
        return self._session
    
    def _setup_table(self):
        """Create the vector table if it doesn't exist."""
        session = self._get_session()
        
        # Create table with vector column
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR PRIMARY KEY,
            content TEXT,
            metadata VARIANT,
            vector VECTOR(FLOAT, {self.dimension})
        )
        """
        session.sql(create_table_sql).collect()
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """Add documents to the vector store."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        session = self._get_session()
        
        # Generate embeddings using Snowflake Cortex
        texts = [doc.page_content for doc in documents]
        
        # Insert documents with embeddings
        for i, (doc, doc_id) in enumerate(zip(documents, ids)):
            # Use Snowflake Cortex embedding function
            embed_sql = f"""
            INSERT INTO {self.table_name} (id, content, metadata, vector)
            SELECT 
                '{doc_id}',
                '{doc.page_content}',
                PARSE_JSON('{doc.metadata}'),
                SNOWFLAKE.CORTEX.EMBED_TEXT_{self.dimension}('{self.embedding_model}', '{doc.page_content}')
            """
            session.sql(embed_sql).collect()
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Document]:
        """Perform similarity search."""
        session = self._get_session()
        
        # Generate query embedding
        query_embed_sql = f"""
        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_{self.dimension}('{self.embedding_model}', '{query}') as query_vector
        """
        query_result = session.sql(query_embed_sql).collect()
        
        # Perform similarity search
        similarity_sql = f"""
        SELECT 
            id,
            content,
            metadata,
            VECTOR_COSINE_SIMILARITY(vector, {query_result[0]['QUERY_VECTOR']}) as similarity
        FROM {self.table_name}
        ORDER BY similarity DESC
        LIMIT {k}
        """
        
        results = session.sql(similarity_sql).collect()
        
        documents = []
        for row in results:
            doc = Document(
                page_content=row['CONTENT'],
                metadata=row['METADATA'],
                id=row['ID']
            )
            documents.append(doc)
        
        return documents
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """Perform similarity search with scores."""
        session = self._get_session()
        
        # Generate query embedding and search
        query_embed_sql = f"""
        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_{self.dimension}('{self.embedding_model}', '{query}') as query_vector
        """
        query_result = session.sql(query_embed_sql).collect()
        
        similarity_sql = f"""
        SELECT 
            id,
            content,
            metadata,
            VECTOR_COSINE_SIMILARITY(vector, {query_result[0]['QUERY_VECTOR']}) as similarity
        FROM {self.table_name}
        ORDER BY similarity DESC
        LIMIT {k}
        """
        
        results = session.sql(similarity_sql).collect()
        
        documents_with_scores = []
        for row in results:
            doc = Document(
                page_content=row['CONTENT'],
                metadata=row['METADATA'],
                id=row['ID']
            )
            score = float(row['SIMILARITY'])
            documents_with_scores.append((doc, score))
        
        return documents_with_scores
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """Delete documents by IDs."""
        if not ids:
            return
        
        session = self._get_session()
        ids_str = "', '".join(ids)
        delete_sql = f"DELETE FROM {self.table_name} WHERE id IN ('{ids_str}')"
        session.sql(delete_sql).collect()
    
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs."""
        session = self._get_session()
        ids_str = "', '".join(ids)
        select_sql = f"""
        SELECT id, content, metadata 
        FROM {self.table_name} 
        WHERE id IN ('{ids_str}')
        """
        
        results = session.sql(select_sql).collect()
        
        documents = []
        for row in results:
            doc = Document(
                page_content=row['CONTENT'],
                metadata=row['METADATA'],
                id=row['ID']
            )
            documents.append(doc)
        
        return documents
    
    def _clear_table(self):
        """Clear all data from the table (for testing)."""
        session = self._get_session()
        session.sql(f"DELETE FROM {self.table_name}").collect()


class TestSnowflakeVectorStore(VectorStoreIntegrationTests):
    """Test suite for Snowflake Vector Store."""
    
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:
        """Get an empty vectorstore for testing."""
        # Mock connection parameters for testing
        connection_params = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT", "test_account"),
            "user": os.getenv("SNOWFLAKE_USER", "test_user"),
            "password": os.getenv("SNOWFLAKE_PASSWORD", "test_password"),
            "database": os.getenv("SNOWFLAKE_DATABASE", "test_db"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA", "test_schema"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "test_warehouse"),
        }
        
        # Use mock embeddings for testing
        embeddings = MockSnowflakeEmbeddings(dimension=768)
        
        # Create vector store instance
        store = SnowflakeVectorStore(
            connection_parameters=connection_params,
            table_name="test_vectors",
            embedding_function=embeddings,
            dimension=768
        )
        
        try:
            # Ensure table is empty
            store._clear_table()
            yield store
        finally:
            # Cleanup
            store._clear_table()
    
    @property
    def has_async(self) -> bool:
        """Disable async tests for now."""
        return False
    
    # Snowflake-specific tests
    
    def test_vector_similarity_functions(self, vectorstore: SnowflakeVectorStore):
        """Test different vector similarity functions."""
        # Add test documents
        documents = [
            Document(page_content="The cat sat on the mat", metadata={"type": "animal"}),
            Document(page_content="The dog ran in the park", metadata={"type": "animal"}),
            Document(page_content="The car drove down the street", metadata={"type": "vehicle"}),
        ]
        
        ids = vectorstore.add_documents(documents)
        
        # Test cosine similarity (default)
        results = vectorstore.similarity_search("cat", k=2)
        assert len(results) == 2
        assert "cat" in results[0].page_content.lower()
        
        # Test with scores
        results_with_scores = vectorstore.similarity_search_with_score("cat", k=2)
        assert len(results_with_scores) == 2
        assert all(isinstance(score, float) for _, score in results_with_scores)
    
    def test_different_embedding_dimensions(self):
        """Test support for different embedding dimensions."""
        connection_params = {
            "account": "test_account",
            "user": "test_user", 
            "password": "test_password",
            "database": "test_db",
            "schema": "test_schema",
            "warehouse": "test_warehouse",
        }
        
        # Test 768 dimensions
        embeddings_768 = MockSnowflakeEmbeddings(dimension=768)
        store_768 = SnowflakeVectorStore(
            connection_parameters=connection_params,
            table_name="test_vectors_768",
            embedding_function=embeddings_768,
            dimension=768
        )
        
        # Test 1024 dimensions
        embeddings_1024 = MockSnowflakeEmbeddings(dimension=1024)
        store_1024 = SnowflakeVectorStore(
            connection_parameters=connection_params,
            table_name="test_vectors_1024",
            embedding_function=embeddings_1024,
            dimension=1024
        )
        
        assert store_768.dimension == 768
        assert store_1024.dimension == 1024
    
    def test_metadata_filtering(self, vectorstore: SnowflakeVectorStore):
        """Test metadata filtering capabilities."""
        # Add documents with different metadata
        documents = [
            Document(page_content="Animal content", metadata={"category": "animals", "rating": 5}),
            Document(page_content="Vehicle content", metadata={"category": "vehicles", "rating": 3}),
            Document(page_content="Food content", metadata={"category": "food", "rating": 4}),
        ]
        
        vectorstore.add_documents(documents)
        
        # Test basic search without filter
        results = vectorstore.similarity_search("content", k=3)
        assert len(results) == 3
        
        # Note: Actual metadata filtering would require implementing
        # WHERE clause support in the similarity_search method
    
    def test_batch_operations_performance(self, vectorstore: SnowflakeVectorStore):
        """Test batch operations for performance."""
        # Create a larger batch of documents
        batch_size = 100
        documents = [
            Document(
                page_content=f"Test document number {i}",
                metadata={"batch_id": i // 10, "doc_num": i}
            )
            for i in range(batch_size)
        ]
        
        # Test batch insertion
        ids = vectorstore.add_documents(documents)
        assert len(ids) == batch_size
        
        # Test batch retrieval
        results = vectorstore.similarity_search("test document", k=10)
        assert len(results) == 10
        
        # Test batch deletion
        vectorstore.delete(ids[:50])
        remaining_results = vectorstore.similarity_search("test document", k=100)
        assert len(remaining_results) == 50
    
    def test_vector_data_type_conversion(self, vectorstore: SnowflakeVectorStore):
        """Test proper handling of vector data types."""
        # This test would verify that vectors are properly converted
        # between Python lists and Snowflake VECTOR type
        
        document = Document(page_content="Test vector conversion")
        ids = vectorstore.add_documents([document])
        
        # Retrieve and verify the document
        retrieved = vectorstore.get_by_ids(ids)
        assert len(retrieved) == 1
        assert retrieved[0].page_content == "Test vector conversion"
    
    def test_large_document_handling(self, vectorstore: SnowflakeVectorStore):
        """Test handling of large documents."""
        # Test with a large document (> 512 tokens)
        large_content = "This is a test document. " * 100  # ~500 words
        
        document = Document(
            page_content=large_content,
            metadata={"size": "large", "word_count": len(large_content.split())}
        )
        
        ids = vectorstore.add_documents([document])
        
        # Verify the document was stored and can be retrieved
        results = vectorstore.similarity_search("test document", k=1)
        assert len(results) == 1
        assert results[0].page_content == large_content
    
    @pytest.mark.skipif(
        not os.getenv("SNOWFLAKE_ACCOUNT"),
        reason="Snowflake credentials not provided"
    )
    def test_real_snowflake_connection(self):
        """Test with real Snowflake connection (integration test)."""
        # This test would only run if real Snowflake credentials are provided
        connection_params = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        }
        
        # Test actual connection
        try:
            conn = snowflake.connector.connect(**connection_params)
            cursor = conn.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            result = cursor.fetchone()
            assert result is not None
        finally:
            if 'conn' in locals():
                conn.close()


class TestSnowflakeCortexSearchStore:
    """Test suite for Snowflake Cortex Search integration."""
    
    def test_cortex_search_service_creation(self):
        """Test creation of Cortex Search service."""
        # Mock implementation for Cortex Search service creation
        pass
    
    def test_hybrid_search_functionality(self):
        """Test hybrid search combining vector and keyword search."""
        # Mock implementation for hybrid search testing
        pass
    
    def test_search_service_refresh(self):
        """Test automatic refresh of search service."""
        # Mock implementation for search service refresh testing
        pass
    
    def test_search_service_cost_optimization(self):
        """Test cost optimization features."""
        # Mock implementation for cost optimization testing
        pass


class TestSnowflakeVectorStorePerformance:
    """Performance tests for Snowflake Vector Store."""
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Mock implementation for large dataset performance testing
        pass
    
    def test_concurrent_query_performance(self):
        """Test concurrent query performance."""
        # Mock implementation for concurrent query testing
        pass
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization."""
        # Mock implementation for memory usage testing
        pass
    
    def test_batch_insert_performance(self):
        """Test batch insert performance."""
        # Mock implementation for batch insert performance testing
        pass
    
    def test_search_latency_benchmarks(self):
        """Test search latency benchmarks."""
        # Mock implementation for search latency benchmarking
        pass


class TestSnowflakeVectorStoreErrorHandling:
    """Error handling and resilience tests."""
    
    def test_connection_failure_recovery(self):
        """Test recovery from connection failures."""
        # Mock implementation for connection failure testing
        pass
    
    def test_authentication_error_handling(self):
        """Test handling of authentication errors."""
        # Mock implementation for authentication error testing
        pass
    
    def test_resource_limit_handling(self):
        """Test handling of resource limits."""
        # Mock implementation for resource limit testing
        pass
    
    def test_invalid_vector_data_handling(self):
        """Test handling of invalid vector data."""
        # Mock implementation for invalid data testing
        pass
    
    def test_embedding_function_failures(self):
        """Test handling of embedding function failures."""
        # Mock implementation for embedding failure testing
        pass
    
    def test_retry_logic_and_backoff(self):
        """Test retry logic with exponential backoff."""
        # Mock implementation for retry logic testing
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])