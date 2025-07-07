"""
Comprehensive test suite for Snowflake Vector Store integration.

This test suite covers:
- Basic vector store operations (add, search, delete)
- Snowflake-specific features (vector types, embedding functions)
- Performance and scalability tests
- Error handling and edge cases
- Integration with productivity application workflows
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class MockSnowflakeEmbeddings(Embeddings):
    """Mock embeddings class for testing."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return mock embeddings for documents."""
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Return mock embedding for query."""
        return [0.1, 0.2, 0.3]


class TestSnowflakeVectorStore:
    """Test suite for Snowflake Vector Store."""
    
    @pytest.fixture
    def mock_snowflake_connection(self):
        """Mock Snowflake connection for testing."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn, mock_cursor
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings instance."""
        return MockSnowflakeEmbeddings()
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(page_content="This is a test document", metadata={"source": "test1"}),
            Document(page_content="Another document for testing", metadata={"source": "test2"}),
            Document(page_content="Vector search is powerful", metadata={"source": "test3"}),
        ]
    
    @pytest.fixture
    def snowflake_vectorstore(self, mock_snowflake_connection, mock_embeddings):
        """Create a mock Snowflake vector store instance."""
        # This would be replaced with actual SnowflakeVectorStore class
        from snowflake_vectorstore import SnowflakeVectorStore
        
        conn, cursor = mock_snowflake_connection
        return SnowflakeVectorStore(
            connection=conn,
            table_name="test_vectors",
            embedding_function=mock_embeddings,
            vector_dimension=768,
            vector_type="FLOAT"
        )


class TestBasicOperations:
    """Test basic vector store operations."""
    
    def test_initialization(self, mock_snowflake_connection, mock_embeddings):
        """Test vector store initialization."""
        conn, cursor = mock_snowflake_connection
        
        # Test successful initialization
        vectorstore = SnowflakeVectorStore(
            connection=conn,
            table_name="test_vectors",
            embedding_function=mock_embeddings,
            vector_dimension=768,
            vector_type="FLOAT"
        )
        
        assert vectorstore.table_name == "test_vectors"
        assert vectorstore.vector_dimension == 768
        assert vectorstore.vector_type == "FLOAT"
        
        # Verify table creation SQL was executed
        cursor.execute.assert_called()
        create_table_sql = cursor.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS test_vectors" in create_table_sql
        assert "VECTOR(FLOAT, 768)" in create_table_sql
    
    def test_add_texts(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test adding texts to vector store."""
        conn, cursor = mock_snowflake_connection
        
        texts = ["Document 1", "Document 2", "Document 3"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
        
        # Mock successful insertion
        cursor.execute.return_value = None
        
        ids = snowflake_vectorstore.add_texts(texts, metadatas)
        
        # Verify correct number of IDs returned
        assert len(ids) == 3
        
        # Verify INSERT statements were executed
        assert cursor.execute.call_count >= 3
        insert_calls = [call for call in cursor.execute.call_args_list 
                       if "INSERT INTO" in str(call)]
        assert len(insert_calls) == 3
    
    def test_add_documents(self, snowflake_vectorstore, sample_documents):
        """Test adding documents to vector store."""
        ids = snowflake_vectorstore.add_documents(sample_documents)
        
        assert len(ids) == 3
        assert all(isinstance(id, str) for id in ids)
    
    def test_similarity_search(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test similarity search functionality."""
        conn, cursor = mock_snowflake_connection
        
        # Mock search results
        mock_results = [
            ("Document 1", {"source": "doc1"}, "id1", 0.95),
            ("Document 2", {"source": "doc2"}, "id2", 0.85),
            ("Document 3", {"source": "doc3"}, "id3", 0.75),
        ]
        cursor.fetchall.return_value = mock_results
        
        results = snowflake_vectorstore.similarity_search("test query", k=3)
        
        # Verify results format
        assert len(results) == 3
        assert all(isinstance(doc, Document) for doc in results)
        assert results[0].page_content == "Document 1"
        assert results[0].metadata == {"source": "doc1"}
        
        # Verify correct SQL was executed
        cursor.execute.assert_called()
        search_sql = cursor.execute.call_args[0][0]
        assert "VECTOR_COSINE_SIMILARITY" in search_sql
        assert "ORDER BY similarity DESC" in search_sql
        assert "LIMIT 3" in search_sql
    
    def test_similarity_search_with_score(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test similarity search with relevance scores."""
        conn, cursor = mock_snowflake_connection
        
        # Mock search results with scores
        mock_results = [
            ("Document 1", {"source": "doc1"}, "id1", 0.95),
            ("Document 2", {"source": "doc2"}, "id2", 0.85),
        ]
        cursor.fetchall.return_value = mock_results
        
        results = snowflake_vectorstore.similarity_search_with_score("test query", k=2)
        
        assert len(results) == 2
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
        
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert score == 0.95
    
    def test_delete_documents(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test document deletion."""
        conn, cursor = mock_snowflake_connection
        
        ids_to_delete = ["id1", "id2", "id3"]
        
        # Mock successful deletion
        cursor.execute.return_value = None
        cursor.rowcount = 3
        
        result = snowflake_vectorstore.delete(ids_to_delete)
        
        # Verify deletion was successful
        assert result is True
        
        # Verify DELETE SQL was executed
        cursor.execute.assert_called()
        delete_sql = cursor.execute.call_args[0][0]
        assert "DELETE FROM" in delete_sql
        assert "WHERE id IN" in delete_sql


class TestSnowflakeSpecificFeatures:
    """Test Snowflake-specific vector store features."""
    
    def test_vector_data_types(self, mock_snowflake_connection, mock_embeddings):
        """Test different vector data types (INT, FLOAT)."""
        conn, cursor = mock_snowflake_connection
        
        # Test FLOAT vectors
        float_store = SnowflakeVectorStore(
            connection=conn,
            table_name="float_vectors",
            embedding_function=mock_embeddings,
            vector_dimension=768,
            vector_type="FLOAT"
        )
        
        # Test INT vectors
        int_store = SnowflakeVectorStore(
            connection=conn,
            table_name="int_vectors", 
            embedding_function=mock_embeddings,
            vector_dimension=256,
            vector_type="INT"
        )
        
        # Verify correct SQL generation for different types
        calls = cursor.execute.call_args_list
        float_sql = str(calls[0])
        int_sql = str(calls[1])
        
        assert "VECTOR(FLOAT, 768)" in float_sql
        assert "VECTOR(INT, 256)" in int_sql
    
    def test_embedding_functions(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test integration with Snowflake embedding functions."""
        conn, cursor = mock_snowflake_connection
        
        # Test using Snowflake's EMBED_TEXT_768 function
        vectorstore = SnowflakeVectorStore(
            connection=conn,
            table_name="embed_vectors",
            use_snowflake_embeddings=True,
            embedding_model="snowflake-arctic-embed-m",
            vector_dimension=768
        )
        
        texts = ["Test document"]
        vectorstore.add_texts(texts)
        
        # Verify EMBED_TEXT_768 was used in SQL
        cursor.execute.assert_called()
        sql = cursor.execute.call_args[0][0]
        assert "SNOWFLAKE.CORTEX.EMBED_TEXT_768" in sql
    
    def test_vector_similarity_functions(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test different vector similarity functions."""
        conn, cursor = mock_snowflake_connection
        
        # Test cosine similarity
        snowflake_vectorstore.similarity_search("query", k=3, similarity_function="cosine")
        cosine_sql = cursor.execute.call_args[0][0]
        assert "VECTOR_COSINE_SIMILARITY" in cosine_sql
        
        # Test L2 distance
        snowflake_vectorstore.similarity_search("query", k=3, similarity_function="l2")
        l2_sql = cursor.execute.call_args[0][0]
        assert "VECTOR_L2_DISTANCE" in l2_sql
        
        # Test inner product
        snowflake_vectorstore.similarity_search("query", k=3, similarity_function="inner_product")
        inner_sql = cursor.execute.call_args[0][0]
        assert "VECTOR_INNER_PRODUCT" in inner_sql
    
    def test_metadata_filtering(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test metadata-based filtering."""
        conn, cursor = mock_snowflake_connection
        
        # Mock filtered results
        mock_results = [("Filtered doc", {"category": "test"}, "id1", 0.9)]
        cursor.fetchall.return_value = mock_results
        
        filter_dict = {"category": "test", "status": "active"}
        results = snowflake_vectorstore.similarity_search(
            "query", k=5, filter=filter_dict
        )
        
        # Verify filtering was applied in SQL
        cursor.execute.assert_called()
        sql = cursor.execute.call_args[0][0]
        assert "WHERE" in sql
        assert "metadata" in sql
    
    def test_vector_compression(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test vector compression and truncation features."""
        conn, cursor = mock_snowflake_connection
        
        # Test with compressed vectors (256 dimensions)
        compressed_store = SnowflakeVectorStore(
            connection=conn,
            table_name="compressed_vectors",
            embedding_function=mock_embeddings,
            vector_dimension=256,
            vector_type="FLOAT",
            compression_enabled=True
        )
        
        texts = ["Test compression"]
        compressed_store.add_texts(texts)
        
        # Verify compression settings in SQL
        cursor.execute.assert_called()
        sql = cursor.execute.call_args[0][0]
        assert "VECTOR(FLOAT, 256)" in sql


class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""
    
    def test_batch_operations(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test efficient batch operations."""
        conn, cursor = mock_snowflake_connection
        
        # Large batch of documents
        large_batch = [f"Document {i}" for i in range(1000)]
        metadatas = [{"batch_id": i // 100} for i in range(1000)]
        
        # Mock successful batch insertion
        cursor.execute.return_value = None
        cursor.executemany.return_value = None
        
        ids = snowflake_vectorstore.add_texts(large_batch, metadatas)
        
        # Verify batch operations were used
        assert len(ids) == 1000
        cursor.executemany.assert_called()
    
    def test_connection_pooling(self, mock_embeddings):
        """Test connection pooling for better performance."""
        with patch('snowflake_vectorstore.ConnectionPool') as mock_pool:
            mock_pool.return_value.get_connection.return_value = Mock()
            
            vectorstore = SnowflakeVectorStore(
                connection_pool=mock_pool,
                table_name="pooled_vectors",
                embedding_function=mock_embeddings,
                use_connection_pool=True
            )
            
            # Verify connection pool was used
            mock_pool.assert_called_once()
    
    def test_parallel_search(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test parallel search capabilities."""
        conn, cursor = mock_snowflake_connection
        
        # Mock multiple parallel queries
        queries = ["query1", "query2", "query3"]
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_executor.return_value.map.return_value = [
                [Document(page_content=f"Result for {q}")] for q in queries
            ]
            
            results = snowflake_vectorstore.similarity_search_batch(queries, k=5)
            
            assert len(results) == 3
            mock_executor.assert_called_once()
    
    def test_memory_efficiency(self, snowflake_vectorstore):
        """Test memory-efficient operations for large datasets."""
        # Test streaming results for large queries
        with patch.object(snowflake_vectorstore, '_stream_results') as mock_stream:
            mock_stream.return_value = iter([
                Document(page_content=f"Doc {i}") for i in range(10000)
            ])
            
            results = list(snowflake_vectorstore.similarity_search_stream("query", k=10000))
            
            assert len(results) == 10000
            mock_stream.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_connection_errors(self, mock_embeddings):
        """Test handling of connection errors."""
        with patch('snowflake.connector.connect') as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception) as exc_info:
                SnowflakeVectorStore(
                    connection_string="invalid://connection",
                    table_name="test_vectors",
                    embedding_function=mock_embeddings
                )
            
            assert "Connection failed" in str(exc_info.value)
    
    def test_invalid_vector_dimensions(self, mock_snowflake_connection, mock_embeddings):
        """Test handling of invalid vector dimensions."""
        conn, cursor = mock_snowflake_connection
        
        with pytest.raises(ValueError) as exc_info:
            SnowflakeVectorStore(
                connection=conn,
                table_name="test_vectors",
                embedding_function=mock_embeddings,
                vector_dimension=5000  # Exceeds maximum of 4096
            )
        
        assert "dimension must be between 1 and 4096" in str(exc_info.value)
    
    def test_empty_documents(self, snowflake_vectorstore):
        """Test handling of empty documents."""
        empty_docs = []
        
        ids = snowflake_vectorstore.add_documents(empty_docs)
        assert ids == []
        
        # Test empty search results
        results = snowflake_vectorstore.similarity_search("query", k=5)
        assert isinstance(results, list)
    
    def test_malformed_metadata(self, snowflake_vectorstore):
        """Test handling of malformed metadata."""
        texts = ["Test document"]
        malformed_metadata = [{"nested": {"too": {"deep": "value"}}}]
        
        with pytest.raises(ValueError) as exc_info:
            snowflake_vectorstore.add_texts(texts, malformed_metadata)
        
        assert "metadata complexity" in str(exc_info.value)
    
    def test_sql_injection_prevention(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test prevention of SQL injection attacks."""
        conn, cursor = mock_snowflake_connection
        
        # Attempt SQL injection in query
        malicious_query = "'; DROP TABLE test_vectors; --"
        
        snowflake_vectorstore.similarity_search(malicious_query, k=5)
        
        # Verify parameterized queries were used
        cursor.execute.assert_called()
        sql_call = cursor.execute.call_args
        assert len(sql_call) > 1  # Parameters were passed
        assert "DROP TABLE" not in sql_call[0][0]


class TestIntegrationWithProductivityApp:
    """Test integration with productivity application workflows."""
    
    def test_document_indexing_workflow(self, snowflake_vectorstore):
        """Test document indexing for productivity app."""
        # Simulate productivity app documents
        documents = [
            Document(
                page_content="Meeting notes from project kickoff",
                metadata={
                    "type": "meeting_notes",
                    "project": "project_alpha",
                    "date": "2024-01-15",
                    "participants": ["alice", "bob", "charlie"]
                }
            ),
            Document(
                page_content="Task assignment for feature development",
                metadata={
                    "type": "task",
                    "project": "project_alpha", 
                    "assignee": "alice",
                    "priority": "high",
                    "due_date": "2024-02-01"
                }
            ),
            Document(
                page_content="Project documentation and requirements",
                metadata={
                    "type": "documentation",
                    "project": "project_alpha",
                    "version": "1.0",
                    "author": "bob"
                }
            )
        ]
        
        ids = snowflake_vectorstore.add_documents(documents)
        assert len(ids) == 3
        
        # Test semantic search for productivity use cases
        meeting_results = snowflake_vectorstore.similarity_search(
            "project kickoff discussion", 
            k=3,
            filter={"type": "meeting_notes"}
        )
        
        task_results = snowflake_vectorstore.similarity_search(
            "high priority assignments",
            k=3,
            filter={"priority": "high"}
        )
        
        assert len(meeting_results) >= 1
        assert len(task_results) >= 1
    
    def test_real_time_search_suggestions(self, snowflake_vectorstore):
        """Test real-time search suggestions for productivity app."""
        # Add sample productivity data
        productivity_docs = [
            Document(page_content="Email automation workflow", metadata={"category": "automation"}),
            Document(page_content="Calendar scheduling optimization", metadata={"category": "scheduling"}),
            Document(page_content="Task prioritization algorithms", metadata={"category": "productivity"}),
            Document(page_content="Team collaboration best practices", metadata={"category": "collaboration"})
        ]
        
        snowflake_vectorstore.add_documents(productivity_docs)
        
        # Test partial query matching (as user types)
        partial_queries = ["automat", "schedul", "collaborat"]
        
        for query in partial_queries:
            suggestions = snowflake_vectorstore.similarity_search(
                query, k=3, min_similarity=0.5
            )
            assert len(suggestions) >= 1
    
    def test_multi_tenant_support(self, mock_snowflake_connection, mock_embeddings):
        """Test multi-tenant support for productivity app."""
        conn, cursor = mock_snowflake_connection
        
        # Create tenant-specific vector stores
        tenant1_store = SnowflakeVectorStore(
            connection=conn,
            table_name="vectors_tenant1",
            embedding_function=mock_embeddings,
            tenant_id="tenant1"
        )
        
        tenant2_store = SnowflakeVectorStore(
            connection=conn,
            table_name="vectors_tenant2", 
            embedding_function=mock_embeddings,
            tenant_id="tenant2"
        )
        
        # Verify tenant isolation
        assert tenant1_store.table_name == "vectors_tenant1"
        assert tenant2_store.table_name == "vectors_tenant2"
        
        # Test cross-tenant search prevention
        tenant1_docs = [Document(page_content="Tenant 1 data", metadata={"tenant": "tenant1"})]
        tenant1_store.add_documents(tenant1_docs)
        
        # Search from tenant2 should not return tenant1 data
        results = tenant2_store.similarity_search("Tenant 1 data", k=5)
        assert len(results) == 0


class TestAdvancedFeatures:
    """Test advanced vector store features."""
    
    def test_hybrid_search(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test hybrid search combining vector and text search."""
        conn, cursor = mock_snowflake_connection
        
        # Mock hybrid search results
        mock_results = [
            ("Hybrid result 1", {"score": 0.9}, "id1", 0.9),
            ("Hybrid result 2", {"score": 0.8}, "id2", 0.8),
        ]
        cursor.fetchall.return_value = mock_results
        
        results = snowflake_vectorstore.hybrid_search(
            query="test query",
            text_search_weight=0.3,
            vector_search_weight=0.7,
            k=5
        )
        
        assert len(results) == 2
        
        # Verify hybrid search SQL was generated
        cursor.execute.assert_called()
        sql = cursor.execute.call_args[0][0]
        assert "VECTOR_COSINE_SIMILARITY" in sql
        assert "MATCH" in sql or "CONTAINS" in sql  # Text search component
    
    def test_reranking_support(self, snowflake_vectorstore):
        """Test result reranking for improved relevance."""
        # Mock initial results
        initial_results = [
            Document(page_content="Document 1", metadata={"score": 0.7}),
            Document(page_content="Document 2", metadata={"score": 0.8}),
            Document(page_content="Document 3", metadata={"score": 0.6}),
        ]
        
        with patch.object(snowflake_vectorstore, 'similarity_search') as mock_search:
            mock_search.return_value = initial_results
            
            # Test reranking with custom function
            reranked_results = snowflake_vectorstore.similarity_search_with_reranking(
                "query", k=3, rerank_function=lambda docs: sorted(docs, key=lambda d: d.metadata['score'], reverse=True)
            )
            
            assert len(reranked_results) == 3
            assert reranked_results[0].metadata['score'] == 0.8  # Highest score first
    
    def test_vector_clustering(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test vector clustering for improved organization."""
        conn, cursor = mock_snowflake_connection
        
        # Mock clustering results
        mock_clusters = [
            {"cluster_id": 1, "centroid": [0.1, 0.2, 0.3], "documents": ["id1", "id2"]},
            {"cluster_id": 2, "centroid": [0.4, 0.5, 0.6], "documents": ["id3", "id4"]},
        ]
        
        with patch.object(snowflake_vectorstore, '_perform_clustering') as mock_clustering:
            mock_clustering.return_value = mock_clusters
            
            clusters = snowflake_vectorstore.create_clusters(num_clusters=2)
            
            assert len(clusters) == 2
            assert all('cluster_id' in cluster for cluster in clusters)
    
    def test_incremental_updates(self, snowflake_vectorstore, mock_snowflake_connection):
        """Test incremental updates for changing documents."""
        conn, cursor = mock_snowflake_connection
        
        # Mock successful update
        cursor.execute.return_value = None
        cursor.rowcount = 1
        
        updated_doc = Document(
            page_content="Updated document content",
            metadata={"version": 2, "last_updated": "2024-01-20"}
        )
        
        success = snowflake_vectorstore.update_document("existing_id", updated_doc)
        
        assert success is True
        
        # Verify UPDATE SQL was executed
        cursor.execute.assert_called()
        sql = cursor.execute.call_args[0][0]
        assert "UPDATE" in sql
        assert "SET content = " in sql
        assert "SET vector = " in sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])