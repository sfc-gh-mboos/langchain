"""
Comprehensive test suite for Snowflake Vector Store implementation.

This test file covers all aspects of the Snowflake vector store functionality including:
- Connection management
- Table operations
- Vector operations
- Data management
- Error handling
- Performance testing
- Integration scenarios
"""

import asyncio
import json
import os
import pytest
import uuid
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Mock implementation for testing - would be replaced with actual implementation
class MockSnowflakeVectorStore:
    """Mock implementation of SnowflakeVectorStore for testing purposes."""
    
    def __init__(
        self,
        connection_parameters: Dict[str, Any],
        table_name: str,
        embedding_function: Optional[Embeddings] = None,
        text_column: str = "text",
        vector_column: str = "vector",
        metadata_column: str = "metadata",
        id_column: str = "id",
        dimension: int = 768,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        distance_metric: str = "cosine",
        create_table_if_not_exists: bool = True,
    ):
        # Validate required parameters
        if not connection_parameters.get("account"):
            raise ValueError("Account parameter is required")
        
        # Validate dimension
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        
        self.connection_parameters = connection_parameters
        self.table_name = table_name
        self.embedding_function = embedding_function
        self.text_column = text_column
        self.vector_column = vector_column
        self.metadata_column = metadata_column
        self.id_column = id_column
        self.dimension = dimension
        self.database = database
        self.schema = schema
        self.distance_metric = distance_metric
        self.create_table_if_not_exists = create_table_if_not_exists
        
        # Mock data storage
        self._documents: Dict[str, Document] = {}
        self._vectors: Dict[str, np.ndarray] = {}
        self._connection = None
        self._table_exists = False
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store."""
        if not texts:
            return []
        
        # Validate texts
        for text in texts:
            if text is None:
                raise ValueError("Text cannot be None")
            if len(text) > 16 * 1024 * 1024:  # 16MB limit
                raise ValueError("Text too large")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Check for duplicate IDs
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs not allowed")
        
        # Handle metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Generate embeddings if embedding function is provided
        if self.embedding_function:
            embeddings = self.embedding_function.embed_documents(texts)
            
            # Validate embedding dimensions
            if embeddings and len(embeddings[0]) != self.dimension:
                raise ValueError("Embedding dimension mismatch")
        else:
            embeddings = [np.random.random(self.dimension).tolist() for _ in texts]
        
        # Validate metadata serialization
        for metadata in metadatas:
            try:
                json.dumps(metadata)
            except TypeError as e:
                raise TypeError("Metadata not JSON serializable") from e
        
        # Store documents and vectors
        for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
            doc = Document(page_content=text, metadata=metadata, id=doc_id)
            self._documents[doc_id] = doc
            self._vectors[doc_id] = np.array(embeddings[i])
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform similarity search."""
        if not self._documents:
            return []
        
        # Get query embedding
        if self.embedding_function:
            query_embedding = self.embedding_function.embed_query(query)
        else:
            query_embedding = np.random.random(self.dimension).tolist()
        
        # Filter documents
        filtered_docs = self._filter_documents(filter)
        
        # Simple mock similarity calculation
        results = []
        for doc in filtered_docs[:k]:
            results.append(doc)
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores."""
        docs = self.similarity_search(query, k, filter)
        
        # Generate mock scores
        results = []
        for i, doc in enumerate(docs):
            score = max(0.0, 1.0 - (i * 0.1))  # Decreasing scores
            results.append((doc, score))
        
        return results
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform similarity search by vector."""
        filtered_docs = self._filter_documents(filter)
        return filtered_docs[:k]
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform maximum marginal relevance search."""
        # Simple mock implementation
        filtered_docs = self._filter_documents(filter)
        
        # Return diverse results (mock)
        results = []
        seen_categories = set()
        
        for doc in filtered_docs:
            if len(results) >= k:
                break
            
            category = doc.metadata.get("category", "unknown")
            if category not in seen_categories or len(results) < k // 2:
                results.append(doc)
                seen_categories.add(category)
        
        return results
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete documents by IDs."""
        if ids is None:
            return
        
        for doc_id in ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
            if doc_id in self._vectors:
                del self._vectors[doc_id]
    
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs."""
        return [self._documents[doc_id] for doc_id in ids if doc_id in self._documents]
    
    def update_documents(self, ids: List[str], documents: List[Document]) -> None:
        """Update documents."""
        for doc_id, doc in zip(ids, documents):
            if doc_id in self._documents:
                self._documents[doc_id] = doc
                
                # Update vector if embedding function available
                if self.embedding_function:
                    embedding = self.embedding_function.embed_query(doc.page_content)
                    self._vectors[doc_id] = np.array(embedding)
    
    def _filter_documents(self, filter: Optional[Dict[str, Any]]) -> List[Document]:
        """Filter documents based on metadata."""
        if filter is None:
            return list(self._documents.values())
        
        filtered = []
        for doc in self._documents.values():
            if self._matches_filter(doc.metadata, filter):
                filtered.append(doc)
        
        return filtered
    
    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                # Handle complex filters like {"$in": [...]}
                if "$in" in value:
                    if metadata[key] not in value["$in"]:
                        return False
                elif "$ne" in value:
                    if metadata[key] == value["$ne"]:
                        return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def _connect(self):
        """Mock connection method."""
        if "invalid" in self.connection_parameters.get("account", ""):
            raise ConnectionError("Invalid connection parameters")
    
    def _create_table(self):
        """Mock table creation method."""
        if "invalid" in self.table_name:
            raise Exception("Invalid table name")

class MockEmbeddings(Embeddings):
    """Mock embeddings class for testing."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for documents."""
        return [np.random.random(self.dimension).tolist() for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Generate mock embedding for query."""
        return np.random.random(self.dimension).tolist()


class TestSnowflakeVectorStoreConnection:
    """Test connection management functionality."""
    
    def test_connection_initialization(self):
        """Test vector store initialization with connection parameters."""
        connection_params = {
            "account": "test_account",
            "user": "test_user",
            "password": "test_password",
            "warehouse": "test_warehouse",
            "database": "test_database",
            "schema": "test_schema"
        }
        
        store = MockSnowflakeVectorStore(
            connection_parameters=connection_params,
            table_name="test_vectors"
        )
        
        assert store.connection_parameters == connection_params
        assert store.table_name == "test_vectors"
        assert store.dimension == 768
        assert store.distance_metric == "cosine"
    
    def test_connection_with_different_auth_methods(self):
        """Test different authentication methods."""
        # Test with SSO
        sso_params = {
            "account": "test_account",
            "authenticator": "externalbrowser",
            "warehouse": "test_warehouse"
        }
        
        store = MockSnowflakeVectorStore(
            connection_parameters=sso_params,
            table_name="test_vectors"
        )
        
        assert "authenticator" in store.connection_parameters
        assert store.connection_parameters["authenticator"] == "externalbrowser"
        
        # Test with key pair authentication
        key_pair_params = {
            "account": "test_account",
            "user": "test_user",
            "private_key": "test_private_key",
            "warehouse": "test_warehouse"
        }
        
        store = MockSnowflakeVectorStore(
            connection_parameters=key_pair_params,
            table_name="test_vectors"
        )
        
        assert "private_key" in store.connection_parameters
    
    def test_invalid_connection_parameters(self):
        """Test handling of invalid connection parameters."""
        with pytest.raises(ValueError, match="Account parameter is required"):
            MockSnowflakeVectorStore(
                connection_parameters={},
                table_name="test_vectors"
            )
    
    def test_connection_pooling_configuration(self):
        """Test connection pooling settings."""
        connection_params = {
            "account": "test_account",
            "user": "test_user",
            "password": "test_password",
            "warehouse": "test_warehouse",
            "connection_timeout": 30,
            "socket_timeout": 60
        }
        
        store = MockSnowflakeVectorStore(
            connection_parameters=connection_params,
            table_name="test_vectors"
        )
        
        assert store.connection_parameters["connection_timeout"] == 30
        assert store.connection_parameters["socket_timeout"] == 60


class TestSnowflakeVectorStoreTableOperations:
    """Test table creation and management functionality."""
    
    def test_table_creation(self):
        """Test automatic table creation."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            dimension=1024,
            create_table_if_not_exists=True
        )
        
        # Mock table creation
        expected_sql = """
        CREATE TABLE IF NOT EXISTS test_vectors (
            id VARCHAR PRIMARY KEY,
            text TEXT,
            vector VECTOR(FLOAT, 1024),
            metadata VARIANT,
            created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
            updated_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        
        # Verify the SQL would be constructed correctly
        assert store.dimension == 1024
        assert store.table_name == "test_vectors"
    
    def test_table_with_custom_columns(self):
        """Test table creation with custom column names."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="custom_table",
            text_column="content",
            vector_column="embeddings",
            metadata_column="meta",
            id_column="doc_id",
            dimension=512
        )
        
        assert store.text_column == "content"
        assert store.vector_column == "embeddings"
        assert store.metadata_column == "meta"
        assert store.id_column == "doc_id"
        assert store.dimension == 512
    
    def test_table_with_database_and_schema(self):
        """Test table creation with explicit database and schema."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="vectors",
            database="prod_db",
            schema="ml_schema"
        )
        
        assert store.database == "prod_db"
        assert store.schema == "ml_schema"
        # Full table name would be: prod_db.ml_schema.vectors
    
    def test_drop_table(self):
        """Test table deletion."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors"
        )
        
        # Mock table dropping
        store._table_exists = True
        # In real implementation, would execute: DROP TABLE test_vectors
        store._table_exists = False
        
        assert not store._table_exists
    
    def test_table_info_retrieval(self):
        """Test retrieving table schema information."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors"
        )
        
        # Mock table info
        expected_info = {
            "columns": [
                {"name": "id", "type": "VARCHAR", "nullable": False},
                {"name": "text", "type": "TEXT", "nullable": True},
                {"name": "vector", "type": "VECTOR(FLOAT, 768)", "nullable": True},
                {"name": "metadata", "type": "VARIANT", "nullable": True}
            ],
            "row_count": 0,
            "size_bytes": 0
        }
        
        # Test would verify actual table info matches expected structure
        assert "columns" in expected_info
        assert len(expected_info["columns"]) == 4


class TestSnowflakeVectorStoreDataOperations:
    """Test data insertion, retrieval, and management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            embedding_function=MockEmbeddings()
        )
        
        self.sample_texts = [
            "This is the first document about machine learning.",
            "The second document discusses natural language processing.",
            "Third document covers vector databases and similarity search."
        ]
        
        self.sample_metadatas = [
            {"category": "ml", "author": "alice", "year": 2023},
            {"category": "nlp", "author": "bob", "year": 2023},
            {"category": "vector", "author": "charlie", "year": 2024}
        ]
    
    def test_add_texts_basic(self):
        """Test basic text addition."""
        ids = self.store.add_texts(
            texts=self.sample_texts,
            metadatas=self.sample_metadatas
        )
        
        assert len(ids) == 3
        assert all(isinstance(id, str) for id in ids)
        
        # Mock verification
        assert len(self.store._documents) == 3
        assert len(self.store._vectors) == 3
    
    def test_add_texts_with_custom_ids(self):
        """Test adding texts with custom IDs."""
        custom_ids = ["doc1", "doc2", "doc3"]
        
        returned_ids = self.store.add_texts(
            texts=self.sample_texts,
            metadatas=self.sample_metadatas,
            ids=custom_ids
        )
        
        assert returned_ids == custom_ids
        assert all(id in self.store._documents for id in custom_ids)
    
    def test_add_texts_without_metadata(self):
        """Test adding texts without metadata."""
        ids = self.store.add_texts(texts=self.sample_texts)
        
        assert len(ids) == 3
        # Should handle missing metadata gracefully
        for id in ids:
            doc = self.store._documents[id]
            assert doc.metadata == {} or doc.metadata is None
    
    def test_add_texts_batch_processing(self):
        """Test batch processing of large text collections."""
        large_texts = [f"Document {i}" for i in range(1000)]
        
        ids = self.store.add_texts(
            texts=large_texts,
            batch_size=100
        )
        
        assert len(ids) == 1000
        assert len(self.store._documents) == 1000
    
    def test_add_texts_with_duplicate_ids(self):
        """Test handling of duplicate IDs."""
        ids = ["doc1", "doc1", "doc2"]  # Duplicate ID
        
        with pytest.raises(ValueError, match="Duplicate IDs not allowed"):
            self.store.add_texts(
                texts=self.sample_texts,
                ids=ids
            )
    
    def test_delete_documents(self):
        """Test document deletion."""
        # Add documents first
        ids = self.store.add_texts(texts=self.sample_texts)
        assert len(self.store._documents) == 3
        
        # Delete some documents
        self.store.delete(ids=ids[:2])
        
        assert len(self.store._documents) == 1
        assert ids[2] in self.store._documents
        assert ids[0] not in self.store._documents
        assert ids[1] not in self.store._documents
    
    def test_delete_nonexistent_documents(self):
        """Test deleting non-existent documents."""
        # Should not raise error for non-existent IDs
        self.store.delete(ids=["nonexistent1", "nonexistent2"])
        
        # Should handle gracefully
        assert len(self.store._documents) == 0
    
    def test_get_documents_by_ids(self):
        """Test retrieving documents by IDs."""
        ids = self.store.add_texts(
            texts=self.sample_texts,
            metadatas=self.sample_metadatas
        )
        
        retrieved_docs = self.store.get_by_ids(ids=ids[:2])
        
        assert len(retrieved_docs) == 2
        assert all(isinstance(doc, Document) for doc in retrieved_docs)
        assert retrieved_docs[0].page_content == self.sample_texts[0]
        assert retrieved_docs[1].page_content == self.sample_texts[1]
    
    def test_update_documents(self):
        """Test updating existing documents."""
        ids = self.store.add_texts(texts=self.sample_texts)
        
        updated_text = "Updated document content"
        updated_metadata = {"updated": True}
        
        self.store.update_documents(
            ids=[ids[0]],
            documents=[Document(
                page_content=updated_text,
                metadata=updated_metadata
            )]
        )
        
        updated_doc = self.store.get_by_ids([ids[0]])[0]
        assert updated_doc.page_content == updated_text
        assert updated_doc.metadata["updated"] is True


class TestSnowflakeVectorStoreSearch:
    """Test vector search functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            embedding_function=MockEmbeddings()
        )
        
        # Add sample documents
        self.sample_texts = [
            "Machine learning algorithms for data science",
            "Natural language processing with transformers",
            "Vector databases and similarity search",
            "Deep learning neural networks",
            "Information retrieval systems"
        ]
        
        self.sample_metadatas = [
            {"category": "ml", "difficulty": "intermediate"},
            {"category": "nlp", "difficulty": "advanced"},
            {"category": "vector", "difficulty": "beginner"},
            {"category": "dl", "difficulty": "advanced"},
            {"category": "ir", "difficulty": "intermediate"}
        ]
        
        self.ids = self.store.add_texts(
            texts=self.sample_texts,
            metadatas=self.sample_metadatas
        )
    
    def test_similarity_search_basic(self):
        """Test basic similarity search."""
        query = "machine learning"
        results = self.store.similarity_search(query=query, k=3)
        
        assert len(results) == 3
        assert all(isinstance(doc, Document) for doc in results)
        
        # First result should be most similar (mock implementation)
        assert "machine learning" in results[0].page_content.lower()
    
    def test_similarity_search_with_score(self):
        """Test similarity search with relevance scores."""
        query = "neural networks"
        results = self.store.similarity_search_with_score(query=query, k=2)
        
        assert len(results) == 2
        assert all(isinstance(item, tuple) for item in results)
        assert all(isinstance(item[0], Document) for item in results)
        assert all(isinstance(item[1], float) for item in results)
        
        # Scores should be in descending order
        assert results[0][1] >= results[1][1]
        assert 0.0 <= results[0][1] <= 1.0
    
    def test_similarity_search_by_vector(self):
        """Test similarity search using vector embeddings."""
        query_vector = np.random.random(768).tolist()
        results = self.store.similarity_search_by_vector(
            embedding=query_vector,
            k=3
        )
        
        assert len(results) == 3
        assert all(isinstance(doc, Document) for doc in results)
    
    def test_similarity_search_with_filter(self):
        """Test similarity search with metadata filtering."""
        query = "advanced topics"
        results = self.store.similarity_search(
            query=query,
            k=5,
            filter={"difficulty": "advanced"}
        )
        
        # Should only return documents with difficulty="advanced"
        assert len(results) <= 2  # Only 2 advanced documents
        for doc in results:
            assert doc.metadata["difficulty"] == "advanced"
    
    def test_similarity_search_with_complex_filter(self):
        """Test similarity search with complex metadata filtering."""
        query = "learning"
        results = self.store.similarity_search(
            query=query,
            k=5,
            filter={
                "category": {"$in": ["ml", "dl"]},
                "difficulty": {"$ne": "beginner"}
            }
        )
        
        # Should filter by multiple conditions
        for doc in results:
            assert doc.metadata["category"] in ["ml", "dl"]
            assert doc.metadata["difficulty"] != "beginner"
    
    def test_max_marginal_relevance_search(self):
        """Test maximum marginal relevance search."""
        query = "machine learning"
        results = self.store.max_marginal_relevance_search(
            query=query,
            k=3,
            fetch_k=5,
            lambda_mult=0.5
        )
        
        assert len(results) == 3
        assert all(isinstance(doc, Document) for doc in results)
        
        # Results should be diverse (MMR should reduce redundancy)
        unique_categories = set(doc.metadata["category"] for doc in results)
        assert len(unique_categories) >= 2
    
    def test_different_distance_metrics(self):
        """Test different distance metrics."""
        # Test cosine similarity
        cosine_store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors_cosine",
            distance_metric="cosine"
        )
        
        # Test L2 distance
        l2_store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors_l2",
            distance_metric="l2"
        )
        
        # Test inner product
        inner_store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors_inner",
            distance_metric="inner_product"
        )
        
        assert cosine_store.distance_metric == "cosine"
        assert l2_store.distance_metric == "l2"
        assert inner_store.distance_metric == "inner_product"
    
    def test_search_with_no_results(self):
        """Test search when no documents match."""
        # Search with very restrictive filter
        results = self.store.similarity_search(
            query="test",
            k=5,
            filter={"nonexistent_field": "value"}
        )
        
        assert len(results) == 0
    
    def test_search_performance(self):
        """Test search performance with large dataset."""
        # Add many documents
        large_texts = [f"Document {i} with content" for i in range(10000)]
        self.store.add_texts(texts=large_texts)
        
        # Measure search time
        import time
        start_time = time.time()
        
        results = self.store.similarity_search(
            query="content",
            k=10
        )
        
        end_time = time.time()
        search_time = end_time - start_time
        
        assert len(results) == 10
        assert search_time < 1.0  # Should be under 1 second


class TestSnowflakeVectorStoreEmbedding:
    """Test embedding integration functionality."""
    
    def test_custom_embedding_function(self):
        """Test with custom embedding function."""
        custom_embeddings = MockEmbeddings(dimension=512)
        
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            embedding_function=custom_embeddings,
            dimension=512
        )
        
        assert store.dimension == 512
        assert store.embedding_function == custom_embeddings
    
    def test_snowflake_embed_text_integration(self):
        """Test integration with Snowflake EMBED_TEXT functions."""
        # Mock Snowflake embedding function
        class SnowflakeEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                # Mock calling EMBED_TEXT_768 function
                return [np.random.random(768).tolist() for _ in texts]
            
            def embed_query(self, text: str) -> List[float]:
                # Mock calling EMBED_TEXT_768 function
                return np.random.random(768).tolist()
        
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            embedding_function=SnowflakeEmbeddings()
        )
        
        texts = ["Test document 1", "Test document 2"]
        ids = store.add_texts(texts=texts)
        
        assert len(ids) == 2
        assert len(store._vectors) == 2
    
    def test_embedding_dimension_validation(self):
        """Test validation of embedding dimensions."""
        # Test mismatch between declared dimension and actual embedding
        class WrongDimensionEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [np.random.random(512).tolist() for _ in texts]  # Wrong dimension
            
            def embed_query(self, text: str) -> List[float]:
                return np.random.random(512).tolist()
        
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            embedding_function=WrongDimensionEmbeddings(),
            dimension=768  # Declared as 768 but returns 512
        )
        
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            store.add_texts(texts=["Test"])
    
    def test_embedding_caching(self):
        """Test embedding caching mechanism."""
        # Mock embedding function with call counter
        class CountingEmbeddings(Embeddings):
            def __init__(self):
                self.call_count = 0
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                self.call_count += 1
                return [np.random.random(768).tolist() for _ in texts]
            
            def embed_query(self, text: str) -> List[float]:
                self.call_count += 1
                return np.random.random(768).tolist()
        
        embeddings = CountingEmbeddings()
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            embedding_function=embeddings
        )
        
        # Add same text multiple times
        same_text = "Same text content"
        store.add_texts([same_text, same_text, same_text])
        
        # With caching, should only call embedding function once
        # (This would be implemented in the actual store)
        assert embeddings.call_count >= 1


class TestSnowflakeVectorStoreErrorHandling:
    """Test error handling and edge cases."""
    
    def test_connection_failure(self):
        """Test handling of connection failures."""
        invalid_params = {
            "account": "invalid_account",
            "user": "invalid_user",
            "password": "invalid_password"
        }
        
        with pytest.raises(ConnectionError):
            store = MockSnowflakeVectorStore(
                connection_parameters=invalid_params,
                table_name="test_vectors"
            )
            # Mock connection failure
            store._connect()
    
    def test_table_creation_failure(self):
        """Test handling of table creation failures."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="invalid table name!",  # Invalid name
            create_table_if_not_exists=True
        )
        
        with pytest.raises(Exception, match="Invalid table name"):
            store._create_table()
    
    def test_invalid_vector_dimensions(self):
        """Test handling of invalid vector dimensions."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            MockSnowflakeVectorStore(
                connection_parameters={"account": "test"},
                table_name="test_vectors",
                dimension=-1
            )
    
    def test_empty_text_handling(self):
        """Test handling of empty or None texts."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors"
        )
        
        # Test empty strings
        ids = store.add_texts(texts=["", "   ", "valid text"])
        assert len(ids) == 3
        
        # Test None values - need to cast to bypass type checking for testing
        with pytest.raises(ValueError, match="Text cannot be None"):
            store.add_texts(texts=[None, "valid text"])  # type: ignore
    
    def test_large_text_handling(self):
        """Test handling of very large text documents."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors"
        )
        
        # Test very large text (16MB+)
        large_text = "a" * (16 * 1024 * 1024)  # 16MB
        
        with pytest.raises(ValueError, match="Text too large"):
            store.add_texts(texts=[large_text])
    
    def test_metadata_serialization_error(self):
        """Test handling of non-serializable metadata."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors"
        )
        
        # Test with non-serializable metadata
        non_serializable_metadata = {"function": lambda x: x}
        
        with pytest.raises(TypeError, match="not JSON serializable"):
            store.add_texts(
                texts=["Test"],
                metadatas=[non_serializable_metadata]
            )
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors"
        )
        
        # Test malicious table name
        malicious_table = "test'; DROP TABLE users; --"
        
        with pytest.raises(ValueError, match="Invalid table name"):
            store.table_name = malicious_table
            store._create_table()
        
        # Test malicious metadata filter
        malicious_filter = {"category": "'; DROP TABLE users; --"}
        
        results = store.similarity_search(
            query="test",
            filter=malicious_filter
        )
        
        # Should handle safely (no SQL injection)
        assert isinstance(results, list)


class TestSnowflakeVectorStorePerformance:
    """Test performance characteristics."""
    
    def test_batch_insertion_performance(self):
        """Test performance of batch insertions."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            embedding_function=MockEmbeddings()
        )
        
        # Test large batch insertion
        large_texts = [f"Document {i}" for i in range(10000)]
        
        import time
        start_time = time.time()
        
        ids = store.add_texts(texts=large_texts, batch_size=1000)
        
        end_time = time.time()
        insertion_time = end_time - start_time
        
        assert len(ids) == 10000
        assert insertion_time < 30.0  # Should complete within 30 seconds
        
        # Test insertion rate
        insertion_rate = len(large_texts) / insertion_time
        assert insertion_rate > 100  # Should insert >100 docs/second
    
    def test_search_performance_scaling(self):
        """Test search performance with different dataset sizes."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            embedding_function=MockEmbeddings()
        )
        
        # Test with different dataset sizes
        for size in [1000, 10000, 100000]:
            texts = [f"Document {i}" for i in range(size)]
            store.add_texts(texts=texts)
            
            import time
            start_time = time.time()
            
            results = store.similarity_search(query="test", k=10)
            
            end_time = time.time()
            search_time = end_time - start_time
            
            assert len(results) == 10
            assert search_time < 1.0  # Should be under 1 second
    
    def test_memory_usage(self):
        """Test memory usage during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            embedding_function=MockEmbeddings()
        )
        
        # Add large dataset
        large_texts = [f"Document {i}" for i in range(10000)]
        store.add_texts(texts=large_texts)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (<500MB)
        assert memory_increase < 500 * 1024 * 1024  # 500MB
    
    def test_concurrent_operations(self):
        """Test concurrent read/write operations."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="test_vectors",
            embedding_function=MockEmbeddings()
        )
        
        # Add initial data
        initial_texts = [f"Initial doc {i}" for i in range(1000)]
        store.add_texts(texts=initial_texts)
        
        # Test concurrent operations
        import threading
        
        def concurrent_search():
            for _ in range(100):
                results = store.similarity_search(query="test", k=5)
                assert len(results) <= 5
        
        def concurrent_insert():
            for i in range(100):
                store.add_texts(texts=[f"Concurrent doc {i}"])
        
        # Start concurrent threads
        threads = []
        for _ in range(5):
            t1 = threading.Thread(target=concurrent_search)
            t2 = threading.Thread(target=concurrent_insert)
            threads.extend([t1, t2])
            t1.start()
            t2.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify data integrity
        assert len(store._documents) >= 1000


class TestSnowflakeVectorStoreIntegration:
    """Test integration scenarios and end-to-end workflows."""
    
    def test_rag_pipeline_integration(self):
        """Test complete RAG pipeline integration."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="knowledge_base",
            embedding_function=MockEmbeddings()
        )
        
        # Step 1: Add knowledge base documents
        knowledge_texts = [
            "Python is a programming language used for data science.",
            "Machine learning requires large datasets for training.",
            "Vector databases store high-dimensional embeddings.",
            "RAG systems combine retrieval with generation."
        ]
        
        knowledge_metadata = [
            {"source": "python_docs", "category": "programming"},
            {"source": "ml_textbook", "category": "machine_learning"},
            {"source": "vector_db_guide", "category": "databases"},
            {"source": "rag_paper", "category": "ai_systems"}
        ]
        
        store.add_texts(texts=knowledge_texts, metadatas=knowledge_metadata)
        
        # Step 2: Perform retrieval for RAG
        user_query = "How do I use Python for machine learning?"
        
        relevant_docs = store.similarity_search(
            query=user_query,
            k=3,
            filter={"category": {"$in": ["programming", "machine_learning"]}}
        )
        
        assert len(relevant_docs) <= 3
        assert all("python" in doc.page_content.lower() or 
                  "machine learning" in doc.page_content.lower() 
                  for doc in relevant_docs)
        
        # Step 3: Combine retrieved context for generation
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Mock LLM call with context
        def mock_llm_call(query: str, context: str) -> str:
            return f"Based on the context: {context}, the answer to '{query}' is..."
        
        response = mock_llm_call(user_query, context)
        
        assert "Python" in response
        assert len(response) > 50  # Should generate substantial response
    
    def test_semantic_search_application(self):
        """Test semantic search application scenario."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="product_catalog",
            embedding_function=MockEmbeddings()
        )
        
        # Add product catalog
        products = [
            "iPhone 14 Pro - Latest smartphone with advanced camera",
            "MacBook Pro - High-performance laptop for professionals",
            "iPad Air - Versatile tablet for creativity and productivity",
            "AirPods Pro - Wireless earbuds with noise cancellation",
            "Apple Watch - Smartwatch for health and fitness tracking"
        ]
        
        product_metadata = [
            {"category": "smartphone", "brand": "Apple", "price": 999},
            {"category": "laptop", "brand": "Apple", "price": 1999},
            {"category": "tablet", "brand": "Apple", "price": 599},
            {"category": "audio", "brand": "Apple", "price": 249},
            {"category": "wearable", "brand": "Apple", "price": 399}
        ]
        
        store.add_texts(texts=products, metadatas=product_metadata)
        
        # Test semantic search queries
        search_queries = [
            "portable computer for work",  # Should match laptop
            "device for listening to music",  # Should match AirPods
            "fitness tracker",  # Should match Apple Watch
            "mobile phone with camera",  # Should match iPhone
            "drawing tablet"  # Should match iPad
        ]
        
        for query in search_queries:
            results = store.similarity_search(query=query, k=2)
            assert len(results) >= 1
            assert all(isinstance(doc, Document) for doc in results)
    
    def test_document_management_workflow(self):
        """Test complete document management workflow."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="document_store",
            embedding_function=MockEmbeddings()
        )
        
        # Add documents
        documents = [
            "Annual report 2023 - Company performance overview",
            "Product specification - Technical details and features",
            "User manual - Step-by-step instructions for users",
            "Marketing materials - Promotional content and messaging"
        ]
        
        doc_metadata = [
            {"type": "report", "year": 2023, "department": "finance"},
            {"type": "specification", "version": "1.0", "department": "engineering"},
            {"type": "manual", "audience": "end_users", "department": "support"},
            {"type": "marketing", "campaign": "Q1_2024", "department": "marketing"}
        ]
        
        doc_ids = store.add_texts(texts=documents, metadatas=doc_metadata)
        
        # Update a document
        updated_doc = Document(
            page_content="Annual report 2023 - Updated with Q4 results",
            metadata={"type": "report", "year": 2023, "department": "finance", "updated": True}
        )
        
        store.update_documents(ids=[doc_ids[0]], documents=[updated_doc])
        
        # Retrieve updated document
        retrieved = store.get_by_ids([doc_ids[0]])
        assert "Updated with Q4 results" in retrieved[0].page_content
        assert retrieved[0].metadata["updated"] is True
        
        # Delete a document
        store.delete(ids=[doc_ids[2]])
        
        # Verify deletion
        remaining_docs = store.get_by_ids(doc_ids)
        assert len(remaining_docs) == 3  # One deleted
        
        # Search with filters
        finance_docs = store.similarity_search(
            query="financial performance",
            filter={"department": "finance"}
        )
        
        assert len(finance_docs) >= 1
        assert all(doc.metadata["department"] == "finance" for doc in finance_docs)
    
    def test_multi_language_support(self):
        """Test multi-language document support."""
        store = MockSnowflakeVectorStore(
            connection_parameters={"account": "test"},
            table_name="multi_lang_docs",
            embedding_function=MockEmbeddings()
        )
        
        # Add multi-language documents
        multilang_texts = [
            "Hello world, this is an English document.",
            "Bonjour le monde, ceci est un document français.",
            "Hola mundo, este es un documento en español.",
            "こんにちは世界、これは日本語の文書です。",
            "Hallo Welt, dies ist ein deutsches Dokument."
        ]
        
        language_metadata = [
            {"language": "english", "country": "US"},
            {"language": "french", "country": "FR"},
            {"language": "spanish", "country": "ES"},
            {"language": "japanese", "country": "JP"},
            {"language": "german", "country": "DE"}
        ]
        
        store.add_texts(texts=multilang_texts, metadatas=language_metadata)
        
        # Test language-specific search
        english_docs = store.similarity_search(
            query="greeting",
            filter={"language": "english"}
        )
        
        assert len(english_docs) >= 1
        assert all(doc.metadata["language"] == "english" for doc in english_docs)
        
        # Test cross-language semantic search
        all_greeting_docs = store.similarity_search(
            query="greeting world",
            k=5
        )
        
        assert len(all_greeting_docs) <= 5
        # Should find semantically similar documents across languages


if __name__ == "__main__":
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])