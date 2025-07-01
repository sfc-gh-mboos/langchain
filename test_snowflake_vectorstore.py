"""
Comprehensive test suite for Snowflake Vector Store implementation.

This test file demonstrates the testing approach for the Snowflake vector store,
including unit tests, integration tests, and LangChain standard test compliance.

Note: This test file requires dependencies listed in requirements_test.txt to run.
The imports may show as unresolved in this context but would work in a proper
LangChain development environment.
"""

import asyncio
import os
import uuid
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests


# Mock Embeddings for testing
class MockEmbeddings(Embeddings):
    """Mock embeddings class for testing purposes."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for documents."""
        return [self._generate_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Generate mock embedding for a query."""
        return self._generate_embedding(text)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a deterministic embedding based on text."""
        # Use hash of text to generate deterministic but varied embeddings
        np.random.seed(hash(text) % (2**32))
        return np.random.random(self.dimension).tolist()


# Mock SnowflakeVectorStore implementation for testing
class MockSnowflakeVectorStore(VectorStore):
    """Mock implementation of SnowflakeVectorStore for testing."""
    
    def __init__(
        self,
        embedding: Embeddings,
        account: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        warehouse: Optional[str] = None,
        role: Optional[str] = None,
        table_name: str = "vector_store",
        **kwargs: Any
    ):
        """Initialize the mock vector store."""
        self.embedding = embedding
        self.account = account or os.getenv("SNOWFLAKE_ACCOUNT")
        self.username = username or os.getenv("SNOWFLAKE_USERNAME")
        self.password = password or os.getenv("SNOWFLAKE_PASSWORD")
        self.database = database or os.getenv("SNOWFLAKE_DATABASE")
        self.schema = schema or os.getenv("SNOWFLAKE_SCHEMA")
        self.warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE")
        self.role = role or os.getenv("SNOWFLAKE_ROLE")
        self.table_name = table_name
        
        # Mock storage for testing
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._session = Mock()
        
        # Validate required parameters
        if not all([self.account, self.username, self.password, 
                   self.database, self.schema, self.warehouse]):
            raise ValueError("Missing required Snowflake connection parameters")
    
    @property
    def embeddings(self) -> Embeddings:
        """Return the embeddings instance."""
        return self.embedding
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add documents to the vector store."""
        if ids and len(ids) != len(documents):
            raise ValueError(f"Number of ids ({len(ids)}) must match number of documents ({len(documents)})")
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding.embed_documents(texts)
        
        doc_ids = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = ids[i] if ids else str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            self._documents[doc_id] = {
                "id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": embedding
            }
        
        return doc_ids
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete documents from the vector store."""
        if ids:
            for doc_id in ids:
                self._documents.pop(doc_id, None)
    
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by their IDs."""
        documents = []
        for doc_id in ids:
            if doc_id in self._documents:
                doc_data = self._documents[doc_id]
                documents.append(Document(
                    page_content=doc_data["content"],
                    metadata=doc_data["metadata"]
                ))
        return documents
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Perform similarity search."""
        query_embedding = self.embedding.embed_query(query)
        results = self._similarity_search_with_score_by_vector(
            query_embedding, k, filter, **kwargs
        )
        return [doc for doc, _ in results]
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """Perform similarity search with scores."""
        query_embedding = self.embedding.embed_query(query)
        return self._similarity_search_with_score_by_vector(
            query_embedding, k, filter, **kwargs
        )
    
    def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """Internal method for similarity search by vector."""
        if not self._documents:
            return []
        
        # Calculate cosine similarity
        similarities = []
        for doc_id, doc_data in self._documents.items():
            # Apply filter if provided
            if filter:
                doc_metadata = doc_data["metadata"]
                if not all(doc_metadata.get(k) == v for k, v in filter.items()):
                    continue
            
            # Calculate cosine similarity
            doc_embedding = np.array(doc_data["embedding"])
            query_embedding_np = np.array(embedding)
            
            # Normalize vectors
            doc_norm = np.linalg.norm(doc_embedding)
            query_norm = np.linalg.norm(query_embedding_np)
            
            if doc_norm == 0 or query_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(doc_embedding, query_embedding_np) / (doc_norm * query_norm)
            
            similarities.append((doc_data, float(similarity)))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:k]
        
        # Convert to Document objects
        results = []
        for doc_data, score in similarities:
            doc = Document(
                page_content=doc_data["content"],
                metadata=doc_data["metadata"]
            )
            results.append((doc, score))
        
        return results
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> "MockSnowflakeVectorStore":
        """Create vector store from texts."""
        store = cls(embedding=embedding, **kwargs)
        documents = [
            Document(page_content=text, metadata=metadatas[i] if metadatas else {})
            for i, text in enumerate(texts)
        ]
        store.add_documents(documents)
        return store


class TestSnowflakeVectorStoreUnit:
    """Unit tests for SnowflakeVectorStore."""
    
    @pytest.fixture
    def mock_embeddings(self) -> MockEmbeddings:
        """Create mock embeddings for testing."""
        return MockEmbeddings(dimension=384)
    
    @pytest.fixture
    def mock_snowflake_env(self, monkeypatch):
        """Mock Snowflake environment variables."""
        env_vars = {
            "SNOWFLAKE_ACCOUNT": "test_account",
            "SNOWFLAKE_USERNAME": "test_user",
            "SNOWFLAKE_PASSWORD": "test_password",
            "SNOWFLAKE_DATABASE": "test_db",
            "SNOWFLAKE_SCHEMA": "test_schema",
            "SNOWFLAKE_WAREHOUSE": "test_warehouse",
            "SNOWFLAKE_ROLE": "test_role"
        }
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
        return env_vars
    
    @pytest.fixture
    def vector_store(self, mock_embeddings, mock_snowflake_env) -> MockSnowflakeVectorStore:
        """Create a mock vector store for testing."""
        return MockSnowflakeVectorStore(embedding=mock_embeddings)
    
    def test_initialization_with_env_vars(self, mock_embeddings, mock_snowflake_env):
        """Test vector store initialization with environment variables."""
        store = MockSnowflakeVectorStore(embedding=mock_embeddings)
        assert store.account == "test_account"
        assert store.username == "test_user"
        assert store.database == "test_db"
    
    def test_initialization_with_parameters(self, mock_embeddings):
        """Test vector store initialization with explicit parameters."""
        store = MockSnowflakeVectorStore(
            embedding=mock_embeddings,
            account="param_account",
            username="param_user",
            password="param_password",
            database="param_db",
            schema="param_schema",
            warehouse="param_warehouse",
            role="param_role"
        )
        assert store.account == "param_account"
        assert store.username == "param_user"
    
    def test_initialization_missing_parameters(self, mock_embeddings):
        """Test that initialization fails with missing parameters."""
        with pytest.raises(ValueError, match="Missing required Snowflake connection parameters"):
            MockSnowflakeVectorStore(embedding=mock_embeddings)
    
    def test_add_documents(self, vector_store):
        """Test adding documents to the vector store."""
        documents = [
            Document(page_content="test document 1", metadata={"type": "test"}),
            Document(page_content="test document 2", metadata={"type": "test"})
        ]
        
        ids = vector_store.add_documents(documents)
        
        assert len(ids) == 2
        assert all(isinstance(doc_id, str) for doc_id in ids)
        assert len(vector_store._documents) == 2
    
    def test_add_documents_with_ids(self, vector_store):
        """Test adding documents with custom IDs."""
        documents = [
            Document(page_content="test document 1", metadata={"type": "test"})
        ]
        custom_ids = ["custom_id_1"]
        
        ids = vector_store.add_documents(documents, ids=custom_ids)
        
        assert ids == custom_ids
        assert "custom_id_1" in vector_store._documents
    
    def test_add_documents_mismatched_ids(self, vector_store):
        """Test error when number of IDs doesn't match number of documents."""
        documents = [
            Document(page_content="test document 1"),
            Document(page_content="test document 2")
        ]
        ids = ["id1"]  # Only one ID for two documents
        
        with pytest.raises(ValueError, match="Number of ids .* must match number of documents"):
            vector_store.add_documents(documents, ids=ids)
    
    def test_delete_documents(self, vector_store):
        """Test deleting documents from the vector store."""
        documents = [
            Document(page_content="test document 1"),
            Document(page_content="test document 2")
        ]
        ids = vector_store.add_documents(documents)
        
        # Delete one document
        vector_store.delete([ids[0]])
        
        assert len(vector_store._documents) == 1
        assert ids[0] not in vector_store._documents
        assert ids[1] in vector_store._documents
    
    def test_get_by_ids(self, vector_store):
        """Test retrieving documents by IDs."""
        documents = [
            Document(page_content="test document 1", metadata={"type": "test1"}),
            Document(page_content="test document 2", metadata={"type": "test2"})
        ]
        ids = vector_store.add_documents(documents)
        
        retrieved_docs = vector_store.get_by_ids([ids[0]])
        
        assert len(retrieved_docs) == 1
        assert retrieved_docs[0].page_content == "test document 1"
        assert retrieved_docs[0].metadata == {"type": "test1"}
    
    def test_similarity_search(self, vector_store):
        """Test similarity search functionality."""
        documents = [
            Document(page_content="machine learning algorithms", metadata={"topic": "ml"}),
            Document(page_content="natural language processing", metadata={"topic": "nlp"}),
            Document(page_content="computer vision techniques", metadata={"topic": "cv"})
        ]
        vector_store.add_documents(documents)
        
        results = vector_store.similarity_search("machine learning", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
    
    def test_similarity_search_with_score(self, vector_store):
        """Test similarity search with scores."""
        documents = [
            Document(page_content="machine learning algorithms"),
            Document(page_content="deep learning neural networks"),
            Document(page_content="cooking recipes")
        ]
        vector_store.add_documents(documents)
        
        results = vector_store.similarity_search_with_score("machine learning", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
        assert all(isinstance(doc, Document) and isinstance(score, float) 
                  for doc, score in results)
        
        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_similarity_search_with_filter(self, vector_store):
        """Test similarity search with metadata filtering."""
        documents = [
            Document(page_content="machine learning", metadata={"category": "ai"}),
            Document(page_content="machine learning", metadata={"category": "ml"}),
            Document(page_content="cooking", metadata={"category": "food"})
        ]
        vector_store.add_documents(documents)
        
        results = vector_store.similarity_search(
            "machine learning", 
            k=10, 
            filter={"category": "ai"}
        )
        
        assert len(results) == 1
        assert results[0].metadata["category"] == "ai"
    
    def test_from_texts(self, mock_embeddings, mock_snowflake_env):
        """Test creating vector store from texts."""
        texts = ["text 1", "text 2", "text 3"]
        metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        store = MockSnowflakeVectorStore.from_texts(
            texts=texts,
            embedding=mock_embeddings,
            metadatas=metadatas
        )
        
        assert len(store._documents) == 3
        
        # Test search functionality
        results = store.similarity_search("text 1", k=1)
        assert len(results) == 1


class TestSnowflakeVectorStoreIntegration(VectorStoreIntegrationTests):
    """Integration tests using LangChain's standard test suite."""
    
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:
        """Get an empty vectorstore for integration tests."""
        # Mock environment variables for testing
        with patch.dict(os.environ, {
            "SNOWFLAKE_ACCOUNT": "test_account",
            "SNOWFLAKE_USERNAME": "test_user", 
            "SNOWFLAKE_PASSWORD": "test_password",
            "SNOWFLAKE_DATABASE": "test_db",
            "SNOWFLAKE_SCHEMA": "test_schema",
            "SNOWFLAKE_WAREHOUSE": "test_warehouse",
            "SNOWFLAKE_ROLE": "test_role"
        }):
            store = MockSnowflakeVectorStore(embedding=self.get_embeddings())
            try:
                yield store
            finally:
                # Cleanup - clear all documents
                store._documents.clear()


class TestSnowflakeVectorStoreAsync:
    """Test async functionality (when implemented)."""
    
    @pytest.fixture
    def vector_store(self, mock_snowflake_env) -> MockSnowflakeVectorStore:
        """Create vector store for async testing."""
        return MockSnowflakeVectorStore(embedding=MockEmbeddings())
    
    @pytest.mark.asyncio
    async def test_async_add_documents(self, vector_store):
        """Test async document addition (placeholder for future implementation)."""
        # This would test the async version when implemented
        documents = [
            Document(page_content="async test document", metadata={"async": True})
        ]
        
        # For now, just test that sync version works
        ids = vector_store.add_documents(documents)
        assert len(ids) == 1
    
    @pytest.mark.asyncio
    async def test_async_similarity_search(self, vector_store):
        """Test async similarity search (placeholder for future implementation)."""
        documents = [
            Document(page_content="async search test", metadata={"async": True})
        ]
        vector_store.add_documents(documents)
        
        # For now, just test that sync version works
        results = vector_store.similarity_search("async search", k=1)
        assert len(results) == 1


class TestSnowflakeVectorStorePerformance:
    """Performance tests for the vector store."""
    
    @pytest.fixture
    def large_vector_store(self, mock_snowflake_env) -> MockSnowflakeVectorStore:
        """Create vector store with larger dataset for performance testing."""
        store = MockSnowflakeVectorStore(embedding=MockEmbeddings())
        
        # Add a larger number of documents for performance testing
        documents = [
            Document(
                page_content=f"performance test document {i}",
                metadata={"index": i, "batch": i // 100}
            )
            for i in range(1000)
        ]
        store.add_documents(documents)
        return store
    
    def test_large_batch_insertion(self, mock_snowflake_env):
        """Test performance of large batch document insertion."""
        store = MockSnowflakeVectorStore(embedding=MockEmbeddings())
        
        # Create a large batch of documents
        documents = [
            Document(page_content=f"batch document {i}")
            for i in range(1000)
        ]
        
        import time
        start_time = time.time()
        ids = store.add_documents(documents)
        end_time = time.time()
        
        assert len(ids) == 1000
        assert (end_time - start_time) < 10  # Should complete within 10 seconds
    
    def test_similarity_search_performance(self, large_vector_store):
        """Test similarity search performance with large dataset."""
        import time
        
        start_time = time.time()
        results = large_vector_store.similarity_search("performance test", k=10)
        end_time = time.time()
        
        assert len(results) == 10
        assert (end_time - start_time) < 5  # Should complete within 5 seconds
    
    def test_filtered_search_performance(self, large_vector_store):
        """Test filtered search performance."""
        import time
        
        start_time = time.time()
        results = large_vector_store.similarity_search(
            "performance test", 
            k=10, 
            filter={"batch": 5}
        )
        end_time = time.time()
        
        assert len(results) <= 10
        assert (end_time - start_time) < 5  # Should complete within 5 seconds
        
        # All results should match the filter
        for result in results:
            assert result.metadata["batch"] == 5


class TestSnowflakeVectorStoreErrorHandling:
    """Test error handling and edge cases."""
    
    def test_connection_failure(self, mock_embeddings):
        """Test handling of connection failures."""
        # Test with invalid connection parameters
        with pytest.raises(ValueError):
            MockSnowflakeVectorStore(
                embedding=mock_embeddings,
                account="invalid"
                # Missing other required parameters
            )
    
    def test_empty_documents_list(self, mock_snowflake_env):
        """Test handling of empty documents list."""
        store = MockSnowflakeVectorStore(embedding=MockEmbeddings())
        
        ids = store.add_documents([])
        assert ids == []
    
    def test_search_empty_store(self, mock_snowflake_env):
        """Test search on empty vector store."""
        store = MockSnowflakeVectorStore(embedding=MockEmbeddings())
        
        results = store.similarity_search("test query")
        assert results == []
        
        results_with_score = store.similarity_search_with_score("test query")
        assert results_with_score == []
    
    def test_delete_nonexistent_ids(self, mock_snowflake_env):
        """Test deleting non-existent document IDs."""
        store = MockSnowflakeVectorStore(embedding=MockEmbeddings())
        
        # Should not raise an error
        store.delete(["nonexistent_id"])
        assert len(store._documents) == 0
    
    def test_get_nonexistent_ids(self, mock_snowflake_env):
        """Test retrieving non-existent document IDs."""
        store = MockSnowflakeVectorStore(embedding=MockEmbeddings())
        
        results = store.get_by_ids(["nonexistent_id"])
        assert results == []


# Utility functions for test data generation
def generate_test_documents(count: int = 10) -> List[Document]:
    """Generate test documents for testing."""
    return [
        Document(
            page_content=f"Test document {i} with content about topic {i % 3}",
            metadata={
                "id": i,
                "topic": f"topic_{i % 3}",
                "category": "test",
                "timestamp": f"2024-01-{i:02d}"
            }
        )
        for i in range(count)
    ]


def generate_test_embeddings(texts: List[str], dimension: int = 384) -> List[List[float]]:
    """Generate deterministic test embeddings."""
    embeddings = []
    for text in texts:
        # Use hash for deterministic but varied embeddings
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.random(dimension).tolist()
        embeddings.append(embedding)
    return embeddings


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])