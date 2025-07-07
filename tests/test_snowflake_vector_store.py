"""
Test suite for Snowflake Vector Store integration in task management applications.

This test suite validates:
1. Core vector store operations (CRUD, similarity search)
2. Task management use cases (task similarity, recommendations)
3. Performance and scalability
4. Integration with LangChain ecosystem
5. Error handling and edge cases
"""

import asyncio
import os
import time
import uuid
from typing import Generator, List, Optional, Dict, Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

try:
    import snowflake.connector
    from snowflake.connector import SnowflakeConnection
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False


class MockSnowflakeEmbeddings(Embeddings):
    """Mock embeddings for testing without actual Snowflake Cortex calls."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1 * i for i in range(self.dimension)] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return [0.1 * i for i in range(self.dimension)]


class SnowflakeVectorStore(VectorStore):
    """
    Snowflake Vector Store implementation for task management applications.
    
    This implementation provides vector similarity search capabilities
    using Snowflake's native vector functions and data types.
    """
    
    def __init__(
        self,
        connection: SnowflakeConnection,
        embedding: Embeddings,
        table_name: str = "vector_store",
        dimension: int = 768,
        distance_strategy: str = "cosine"
    ):
        self.connection = connection
        self._embedding = embedding
        self.table_name = table_name
        self.dimension = dimension
        self.distance_strategy = distance_strategy
        self._initialize_table()
    
    def _initialize_table(self):
        """Initialize the vector store table with proper schema."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id STRING,
                    content TEXT,
                    metadata VARIANT,
                    vector VECTOR(FLOAT, {self.dimension}),
                    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
                )
            """)
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_id ON {self.table_name}(id)")
        finally:
            cursor.close()
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to the vector store."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        texts = [doc.page_content for doc in documents]
        embeddings = self._embedding.embed_documents(texts)
        
        cursor = self.connection.cursor()
        try:
            for i, doc in enumerate(documents):
                cursor.execute(f"""
                    INSERT INTO {self.table_name} (id, content, metadata, vector)
                    VALUES (?, ?, ?, ?)
                """, (
                    ids[i],
                    doc.page_content,
                    doc.metadata,
                    embeddings[i]
                ))
        finally:
            cursor.close()
        
        return ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents using vector similarity."""
        query_embedding = self._embedding.embed_query(query)
        
        similarity_func = {
            "cosine": "VECTOR_COSINE_SIMILARITY",
            "inner_product": "VECTOR_INNER_PRODUCT",
            "l1": "VECTOR_L1_DISTANCE",
            "l2": "VECTOR_L2_DISTANCE"
        }[self.distance_strategy]
        
        cursor = self.connection.cursor()
        try:
            base_query = f"""
                SELECT id, content, metadata, 
                       {similarity_func}(vector, ?::VECTOR(FLOAT, {self.dimension})) as similarity
                FROM {self.table_name}
            """
            
            params = [query_embedding]
            
            if filter:
                filter_conditions = []
                for key, value in filter.items():
                    filter_conditions.append(f"metadata:{key}::STRING = ?")
                    params.append(str(value))
                base_query += " WHERE " + " AND ".join(filter_conditions)
            
            if self.distance_strategy in ["l1", "l2"]:
                base_query += " ORDER BY similarity ASC"
            else:
                base_query += " ORDER BY similarity DESC"
            
            base_query += f" LIMIT {k}"
            
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            
            documents = []
            for row in results:
                doc = Document(
                    page_content=row[1],
                    metadata=row[2] or {},
                    id=row[0]
                )
                documents.append(doc)
            
            return documents
        finally:
            cursor.close()
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        cursor = self.connection.cursor()
        try:
            placeholders = ",".join(["?" for _ in ids])
            cursor.execute(f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})", ids)
        finally:
            cursor.close()
    
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs."""
        cursor = self.connection.cursor()
        try:
            placeholders = ",".join(["?" for _ in ids])
            cursor.execute(f"""
                SELECT id, content, metadata 
                FROM {self.table_name} 
                WHERE id IN ({placeholders})
            """, ids)
            results = cursor.fetchall()
            
            id_to_doc = {row[0]: Document(page_content=row[1], metadata=row[2] or {}, id=row[0]) 
                        for row in results}
            
            return [id_to_doc.get(id) for id in ids if id in id_to_doc]
        finally:
            cursor.close()
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any
    ) -> "SnowflakeVectorStore":
        """Create a vector store from documents."""
        instance = cls(embedding=embedding, **kwargs)
        instance.add_documents(documents)
        return instance


@pytest.mark.skipif(not SNOWFLAKE_AVAILABLE, reason="Snowflake not available")
class TestSnowflakeVectorStoreStandard(VectorStoreIntegrationTests):
    """Standard LangChain vector store tests for Snowflake."""
    
    @pytest.fixture
    def connection(self):
        """Create a Snowflake connection for testing."""
        return Mock(spec=SnowflakeConnection)
    
    @pytest.fixture
    def vectorstore(self, connection) -> Generator[VectorStore, None, None]:
        """Get an empty vectorstore for testing."""
        embeddings = MockSnowflakeEmbeddings(dimension=768)
        
        with patch('snowflake.connector.connect') as mock_connect:
            mock_connect.return_value = connection
            
            mock_cursor = Mock()
            connection.cursor.return_value = mock_cursor
            
            store = SnowflakeVectorStore(
                connection=connection,
                embedding=embeddings,
                table_name=f"test_vectors_{uuid.uuid4().hex}"
            )
            
            try:
                yield store
            finally:
                try:
                    store.delete_collection()
                except:
                    pass
    
    def test_snowflake_vector_types(self, vectorstore):
        """Test Snowflake-specific vector type handling."""
        documents = [
            Document(page_content="Task: Implement user authentication", metadata={"priority": "high"}),
            Document(page_content="Task: Update documentation", metadata={"priority": "low"})
        ]
        
        ids = vectorstore.add_documents(documents)
        assert len(ids) == 2
        
        retrieved = vectorstore.get_by_ids(ids)
        assert len(retrieved) == 2
        assert retrieved[0].page_content == documents[0].page_content


class TestSnowflakeVectorStoreTaskManagement:
    """Task management specific tests for Snowflake vector store."""
    
    @pytest.fixture
    def task_vectorstore(self):
        """Create a vector store with task management sample data."""
        embeddings = MockSnowflakeEmbeddings(dimension=768)
        connection = Mock(spec=SnowflakeConnection)
        
        with patch('snowflake.connector.connect') as mock_connect:
            mock_connect.return_value = connection
            
            mock_cursor = Mock()
            connection.cursor.return_value = mock_cursor
            
            store = SnowflakeVectorStore(
                connection=connection,
                embedding=embeddings,
                table_name="task_vectors"
            )
            
            sample_tasks = [
                Document(
                    page_content="Implement user authentication system with OAuth2",
                    metadata={"project": "auth", "priority": "high", "assignee": "john", "type": "feature"}
                ),
                Document(
                    page_content="Fix login bug causing session timeout",
                    metadata={"project": "auth", "priority": "critical", "assignee": "jane", "type": "bug"}
                ),
                Document(
                    page_content="Write unit tests for authentication module",
                    metadata={"project": "auth", "priority": "medium", "assignee": "bob", "type": "test"}
                ),
                Document(
                    page_content="Update user documentation for new login flow",
                    metadata={"project": "docs", "priority": "low", "assignee": "alice", "type": "documentation"}
                ),
                Document(
                    page_content="Design new dashboard UI mockups",
                    metadata={"project": "ui", "priority": "medium", "assignee": "charlie", "type": "design"}
                )
            ]
            
            store.add_documents(sample_tasks)
            return store
    
    def test_task_similarity_search(self, task_vectorstore):
        """Test finding similar tasks based on content."""
        query = "authentication login issues"
        similar_tasks = task_vectorstore.similarity_search(query, k=3)
        
        assert len(similar_tasks) == 3
        assert "auth" in similar_tasks[0].page_content.lower()
    
    def test_task_filtering_by_project(self, task_vectorstore):
        """Test filtering tasks by project metadata."""
        query = "authentication"
        auth_tasks = task_vectorstore.similarity_search(
            query, 
            k=5, 
            filter={"project": "auth"}
        )
        
        assert len(auth_tasks) <= 5
        for task in auth_tasks:
            assert task.metadata["project"] == "auth"
    
    def test_task_priority_filtering(self, task_vectorstore):
        """Test filtering tasks by priority level."""
        query = "important urgent tasks"
        high_priority_tasks = task_vectorstore.similarity_search(
            query,
            k=5,
            filter={"priority": "high"}
        )
        
        for task in high_priority_tasks:
            assert task.metadata["priority"] == "high"
    
    def test_duplicate_task_detection(self, task_vectorstore):
        """Test detecting duplicate or similar tasks."""
        new_task = Document(
            page_content="Implement OAuth2 authentication for users",
            metadata={"project": "auth", "priority": "high", "assignee": "new_dev", "type": "feature"}
        )
        
        similar_tasks = task_vectorstore.similarity_search(
            new_task.page_content,
            k=3
        )
        
        assert len(similar_tasks) > 0
        assert "oauth" in similar_tasks[0].page_content.lower() or "auth" in similar_tasks[0].page_content.lower()
    
    def test_task_recommendation_system(self, task_vectorstore):
        """Test task recommendation based on user context."""
        user_context = "I work on authentication and security features"
        recommended_tasks = task_vectorstore.similarity_search(
            user_context,
            k=3
        )
        
        assert len(recommended_tasks) > 0
        auth_related = any("auth" in task.page_content.lower() for task in recommended_tasks)
        assert auth_related
    
    def test_project_categorization(self, task_vectorstore):
        """Test automatic project categorization based on task content."""
        uncategorized_task = "Create login page with forgot password link"
        similar_tasks = task_vectorstore.similarity_search(uncategorized_task, k=3)
        
        assert len(similar_tasks) > 0
        suggested_projects = [task.metadata.get("project") for task in similar_tasks]
        assert "auth" in suggested_projects


class TestSnowflakeVectorStorePerformance:
    """Performance and scalability tests for Snowflake vector store."""
    
    @pytest.fixture
    def large_vectorstore(self):
        """Create a vector store with large dataset for performance testing."""
        embeddings = MockSnowflakeEmbeddings(dimension=768)
        connection = Mock(spec=SnowflakeConnection)
        
        with patch('snowflake.connector.connect') as mock_connect:
            mock_connect.return_value = connection
            
            mock_cursor = Mock()
            connection.cursor.return_value = mock_cursor
            
            store = SnowflakeVectorStore(
                connection=connection,
                embedding=embeddings,
                table_name="large_task_vectors"
            )
            
            return store
    
    def test_batch_insert_performance(self, large_vectorstore):
        """Test batch insert performance with large datasets."""
        batch_size = 1000
        tasks = [
            Document(
                page_content=f"Task {i}: Implement feature {i}",
                metadata={"task_id": i, "priority": ["low", "medium", "high"][i % 3]}
            ) for i in range(batch_size)
        ]
        
        start_time = time.time()
        ids = large_vectorstore.add_documents(tasks)
        end_time = time.time()
        
        assert len(ids) == batch_size
        insert_time = end_time - start_time
        throughput = batch_size / insert_time
        
        assert throughput > 100, f"Insert throughput {throughput} docs/sec is too low"
    
    def test_search_performance(self, large_vectorstore):
        """Test search performance with large datasets."""
        query = "implement authentication feature"
        k_values = [1, 10, 50, 100]
        
        for k in k_values:
            start_time = time.time()
            results = large_vectorstore.similarity_search(query, k=k)
            end_time = time.time()
            
            search_time = end_time - start_time
            assert search_time < 0.1, f"Search time {search_time}s for k={k} is too slow"
            assert len(results) <= k
    
    def test_concurrent_operations(self, large_vectorstore):
        """Test concurrent read/write operations."""
        async def concurrent_search(query_id):
            return large_vectorstore.similarity_search(f"query {query_id}", k=5)
        
        async def run_concurrent_searches():
            tasks = [concurrent_search(i) for i in range(10)]
            return await asyncio.gather(*tasks)
        
        start_time = time.time()
        results = asyncio.run(run_concurrent_searches())
        end_time = time.time()
        
        assert len(results) == 10
        concurrent_time = end_time - start_time
        assert concurrent_time < 1.0, f"Concurrent operations took {concurrent_time}s"
    
    def test_memory_usage_monitoring(self, large_vectorstore):
        """Test memory usage during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        batch_size = 10000
        tasks = [
            Document(
                page_content=f"Memory test task {i}",
                metadata={"task_id": i}
            ) for i in range(batch_size)
        ]
        
        large_vectorstore.add_documents(tasks)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 512, f"Memory increase {memory_increase}MB is too high"


class TestSnowflakeVectorStoreIntegration:
    """Integration tests with LangChain ecosystem."""
    
    @pytest.fixture
    def integrated_vectorstore(self):
        """Create a vector store integrated with LangChain components."""
        embeddings = MockSnowflakeEmbeddings(dimension=768)
        connection = Mock(spec=SnowflakeConnection)
        
        with patch('snowflake.connector.connect') as mock_connect:
            mock_connect.return_value = connection
            
            mock_cursor = Mock()
            connection.cursor.return_value = mock_cursor
            
            store = SnowflakeVectorStore(
                connection=connection,
                embedding=embeddings,
                table_name="integrated_vectors"
            )
            
            return store
    
    def test_retriever_integration(self, integrated_vectorstore):
        """Test integration with LangChain retrievers."""
        from langchain_core.retrievers import BaseRetriever
        
        retriever = integrated_vectorstore.as_retriever(search_kwargs={"k": 3})
        assert isinstance(retriever, BaseRetriever)
        
        documents = [
            Document(page_content="Test document for retrieval"),
            Document(page_content="Another test document")
        ]
        integrated_vectorstore.add_documents(documents)
        
        results = retriever.get_relevant_documents("test document")
        assert len(results) <= 3
    
    def test_multiple_embedding_models(self, integrated_vectorstore):
        """Test support for different embedding models."""
        models = [
            MockSnowflakeEmbeddings(dimension=768),
            MockSnowflakeEmbeddings(dimension=1024),
        ]
        
        for model in models:
            test_doc = Document(page_content="Test document for model compatibility")
            embedding = model.embed_query(test_doc.page_content)
            
            assert len(embedding) == model.dimension
            assert all(isinstance(x, float) for x in embedding)
    
    def test_async_operations(self, integrated_vectorstore):
        """Test async operations support."""
        async def async_search():
            return integrated_vectorstore.similarity_search("async test", k=1)
        
        results = asyncio.run(async_search())
        assert isinstance(results, list)


class TestSnowflakeVectorStoreErrorHandling:
    """Error handling and edge case tests."""
    
    def test_connection_failure_handling(self):
        """Test handling of connection failures."""
        with patch('snowflake.connector.connect') as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception):
                SnowflakeVectorStore(
                    connection=None,
                    embedding=MockSnowflakeEmbeddings(),
                    table_name="test_table"
                )
    
    def test_invalid_vector_dimensions(self):
        """Test handling of invalid vector dimensions."""
        connection = Mock(spec=SnowflakeConnection)
        
        with pytest.raises(ValueError):
            SnowflakeVectorStore(
                connection=connection,
                embedding=MockSnowflakeEmbeddings(dimension=5000),  # Exceeds Snowflake limit
                table_name="test_table",
                dimension=5000
            )
    
    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        connection = Mock(spec=SnowflakeConnection)
        mock_cursor = Mock()
        connection.cursor.return_value = mock_cursor
        
        store = SnowflakeVectorStore(
            connection=connection,
            embedding=MockSnowflakeEmbeddings(),
            table_name="test_table"
        )
        
        empty_docs = [Document(page_content="")]
        ids = store.add_documents(empty_docs)
        assert len(ids) == 1
    
    def test_malformed_metadata_handling(self):
        """Test handling of malformed metadata."""
        connection = Mock(spec=SnowflakeConnection)
        mock_cursor = Mock()
        connection.cursor.return_value = mock_cursor
        
        store = SnowflakeVectorStore(
            connection=connection,
            embedding=MockSnowflakeEmbeddings(),
            table_name="test_table"
        )
        
        malformed_doc = Document(
            page_content="Test document",
            metadata={"invalid": object()}  # Non-serializable object
        )
        
        with pytest.raises(Exception):
            store.add_documents([malformed_doc])
    
    def test_query_timeout_handling(self):
        """Test handling of query timeouts."""
        connection = Mock(spec=SnowflakeConnection)
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = Exception("Query timeout")
        connection.cursor.return_value = mock_cursor
        
        store = SnowflakeVectorStore(
            connection=connection,
            embedding=MockSnowflakeEmbeddings(),
            table_name="test_table"
        )
        
        with pytest.raises(Exception):
            store.similarity_search("test query")


class TestSnowflakeVectorStoreScenarios:
    """Real-world scenario tests for task management applications."""
    
    def test_daily_task_planning_scenario(self):
        """Test daily task planning with AI suggestions."""
        embeddings = MockSnowflakeEmbeddings(dimension=768)
        connection = Mock(spec=SnowflakeConnection)
        mock_cursor = Mock()
        connection.cursor.return_value = mock_cursor
        
        store = SnowflakeVectorStore(
            connection=connection,
            embedding=embeddings,
            table_name="daily_tasks"
        )
        
        historical_tasks = [
            Document(
                page_content="Review pull requests for authentication module",
                metadata={"completed": True, "duration": 30, "priority": "high"}
            ),
            Document(
                page_content="Write unit tests for login functionality",
                metadata={"completed": True, "duration": 45, "priority": "medium"}
            ),
            Document(
                page_content="Update API documentation for user endpoints",
                metadata={"completed": True, "duration": 60, "priority": "low"}
            )
        ]
        
        store.add_documents(historical_tasks)
        
        today_context = "I need to work on authentication features today"
        suggested_tasks = store.similarity_search(today_context, k=3)
        
        assert len(suggested_tasks) > 0
        assert any("auth" in task.page_content.lower() for task in suggested_tasks)
    
    def test_project_planning_scenario(self):
        """Test project planning with historical data analysis."""
        embeddings = MockSnowflakeEmbeddings(dimension=768)
        connection = Mock(spec=SnowflakeConnection)
        mock_cursor = Mock()
        connection.cursor.return_value = mock_cursor
        
        store = SnowflakeVectorStore(
            connection=connection,
            embedding=embeddings,
            table_name="project_history"
        )
        
        project_templates = [
            Document(
                page_content="Mobile app development: Setup development environment, implement core features, testing, deployment",
                metadata={"project_type": "mobile", "estimated_duration": 120, "team_size": 4}
            ),
            Document(
                page_content="Web application: Design UI/UX, implement backend API, frontend development, integration testing",
                metadata={"project_type": "web", "estimated_duration": 90, "team_size": 3}
            ),
            Document(
                page_content="Data analytics platform: Data pipeline setup, analytics engine, dashboard creation, reporting",
                metadata={"project_type": "analytics", "estimated_duration": 150, "team_size": 5}
            )
        ]
        
        store.add_documents(project_templates)
        
        new_project = "Build a mobile e-commerce application with payment integration"
        similar_projects = store.similarity_search(new_project, k=2)
        
        assert len(similar_projects) > 0
        mobile_project = any("mobile" in proj.metadata.get("project_type", "") for proj in similar_projects)
        assert mobile_project
    
    def test_knowledge_management_scenario(self):
        """Test knowledge management and learning from completed tasks."""
        embeddings = MockSnowflakeEmbeddings(dimension=768)
        connection = Mock(spec=SnowflakeConnection)
        mock_cursor = Mock()
        connection.cursor.return_value = mock_cursor
        
        store = SnowflakeVectorStore(
            connection=connection,
            embedding=embeddings,
            table_name="knowledge_base"
        )
        
        knowledge_docs = [
            Document(
                page_content="Authentication implementation: Use JWT tokens, implement refresh mechanism, handle edge cases",
                metadata={"category": "best_practices", "domain": "security", "difficulty": "intermediate"}
            ),
            Document(
                page_content="Database optimization: Use proper indexing, query optimization, connection pooling",
                metadata={"category": "best_practices", "domain": "database", "difficulty": "advanced"}
            ),
            Document(
                page_content="API design principles: RESTful endpoints, proper HTTP status codes, error handling",
                metadata={"category": "guidelines", "domain": "api", "difficulty": "beginner"}
            )
        ]
        
        store.add_documents(knowledge_docs)
        
        question = "How should I implement user authentication securely?"
        relevant_knowledge = store.similarity_search(question, k=2)
        
        assert len(relevant_knowledge) > 0
        auth_related = any("auth" in doc.page_content.lower() for doc in relevant_knowledge)
        assert auth_related


if __name__ == "__main__":
    pytest.main([__file__, "-v"])