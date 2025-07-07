"""
Shared test configuration and fixtures for Snowflake Vector Store tests.
"""

import os
import pytest
from typing import Dict, Any, Optional, Generator
from unittest.mock import Mock, patch

try:
    import snowflake.connector
    from snowflake.connector import SnowflakeConnection
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance tests"
    )
    config.addinivalue_line(
        "markers", "scenario: Real-world scenario tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "requires_snowflake: Tests requiring Snowflake connection"
    )
    config.addinivalue_line(
        "markers", "requires_embeddings: Tests requiring embedding models"
    )
    config.addinivalue_line(
        "markers", "task_management: Task management specific tests"
    )
    config.addinivalue_line(
        "markers", "security: Security and compliance tests"
    )
    config.addinivalue_line(
        "markers", "error_handling: Error handling tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests that require Snowflake
        if "snowflake" in item.name.lower() or "snowflake" in str(item.fspath).lower():
            item.add_marker(pytest.mark.requires_snowflake)
        
        # Mark performance tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Mark task management tests
        if "task" in item.name.lower() and "management" in item.name.lower():
            item.add_marker(pytest.mark.task_management)
        
        # Mark scenario tests
        if "scenario" in item.name.lower():
            item.add_marker(pytest.mark.scenario)
        
        # Mark error handling tests
        if "error" in item.name.lower() or "exception" in item.name.lower():
            item.add_marker(pytest.mark.error_handling)


@pytest.fixture(scope="session")
def snowflake_config() -> Dict[str, Any]:
    """Snowflake connection configuration."""
    return {
        "account": os.getenv("SNOWFLAKE_ACCOUNT", "test_account"),
        "user": os.getenv("SNOWFLAKE_USER", "test_user"),
        "password": os.getenv("SNOWFLAKE_PASSWORD", "test_password"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "test_warehouse"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "test_database"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "test_schema"),
        "role": os.getenv("SNOWFLAKE_ROLE", "test_role"),
    }


@pytest.fixture(scope="session")
def real_snowflake_connection(snowflake_config) -> Optional[SnowflakeConnection]:
    """Real Snowflake connection for integration tests."""
    if not SNOWFLAKE_AVAILABLE:
        pytest.skip("Snowflake connector not available")
    
    if os.getenv("SKIP_INTEGRATION_TESTS", "true").lower() == "true":
        pytest.skip("Integration tests skipped")
    
    try:
        conn = snowflake.connector.connect(**snowflake_config)
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"Could not connect to Snowflake: {e}")


@pytest.fixture
def mock_snowflake_connection():
    """Mock Snowflake connection for unit tests."""
    mock_conn = Mock(spec=SnowflakeConnection)
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@pytest.fixture
def sample_task_documents():
    """Sample task documents for testing."""
    if not LANGCHAIN_AVAILABLE:
        pytest.skip("LangChain not available")
    
    return [
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
        ),
        Document(
            page_content="Optimize database queries for user profile page",
            metadata={"project": "performance", "priority": "high", "assignee": "diana", "type": "optimization"}
        ),
        Document(
            page_content="Set up CI/CD pipeline for automated testing",
            metadata={"project": "devops", "priority": "medium", "assignee": "eve", "type": "infrastructure"}
        ),
        Document(
            page_content="Research new frontend framework for mobile app",
            metadata={"project": "mobile", "priority": "low", "assignee": "frank", "type": "research"}
        ),
        Document(
            page_content="Implement real-time notifications system",
            metadata={"project": "notifications", "priority": "high", "assignee": "grace", "type": "feature"}
        ),
        Document(
            page_content="Conduct security audit of API endpoints",
            metadata={"project": "security", "priority": "critical", "assignee": "henry", "type": "audit"}
        )
    ]


@pytest.fixture
def large_task_dataset():
    """Large dataset of task documents for performance testing."""
    if not LANGCHAIN_AVAILABLE:
        pytest.skip("LangChain not available")
    
    tasks = []
    projects = ["auth", "frontend", "backend", "mobile", "devops", "security", "ui", "api"]
    priorities = ["low", "medium", "high", "critical"]
    types = ["feature", "bug", "test", "documentation", "optimization", "research"]
    assignees = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry"]
    
    for i in range(1000):
        task = Document(
            page_content=f"Task {i}: Implement feature {i} for project optimization",
            metadata={
                "project": projects[i % len(projects)],
                "priority": priorities[i % len(priorities)],
                "type": types[i % len(types)],
                "assignee": assignees[i % len(assignees)],
                "task_id": i,
                "created_at": f"2024-01-{(i % 28) + 1:02d}",
                "estimated_hours": (i % 40) + 1
            }
        )
        tasks.append(task)
    
    return tasks


@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "batch_sizes": [100, 500, 1000, 5000],
        "search_k_values": [1, 5, 10, 50, 100],
        "concurrent_users": [1, 5, 10, 25, 50],
        "max_latency_ms": 100,
        "min_throughput_docs_per_sec": 1000,
        "max_memory_mb": 512,
        "timeout_seconds": 30
    }


@pytest.fixture
def embedding_models():
    """Different embedding model configurations for testing."""
    return {
        "small": {"dimension": 384, "model_name": "small-embedding"},
        "medium": {"dimension": 768, "model_name": "medium-embedding"},
        "large": {"dimension": 1024, "model_name": "large-embedding"},
        "extra_large": {"dimension": 1536, "model_name": "xl-embedding"}
    }


@pytest.fixture
def test_queries():
    """Test queries for different scenarios."""
    return {
        "authentication": [
            "user login authentication",
            "OAuth2 implementation",
            "session management",
            "password reset functionality",
            "two-factor authentication"
        ],
        "frontend": [
            "UI component development",
            "responsive design implementation",
            "user interface mockups",
            "frontend framework integration",
            "mobile-first design"
        ],
        "backend": [
            "API endpoint implementation",
            "database schema design",
            "server-side logic",
            "microservices architecture",
            "data processing pipeline"
        ],
        "performance": [
            "query optimization",
            "caching strategy",
            "load balancing",
            "performance monitoring",
            "scalability improvements"
        ],
        "security": [
            "vulnerability assessment",
            "security audit",
            "data encryption",
            "access control",
            "compliance requirements"
        ]
    }


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    class MockEmbeddings:
        def __init__(self, dimension=768):
            self.dimension = dimension
        
        def embed_documents(self, texts):
            return [[0.1 * i for i in range(self.dimension)] for _ in texts]
        
        def embed_query(self, text):
            return [0.1 * i for i in range(self.dimension)]
    
    return MockEmbeddings()


@pytest.fixture
def task_management_scenarios():
    """Real-world task management scenarios."""
    return {
        "daily_planning": {
            "user_context": "I'm a frontend developer working on user authentication",
            "available_time": 8,  # hours
            "preferred_priority": "high",
            "expected_task_count": 3
        },
        "sprint_planning": {
            "sprint_capacity": 40,  # story points
            "team_size": 5,
            "sprint_duration": 14,  # days
            "focus_areas": ["authentication", "frontend", "testing"]
        },
        "project_estimation": {
            "project_description": "Build a mobile e-commerce application",
            "team_experience": "intermediate",
            "deadline": "3 months",
            "expected_features": ["user auth", "product catalog", "shopping cart", "payment"]
        },
        "knowledge_discovery": {
            "problem": "How to implement secure authentication?",
            "difficulty": "intermediate",
            "domain": "security",
            "expected_solution_count": 5
        },
        "duplicate_detection": {
            "similarity_threshold": 0.8,
            "time_window": 30,  # days
            "check_metadata": ["project", "assignee"]
        }
    }


# Pytest plugins and hooks
def pytest_runtest_setup(item):
    """Setup for each test."""
    if item.get_closest_marker("slow") and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
    
    if item.get_closest_marker("requires_snowflake"):
        if not SNOWFLAKE_AVAILABLE:
            pytest.skip("Snowflake connector not available")
    
    if item.get_closest_marker("requires_embeddings"):
        if not LANGCHAIN_AVAILABLE:
            pytest.skip("LangChain not available")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--run-integration", action="store_true", default=False, 
        help="run integration tests (requires Snowflake connection)"
    )
    parser.addoption(
        "--performance-only", action="store_true", default=False,
        help="run only performance tests"
    )
    parser.addoption(
        "--scenario-only", action="store_true", default=False,
        help="run only scenario tests"
    )


def pytest_configure_node(node):
    """Configure test node for distributed testing."""
    node.workerinput["snowflake_config"] = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT", "test_account"),
        "user": os.getenv("SNOWFLAKE_USER", "test_user"),
        "password": os.getenv("SNOWFLAKE_PASSWORD", "test_password"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "test_warehouse"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "test_database"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "test_schema"),
    }