# Snowflake Vector Store Test Suite for Task Management Applications

This repository contains a comprehensive test suite for Snowflake vector store integration in productivity task management applications. The tests validate vector similarity search capabilities, CRUD operations, performance characteristics, and task-specific use cases.

## Overview

The test suite is designed to validate:
- Core vector store functionality using Snowflake's native vector support
- Task management specific use cases (similarity search, recommendations, categorization)
- Performance and scalability under realistic workloads
- Integration with the LangChain ecosystem
- Error handling and edge cases
- Security and compliance requirements

## Features

### Snowflake Vector Capabilities Tested
- `VECTOR(TYPE, DIMENSION)` data type support (INT/FLOAT, up to 4096 dimensions)
- Vector similarity functions: `VECTOR_COSINE_SIMILARITY`, `VECTOR_INNER_PRODUCT`, `VECTOR_L1_DISTANCE`, `VECTOR_L2_DISTANCE`
- Text embedding functions: `EMBED_TEXT_768`, `EMBED_TEXT_1024`
- Snowflake Cortex AI/ML integration

### Task Management Use Cases
- **Task Similarity Search**: Find similar tasks based on content and context
- **Duplicate Detection**: Identify duplicate or near-duplicate tasks
- **Smart Recommendations**: Suggest relevant tasks based on user context
- **Project Categorization**: Automatically categorize tasks by project
- **Knowledge Management**: Search historical tasks and best practices
- **Sprint Planning**: Estimate effort based on similar completed tasks

## Installation

### Prerequisites
- Python 3.9 or higher
- Snowflake account with vector support enabled
- Access to Snowflake warehouse, database, and schema

### Setup Environment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd snowflake-vector-store-tests
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Snowflake connection**
   ```bash
   export SNOWFLAKE_ACCOUNT="your-account"
   export SNOWFLAKE_USER="your-username"
   export SNOWFLAKE_PASSWORD="your-password"
   export SNOWFLAKE_WAREHOUSE="your-warehouse"
   export SNOWFLAKE_DATABASE="your-database"
   export SNOWFLAKE_SCHEMA="your-schema"
   export SNOWFLAKE_ROLE="your-role"
   ```

   Or create a `.env` file:
   ```env
   SNOWFLAKE_ACCOUNT=your-account
   SNOWFLAKE_USER=your-username
   SNOWFLAKE_PASSWORD=your-password
   SNOWFLAKE_WAREHOUSE=your-warehouse
   SNOWFLAKE_DATABASE=your-database
   SNOWFLAKE_SCHEMA=your-schema
   SNOWFLAKE_ROLE=your-role
   ```

## Running Tests

### Quick Start
```bash
# Run all tests with mocked Snowflake connection
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration             # Integration tests only
pytest -m task_management         # Task management specific tests
pytest -m performance             # Performance tests
pytest -m scenario               # Real-world scenario tests
```

### Integration Tests (Requires Snowflake Connection)
```bash
# Enable integration tests
export SKIP_INTEGRATION_TESTS=false

# Run integration tests
pytest -m integration --run-integration

# Run with real Snowflake connection
pytest -m "integration and requires_snowflake" --run-integration
```

### Performance Tests
```bash
# Run performance tests (may take longer)
pytest -m performance --runslow

# Run performance tests only
pytest --performance-only --runslow
```

### Test Coverage
```bash
# Generate coverage report
pytest --cov=snowflake_vectorstore --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Test Structure

```
tests/
├── conftest.py                           # Shared fixtures and configuration
├── test_snowflake_vector_store.py        # Main test suite
├── unit/                                 # Unit tests
│   ├── test_connection.py
│   ├── test_vector_operations.py
│   ├── test_document_crud.py
│   └── test_similarity_search.py
├── integration/                          # Integration tests
│   ├── test_langchain_integration.py
│   ├── test_embedding_models.py
│   ├── test_retrieval_chains.py
│   └── test_task_workflows.py
├── performance/                          # Performance tests
│   ├── test_scalability.py
│   ├── test_concurrent_operations.py
│   └── test_benchmark_suite.py
├── security/                             # Security tests
│   ├── test_authentication.py
│   ├── test_authorization.py
│   └── test_data_privacy.py
└── scenarios/                            # Real-world scenarios
    ├── test_task_management.py
    ├── test_project_workflows.py
    └── test_knowledge_management.py
```

## Test Categories

### 1. Core Vector Store Operations
- Database connection and setup
- Document storage and retrieval (CRUD)
- Vector similarity search (cosine, inner product, L1, L2)
- Batch operations and performance

### 2. Task Management Use Cases
- Task similarity search and recommendations
- Duplicate task detection
- Project categorization and clustering
- User context and personalization

### 3. Performance & Scalability
- Large dataset handling (1K, 10K, 100K, 1M documents)
- Concurrent operation testing
- Memory usage monitoring
- Latency and throughput benchmarks

### 4. Integration Tests
- LangChain VectorStore interface compliance
- Embedding model integration
- Retrieval chain compatibility
- Async operation support

### 5. Error Handling & Edge Cases
- Connection failures and recovery
- Invalid data handling
- Snowflake-specific limitations
- Query timeout management

### 6. Security & Compliance
- Authentication and authorization
- Data privacy and encryption
- Audit logging
- GDPR compliance considerations

## Configuration

### Environment Variables
- `SNOWFLAKE_ACCOUNT`: Your Snowflake account identifier
- `SNOWFLAKE_USER`: Database user
- `SNOWFLAKE_PASSWORD`: Database password
- `SNOWFLAKE_WAREHOUSE`: Compute warehouse
- `SNOWFLAKE_DATABASE`: Database name
- `SNOWFLAKE_SCHEMA`: Schema name
- `SNOWFLAKE_ROLE`: User role
- `SKIP_INTEGRATION_TESTS`: Skip integration tests (default: true)

### Test Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.requires_snowflake`: Requires Snowflake connection
- `@pytest.mark.task_management`: Task management specific
- `@pytest.mark.scenario`: Real-world scenarios

## Performance Benchmarks

### Expected Performance Metrics
- **Search Latency**: < 100ms for 10K documents
- **Batch Insert**: > 1000 documents/second
- **Memory Usage**: < 512MB for 100K documents
- **Concurrent Users**: 100+ simultaneous operations
- **Scalability**: Support for 1M+ documents

### Benchmark Results
Run `pytest -m performance --benchmark-only` to generate detailed performance reports.

## Task Management Scenarios

### 1. Daily Task Planning
```python
# Example: Morning task planning with AI suggestions
user_context = "I'm a frontend developer working on authentication"
suggestions = vector_store.similarity_search(user_context, k=5)
```

### 2. Sprint Planning
```python
# Example: Sprint capacity planning
sprint_tasks = vector_store.similarity_search(
    "authentication frontend tasks",
    k=10,
    filter={"priority": "high"}
)
```

### 3. Knowledge Discovery
```python
# Example: Finding best practices
question = "How to implement secure authentication?"
knowledge = vector_store.similarity_search(question, k=3)
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify Snowflake credentials
   - Check network connectivity
   - Ensure warehouse is running

2. **Vector Dimension Errors**
   - Snowflake supports up to 4096 dimensions
   - Ensure embedding model matches vector dimension

3. **Performance Issues**
   - Check vector indexing
   - Monitor query complexity
   - Optimize batch sizes

### Debug Mode
```bash
pytest -v --tb=long --log-cli-level=DEBUG
```

## Contributing

### Running Tests Before Commit
```bash
# Run all unit tests
pytest -m unit

# Run code quality checks
black .
isort .
flake8 .
mypy .

# Run test coverage
pytest --cov=snowflake_vectorstore --cov-fail-under=80
```

### Adding New Tests
1. Follow existing test patterns
2. Use appropriate test markers
3. Include docstrings with clear descriptions
4. Add fixtures for reusable test data
5. Ensure tests are deterministic

## Documentation

### API Documentation
```bash
# Generate API documentation
sphinx-build -b html docs docs/_build/html
```

### Test Reports
```bash
# Generate test report
pytest --html=reports/report.html --self-contained-html
```

## License

This test suite is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review test output and logs
3. File an issue with detailed reproduction steps
4. Include environment information and test configuration

## Roadmap

### Future Enhancements
- [ ] Support for additional embedding models
- [ ] Multi-language task support
- [ ] Advanced security testing
- [ ] Performance optimization recommendations
- [ ] Integration with task management platforms
- [ ] Real-time vector indexing
- [ ] Distributed testing capabilities

### Known Limitations
- Snowflake vector dimension limit (4096)
- Server-side binding restrictions
- Limited Snowpark API support for some functions
- Integration test dependency on Snowflake connection

## References

- [Snowflake Vector Functions Documentation](https://docs.snowflake.com/en/sql-reference/functions-vector)
- [LangChain VectorStore Interface](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [Snowflake Cortex AI Functions](https://docs.snowflake.com/en/user-guide/snowflake-cortex)
- [Vector Embeddings Guide](https://docs.snowflake.com/en/user-guide/snowflake-cortex/vector-embeddings)