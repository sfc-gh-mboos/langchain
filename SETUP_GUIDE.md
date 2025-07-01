# Snowflake Vector Store Setup Guide

This guide provides step-by-step instructions for setting up the testing environment for Snowflake vector store support in a productivity application.

## Prerequisites

### Snowflake Account Requirements

1. **Snowflake Account**: You need a Snowflake account with the following features:
   - Cortex AI functions enabled
   - VECTOR data type support (requires Snowflake version 7.40+)
   - Appropriate roles and privileges

2. **Required Privileges**:
   ```sql
   -- Grant the CORTEX_USER database role
   GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE <your_role>;
   
   -- Grant necessary privileges for vector operations
   GRANT USAGE ON WAREHOUSE <warehouse_name> TO ROLE <your_role>;
   GRANT USAGE ON DATABASE <database_name> TO ROLE <your_role>;
   GRANT USAGE ON SCHEMA <schema_name> TO ROLE <your_role>;
   GRANT CREATE TABLE ON SCHEMA <schema_name> TO ROLE <your_role>;
   ```

3. **Warehouse Configuration**:
   - Recommended: SMALL to MEDIUM warehouse for testing
   - Dedicated warehouse for vector operations to avoid interference

### Python Environment Requirements

- Python 3.9 or higher
- Virtual environment (recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository_url>
cd snowflake-vector-store-tests
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```bash
# Snowflake Connection Parameters
SNOWFLAKE_ACCOUNT=your_account.region.cloud
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_ROLE=your_role

# Optional: Specific test configurations
TEST_TABLE_PREFIX=test_vectors_
TEST_BATCH_SIZE=100
TEST_TIMEOUT=300
```

### 5. Verify Snowflake Connection

Run the connection test:

```python
import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv()

# Test connection
try:
    conn = snowflake.connector.connect(
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        role=os.getenv('SNOWFLAKE_ROLE')
    )
    
    cursor = conn.cursor()
    cursor.execute("SELECT CURRENT_VERSION()")
    version = cursor.fetchone()[0]
    print(f"✅ Connected to Snowflake version: {version}")
    
    # Test VECTOR data type support
    cursor.execute("SELECT [1,2,3]::VECTOR(FLOAT,3) as test_vector")
    result = cursor.fetchone()
    print(f"✅ VECTOR data type supported: {result[0]}")
    
    # Test Cortex AI functions
    cursor.execute("SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m-v1.5', 'test') as embedding")
    embedding = cursor.fetchone()
    print(f"✅ Cortex AI embedding function working: {len(embedding[0])} dimensions")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
finally:
    if 'conn' in locals():
        conn.close()
```

## Snowflake Setup

### 1. Create Test Database and Schema

```sql
-- Create test database
CREATE DATABASE IF NOT EXISTS VECTOR_STORE_TESTS;
USE DATABASE VECTOR_STORE_TESTS;

-- Create test schema
CREATE SCHEMA IF NOT EXISTS TESTS;
USE SCHEMA TESTS;

-- Create test warehouse (if needed)
CREATE WAREHOUSE IF NOT EXISTS VECTOR_TEST_WH 
WITH WAREHOUSE_SIZE = 'SMALL' 
AUTO_SUSPEND = 60 
AUTO_RESUME = TRUE;
```

### 2. Enable Change Tracking (for Cortex Search)

```sql
-- Enable change tracking on tables that will be used with Cortex Search
ALTER TABLE <table_name> SET CHANGE_TRACKING = TRUE;
```

### 3. Test Vector Operations

```sql
-- Test basic vector operations
CREATE OR REPLACE TABLE test_vectors (
    id VARCHAR PRIMARY KEY,
    content TEXT,
    vector VECTOR(FLOAT, 768)
);

-- Insert test data with embeddings
INSERT INTO test_vectors (id, content, vector)
SELECT 
    'test_1',
    'This is a test document',
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m-v1.5', 'This is a test document');

-- Test similarity search
SELECT 
    id,
    content,
    VECTOR_COSINE_SIMILARITY(
        vector, 
        SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m-v1.5', 'test document')
    ) as similarity
FROM test_vectors
ORDER BY similarity DESC;

-- Clean up
DROP TABLE test_vectors;
```

## Running Tests

### 1. Unit Tests

Run the basic unit tests:

```bash
pytest test_snowflake_vectorstore.py::TestSnowflakeVectorStore -v
```

### 2. Integration Tests

Run integration tests with real Snowflake connection:

```bash
# Set environment variables
export SNOWFLAKE_ACCOUNT=your_account
export SNOWFLAKE_USER=your_user
# ... other variables

# Run integration tests
pytest test_snowflake_vectorstore.py::TestSnowflakeVectorStore::test_real_snowflake_connection -v
```

### 3. Performance Tests

Run performance benchmarks:

```bash
pytest test_snowflake_vectorstore.py::TestSnowflakeVectorStorePerformance -v --tb=short
```

### 4. Full Test Suite

Run all tests:

```bash
pytest test_snowflake_vectorstore.py -v --tb=short
```

## Development Workflow

### 1. Code Formatting

```bash
# Format code
black test_snowflake_vectorstore.py
isort test_snowflake_vectorstore.py
```

### 2. Linting

```bash
# Check code quality
flake8 test_snowflake_vectorstore.py
mypy test_snowflake_vectorstore.py
```

### 3. Testing with Coverage

```bash
pip install pytest-cov
pytest test_snowflake_vectorstore.py --cov=. --cov-report=html
```

## Troubleshooting

### Common Issues

1. **Connection Timeout**:
   ```
   Error: Connection timeout
   Solution: Check network connectivity and warehouse status
   ```

2. **VECTOR Data Type Not Supported**:
   ```
   Error: SQL compilation error: Unknown data type 'VECTOR'
   Solution: Ensure Snowflake account supports VECTOR data type (version 7.40+)
   ```

3. **Cortex AI Functions Not Available**:
   ```
   Error: Function SNOWFLAKE.CORTEX.EMBED_TEXT_768 does not exist
   Solution: Ensure CORTEX_USER role is granted and functions are available in your region
   ```

4. **Insufficient Privileges**:
   ```
   Error: Insufficient privileges to operate on table
   Solution: Grant necessary table and schema privileges
   ```

### Performance Optimization

1. **Warehouse Sizing**:
   - Start with SMALL warehouse for development
   - Scale up to MEDIUM/LARGE for performance testing
   - Use dedicated warehouse for vector operations

2. **Batch Operations**:
   - Process documents in batches of 100-1000
   - Use bulk insert operations when possible
   - Monitor memory usage during large operations

3. **Cost Management**:
   - Set auto-suspend on warehouses (60 seconds recommended)
   - Monitor credit consumption
   - Use appropriate embedding models for your use case

### Monitoring and Logging

1. **Query History**:
   ```sql
   -- Monitor vector operations
   SELECT 
       query_text,
       execution_time,
       warehouse_name,
       credits_used_cloud_services
   FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
   WHERE query_text ILIKE '%VECTOR%'
   ORDER BY start_time DESC
   LIMIT 10;
   ```

2. **Cortex Function Usage**:
   ```sql
   -- Monitor Cortex AI function usage
   SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
   WHERE start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
   ORDER BY start_time DESC;
   ```

## Next Steps

1. **Implement Core Functionality**: Start with basic vector store operations
2. **Add Snowflake-Specific Features**: Implement Cortex Search integration
3. **Performance Testing**: Benchmark with large datasets
4. **Documentation**: Create comprehensive API documentation
5. **Production Deployment**: Set up production-ready configuration

## Resources

- [Snowflake Vector Data Types Documentation](https://docs.snowflake.com/en/sql-reference/data-types-vector)
- [Snowflake Cortex AI Functions](https://docs.snowflake.com/en/user-guide/snowflake-cortex/vector-embeddings)
- [LangChain Vector Store Documentation](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [Snowflake Python Connector Documentation](https://docs.snowflake.com/en/developer-guide/python-connector/python-connector)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Snowflake documentation
3. Check LangChain community resources
4. File issues in the project repository