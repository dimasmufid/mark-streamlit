#!/usr/bin/env python
import os
import sys
import time
import dotenv
from sqlalchemy import create_engine, Column, String, DateTime, MetaData, Table, inspect, text
from sqlalchemy.sql import func
import psycopg2
from psycopg2 import sql
from urllib.parse import urlparse

# Load environment variables
dotenv.load_dotenv()

# Get database connection details
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is not set.")
    print("Please set it in your .env file, for example:")
    print("DATABASE_URL=postgresql://username:password@localhost:5432/chat_app")
    sys.exit(1)

# Parse the DATABASE_URL to get components
parsed_url = urlparse(DATABASE_URL)
db_user = parsed_url.username
db_password = parsed_url.password
db_host = parsed_url.hostname
db_port = parsed_url.port or 5432
db_name = parsed_url.path[1:]  # Remove leading slash

print(f"Database connection info:")
print(f"  Host: {db_host}")
print(f"  Port: {db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print("")

# First check if we can connect to PostgreSQL at all
print("Step 1: Testing connection to PostgreSQL server...")
try:
    # Connect to the 'postgres' database to check server availability
    conn_string = f"host={db_host} port={db_port} user={db_user} password={db_password} dbname=postgres"
    conn = psycopg2.connect(conn_string)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]
    print(f"✅ Successfully connected to PostgreSQL server: {version}")
except Exception as e:
    print(f"❌ Failed to connect to PostgreSQL server: {str(e)}")
    print("\nPossible solutions:")
    print("1. Make sure PostgreSQL is running on the specified host and port")
    print("2. Check that the username and password are correct")
    print("3. Ensure the PostgreSQL server allows connections from this host")
    sys.exit(1)

# Check if the specified database exists
print(f"\nStep 2: Checking if database '{db_name}' exists...")
try:
    cursor.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), (db_name,))
    db_exists = cursor.fetchone() is not None
    
    if db_exists:
        print(f"✅ Database '{db_name}' already exists")
    else:
        print(f"Database '{db_name}' doesn't exist, creating it...")
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        print(f"✅ Database '{db_name}' created successfully")
    
    # Close connection to 'postgres' database
    conn.close()
except Exception as e:
    print(f"❌ Error checking/creating database: {str(e)}")
    print("\nPossible solutions:")
    print("1. Make sure your database user has permission to create databases")
    print("2. Create the database manually and retry the migration")
    conn.close()
    sys.exit(1)

# Now connect to the actual database and create tables
print(f"\nStep 3: Connecting to '{db_name}' database and creating tables...")
try:
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()
    
    # Define documents table
    documents_table = Table(
        'documents', metadata,
        Column('id', String, primary_key=True),
        Column('name', String, nullable=False),
        Column('type', String, nullable=False),
        Column('content_preview', String),
        Column('minio_path', String, nullable=False),
        Column('upload_date', DateTime, server_default=func.now())
    )
    
    # Create tables
    metadata.create_all(engine)
    print("✅ Tables created successfully")
    
    # Check what tables were created
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables in database: {', '.join(tables)}")
    
    if 'documents' in tables:
        columns = [col['name'] for col in inspector.get_columns('documents')]
        print(f"Columns in 'documents' table: {', '.join(columns)}")
    
except Exception as e:
    print(f"❌ Error creating tables: {str(e)}")
    print("\nPossible solutions:")
    print("1. Check that your DATABASE_URL is correctly formatted")
    print("2. Ensure your database user has permission to create tables")
    sys.exit(1)

# Check MinIO configuration
print("\nStep 4: Checking MinIO configuration...")
minio_endpoint = os.environ.get("MINIO_ENDPOINT")
minio_port = os.environ.get("MINIO_PORT")
minio_access_key = os.environ.get("MINIO_ACCESS_KEY")
minio_secret_key = os.environ.get("MINIO_SECRET_KEY")
minio_bucket_name = os.environ.get("MINIO_BUCKET_NAME")
minio_use_ssl = os.environ.get("MINIO_USE_SSL", "false").lower() == "true"

if not all([minio_endpoint, minio_port, minio_access_key, minio_secret_key, minio_bucket_name]):
    print("❌ Some MinIO configuration values are missing:")
    print(f"  MINIO_ENDPOINT: {'✅' if minio_endpoint else '❌'}")
    print(f"  MINIO_PORT: {'✅' if minio_port else '❌'}")
    print(f"  MINIO_ACCESS_KEY: {'✅' if minio_access_key else '❌'}")
    print(f"  MINIO_SECRET_KEY: {'✅' if minio_secret_key else '❌'}")
    print(f"  MINIO_BUCKET_NAME: {'✅' if minio_bucket_name else '❌'}")
else:
    print("✅ All MinIO environment variables are set")

print("\n=== Migration Complete ===")
print("You can now run the Streamlit app. If you still encounter issues,")
print("check the error messages and make sure the connection details are correct.") 