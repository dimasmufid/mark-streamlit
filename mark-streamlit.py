import os
import streamlit as st
from openai import OpenAI
import dotenv
import uuid
import json
import io
import pandas as pd
from datetime import datetime
import PyPDF2
import docx
import tempfile
import sys
from io import StringIO
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from minio import Minio
from minio.error import S3Error
from sqlalchemy import create_engine, Column, String, DateTime, MetaData, Table, select, insert, delete, update, Integer, Text, ForeignKey
from sqlalchemy.sql import func, text
import psycopg2
import time
import re

dotenv.load_dotenv()

# Configure the page
st.set_page_config(page_title="ChatGPT System Prompt Experimenter", layout="wide")

# Default System Prompt for Mark, the AI Business Analyst
DEFAULT_SYSTEM_PROMPT = """You are Mark, an AI Business Analyst. Your primary objective is to assist users in analyzing business problems, identifying opportunities, creating business reports, and proposing data-driven solutions to improve efficiency and achieve business objectives. You are analytical, objective, thorough, and possess strong communication skills, designed to process information, identify patterns, and present findings in a clear and structured manner.

Your capabilities include:
- Assisting in eliciting, documenting, and analyzing business requirements from user input.
- Analyzing provided data to identify trends, patterns, and key insights relevant to business performance.
- Helping model and analyze business processes to identify inefficiencies and areas for improvement.
- Working with users to clearly define business problems and the scope of potential solutions.
- Suggesting potential solutions and strategies based on analysis and best practices.
- Aiding in creating outlines and content for business documents, reports, and specifications.
- Providing information and insights on industry trends and best practices based on available knowledge.
- Identifying when additional data or clarification is needed for a complete and accurate analysis and clearly requesting that information from the user.

You can only analyze data and information explicitly provided by the user within this application. You do not have access to external websites or real-time external databases unless specifically integrated and stated. You are an assistant providing analysis and suggestions; you cannot make business decisions. You are not a financial advisor, legal counsel, or a substitute for human expertise and judgment.

If information is ambiguous or insufficient, you will ask clarifying questions to the user. If you determine that more data is necessary to perform a thorough analysis or create a comprehensive report, you will clearly explain what data is needed and why, and ask the user to provide it.

Maintain a professional, objective, and helpful tone. Communicate clearly, concisely, and avoid jargon where possible, or explain it when necessary. Be direct and objective in your analysis and feedback, stating clearly if something appears inefficient, ineffective, or problematic based on the data. Frame constructive criticism in a way that helps the user understand the issues and work towards solutions. Acknowledge the user's business challenges and express a focus on helping them achieve their goals, demonstrating a positive and supportive attitude towards their success.

Avoid making assumptions or exhibiting biases in your analysis and recommendations. Handle all user-provided information with implied confidentiality. Adhere to standard safety guidelines and avoid generating harmful content.
"""

# Code generation prompt
CODE_GENERATION_PROMPT = """You are a Python data analysis expert. Generate Python code to analyze the data in response to the user's query.

Your code will be executed to analyze data files, and only the results will be returned to the user.

Guidelines for code generation:
1. ALWAYS include proper error handling
2. For CSV/Excel files, use pandas for data analysis
3. Generate insightful visualizations using matplotlib or seaborn when appropriate
4. Include descriptive comments
5. Make sure your code is efficient for large datasets
6. ONLY include code that is necessary to answer the user's question - be focused and specific
7. Do NOT use external APIs or install packages
8. Print the results clearly for the user to understand
9. Use standard Python libraries (pandas, numpy, matplotlib, seaborn)
10. If analyzing text files, use appropriate text processing techniques

IMPORTANT: Do NOT return markdown formatting (like ```python). Return ONLY pure Python code.

Available data files:
{available_files}

The following variables will be available to your code:
- file_paths: A dictionary mapping filenames to their full paths
- get_file_path(filename): A function to get the path of a file by name
- For each file, a variable named [filename]_path (with special characters replaced by underscores)

Example: For a file named "data.csv", you can use:
- file_paths["data.csv"] or 
- get_file_path("data.csv") or
- data_path

The user's question is:
{user_question}

Respond ONLY with executable Python code without any additional text or explanations.
"""

# MinIO and PostgreSQL Configuration
class StorageConfig:
    def __init__(self):
        # MinIO configuration
        self.minio_endpoint = os.environ.get("MINIO_ENDPOINT", "localhost")
        self.minio_port = os.environ.get("MINIO_PORT", "9000")
        self.minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        self.minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        self.minio_bucket_name = os.environ.get("MINIO_BUCKET_NAME", "documents")
        self.minio_use_ssl = os.environ.get("MINIO_USE_SSL", "false").lower() == "true"
        
        # Full MinIO endpoint with port
        self.minio_endpoint_with_port = f"{self.minio_endpoint}:{self.minio_port}"
        
        # PostgreSQL configuration
        self.database_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/chat_app")
        
        # Initialize connections
        self._init_minio()
        self._init_db()
    
    def _init_minio(self):
        """Initialize MinIO client and ensure bucket exists"""
        try:
            self.minio_client = Minio(
                self.minio_endpoint_with_port,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=self.minio_use_ssl
            )
            
            # Create bucket if it doesn't exist
            if not self.minio_client.bucket_exists(self.minio_bucket_name):
                self.minio_client.make_bucket(self.minio_bucket_name)
                st.success(f"Bucket {self.minio_bucket_name} created successfully")
            
            self.minio_ready = True
        except Exception as e:
            st.error(f"Error initializing MinIO: {str(e)}")
            self.minio_ready = False
    
    def _init_db(self):
        """Initialize PostgreSQL connection and ensure tables exist"""
        try:
            self.engine = create_engine(self.database_url)
            self.metadata = MetaData()
            
            # Define documents table
            self.documents_table = Table(
                'documents', self.metadata,
                Column('id', String, primary_key=True),
                Column('name', String, nullable=False),
                Column('type', String, nullable=False),
                Column('content_preview', String),
                Column('minio_path', String, nullable=False),
                Column('upload_date', DateTime, server_default=func.now())
            )
            
            # Define conversations table
            self.conversations_table = Table(
                'conversations', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('title', String),
                Column('system_prompt', Text),
                Column('created_at', DateTime, server_default=func.now()),
                Column('updated_at', DateTime, server_default=func.now(), onupdate=func.now())
            )
            
            # Define messages table
            self.messages_table = Table(
                'messages', self.metadata,
                Column('id', Integer, primary_key=True),
                Column('conversation_id', Integer, ForeignKey('conversations.id', ondelete='CASCADE')),
                Column('role', String, nullable=False),
                Column('content', Text, nullable=False),
                Column('created_at', DateTime, server_default=func.now())
            )
            
            # Create tables if they don't exist
            self.metadata.create_all(self.engine)
            self.db_ready = True
        except Exception as e:
            st.error(f"Error initializing PostgreSQL: {str(e)}")
            self.db_ready = False
    
    def upload_file(self, file_obj, filename, file_id, file_extension, preview_text):
        """Upload a file to MinIO and save metadata to PostgreSQL"""
        if not self.minio_ready or not self.db_ready:
            return None, "Storage services are not available"
        
        try:
            # Reset file pointer to beginning
            file_obj.seek(0)
            
            # Generate MinIO path
            minio_path = f"{file_id}.{file_extension}"
            
            # Upload to MinIO
            self.minio_client.put_object(
                bucket_name=self.minio_bucket_name,
                object_name=minio_path,
                data=file_obj,
                length=file_obj.getbuffer().nbytes,
                content_type=f"application/{file_extension}"
            )
            
            # Save metadata to PostgreSQL
            with self.engine.begin() as conn:
                stmt = insert(self.documents_table).values(
                    id=file_id,
                    name=filename,
                    type=file_extension,
                    content_preview=preview_text[:5000] if preview_text else "",
                    minio_path=minio_path,
                    upload_date=datetime.now()
                )
                conn.execute(stmt)
            
            return file_id, None
        except Exception as e:
            return None, f"Error uploading file: {str(e)}"
    
    def delete_file(self, file_id):
        """Delete a file from MinIO and remove metadata from PostgreSQL"""
        if not self.minio_ready or not self.db_ready:
            return False, "Storage services are not available"
        
        try:
            # Get file metadata from PostgreSQL
            with self.engine.begin() as conn:
                stmt = select(self.documents_table.c.minio_path).where(self.documents_table.c.id == file_id)
                result = conn.execute(stmt).fetchone()
                
                if result:
                    minio_path = result[0]
                    
                    # Delete from MinIO
                    self.minio_client.remove_object(self.minio_bucket_name, minio_path)
                    
                    # Delete from PostgreSQL
                    stmt = delete(self.documents_table).where(self.documents_table.c.id == file_id)
                    conn.execute(stmt)
                    
                    return True, None
                else:
                    return False, "File not found in database"
        except Exception as e:
            return False, f"Error deleting file: {str(e)}"
    
    def get_all_documents(self):
        """Get all document metadata from PostgreSQL"""
        if not self.db_ready:
            return {}
        
        try:
            documents = {}
            with self.engine.begin() as conn:
                stmt = select(self.documents_table)
                results = conn.execute(stmt).fetchall()
                
                for row in results:
                    documents[row.id] = {
                        "id": row.id,
                        "name": row.name,
                        "type": row.type,
                        "preview": row.content_preview,
                        "minio_path": row.minio_path,
                        "upload_date": row.upload_date.strftime("%Y-%m-%d %H:%M:%S") if row.upload_date else None
                    }
            
            return documents
        except Exception as e:
            st.error(f"Error fetching documents: {str(e)}")
            return {}
    
    def get_document_content(self, document_id, document_metadata):
        """Get document content from MinIO"""
        if not self.minio_ready:
            return "Storage service is not available"
        
        try:
            # Get file from MinIO
            response = self.minio_client.get_object(
                bucket_name=self.minio_bucket_name,
                object_name=document_metadata["minio_path"]
            )
            
            # Read data according to file type
            file_data = response.read()
            file_extension = document_metadata["type"]
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name
            
            return temp_path
        except Exception as e:
            return f"Error retrieving document: {str(e)}"
    
    # Conversation Management Methods
    def get_all_conversations(self):
        """Get all conversations from the database"""
        if not self.db_ready:
            return {}
        
        try:
            conversations = {}
            with self.engine.begin() as conn:
                stmt = select(self.conversations_table)
                results = conn.execute(stmt).fetchall()
                
                for row in results:
                    conversations[row.id] = {
                        "id": row.id,
                        "title": row.title or f"Conversation {row.id}",
                        "system_prompt": row.system_prompt,
                        "created_at": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else None,
                        "updated_at": row.updated_at.strftime("%Y-%m-%d %H:%M:%S") if row.updated_at else None,
                        "messages": []
                    }
            
            return conversations
        except Exception as e:
            st.error(f"Error fetching conversations: {str(e)}")
            return {}
    
    def get_conversation_messages(self, conversation_id):
        """Get all messages for a specific conversation"""
        if not self.db_ready:
            return []
        
        try:
            messages = []
            with self.engine.begin() as conn:
                stmt = select(self.messages_table).where(
                    self.messages_table.c.conversation_id == conversation_id
                ).order_by(self.messages_table.c.id)
                
                results = conn.execute(stmt).fetchall()
                
                for row in results:
                    messages.append({
                        "role": row.role,
                        "content": row.content
                    })
            
            return messages
        except Exception as e:
            st.error(f"Error fetching messages: {str(e)}")
            return []
    
    def create_conversation(self, system_prompt=None, title=None):
        """Create a new conversation"""
        if not self.db_ready:
            return None, "Database is not available"
        
        try:
            with self.engine.begin() as conn:
                stmt = insert(self.conversations_table).values(
                    system_prompt=system_prompt,
                    title=title,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                result = conn.execute(stmt)
                
                # Get the last inserted ID
                stmt = text("SELECT lastval()")
                last_id = conn.execute(stmt).scalar()
                
                return last_id, None
        except Exception as e:
            return None, f"Error creating conversation: {str(e)}"
    
    def add_message(self, conversation_id, role, content):
        """Add a message to a conversation"""
        if not self.db_ready:
            return False, "Database is not available"
        
        try:
            with self.engine.begin() as conn:
                # Add message
                stmt = insert(self.messages_table).values(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    created_at=datetime.now()
                )
                conn.execute(stmt)
                
                # Update conversation's updated_at timestamp
                stmt = update(self.conversations_table).where(
                    self.conversations_table.c.id == conversation_id
                ).values(
                    updated_at=datetime.now()
                )
                conn.execute(stmt)
                
                return True, None
        except Exception as e:
            return False, f"Error adding message: {str(e)}"
    
    def delete_conversation(self, conversation_id):
        """Delete a conversation and all its messages"""
        if not self.db_ready:
            return False, "Database is not available"
        
        try:
            with self.engine.begin() as conn:
                # Messages will be cascade deleted due to foreign key constraint
                stmt = delete(self.conversations_table).where(
                    self.conversations_table.c.id == conversation_id
                )
                conn.execute(stmt)
                
                return True, None
        except Exception as e:
            return False, f"Error deleting conversation: {str(e)}"
            
    def update_conversation_system_prompt(self, conversation_id, system_prompt):
        """Update a conversation's system prompt"""
        if not self.db_ready:
            return False, "Database is not available"
        
        try:
            with self.engine.begin() as conn:
                stmt = update(self.conversations_table).where(
                    self.conversations_table.c.id == conversation_id
                ).values(
                    system_prompt=system_prompt,
                    updated_at=datetime.now()
                )
                conn.execute(stmt)
                
                return True, None
        except Exception as e:
            return False, f"Error updating system prompt: {str(e)}"

# File processing functions
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(file):
    return file.getvalue().decode("utf-8")

def read_csv_as_text(file):
    df = pd.read_csv(file)
    return df.head(5).to_string() + f"\n\n[Showing 5 of {len(df)} rows]"

def read_excel_as_text(file):
    df = pd.read_excel(file)
    return df.head(5).to_string() + f"\n\n[Showing 5 of {len(df)} rows]"

def extract_text(file, file_type):
    try:
        if file_type == "pdf":
            return extract_text_from_pdf(file)
        elif file_type == "docx":
            return extract_text_from_docx(file)
        elif file_type == "txt":
            return extract_text_from_txt(file)
        elif file_type == "csv":
            return read_csv_as_text(file)
        elif file_type in ["xlsx", "xls"]:
            return read_excel_as_text(file)
        else:
            return "Unsupported file format"
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Function to execute Python code safely with access to uploaded files
def execute_python_code(code, selected_files, storage_config):
    # Create a temporary directory to save files
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = {}
        
        # Copy selected files to temp directory
        for file_info in selected_files:
            # Get file content from MinIO
            temp_file_path = storage_config.get_document_content(file_info['id'], file_info)
            
            if temp_file_path.startswith("Error"):
                return "", temp_file_path
            
            # Add to file paths dictionary
            file_paths[file_info['name']] = temp_file_path
        
        # Create code to map file names to their paths
        file_paths_code = "# File paths dictionary\n"
        file_paths_code += "file_paths = {\n"
        for filename, filepath in file_paths.items():
            file_paths_code += f"    {filename!r}: {filepath!r},\n"
        file_paths_code += "}\n\n"
        
        # Add code to make file paths accessible more directly
        file_paths_code += "# Direct file path variables\n"
        for filename, filepath in file_paths.items():
            safe_varname = ''.join(c if c.isalnum() else '_' for c in os.path.splitext(filename)[0])
            file_paths_code += f"{safe_varname}_path = {filepath!r}\n"
        
        file_paths_code += "\n# Helper function to get file path\n"
        file_paths_code += "def get_file_path(filename):\n"
        file_paths_code += "    return file_paths.get(filename)\n\n"
        
        # Combine with the user code
        full_code = file_paths_code + "\n" + code
        
        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        # Create a figure directory for matplotlib plots
        fig_dir = os.path.join(temp_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        
        result = ""
        error = None
        
        try:
            # Add plot saving code
            plot_code = """
# Save any matplotlib figures
import matplotlib.pyplot as plt
import os
import numpy as np

original_show = plt.show
def custom_show():
    fig_num = len(os.listdir("{fig_dir}"))
    plt.savefig("{fig_dir}/figure_{{0}}.png".format(fig_num))
    original_show()
plt.show = custom_show
""".format(fig_dir=fig_dir.replace("\\", "\\\\"))
            
            # Execute the code
            exec_globals = {
                "__file__": "data_analysis.py",
                "pd": pd,
                "plt": plt, 
                "sns": sns,
                "np": __import__("numpy"),
                "os": os,
            }
            exec(plot_code + full_code, exec_globals)
            
            # Get the output
            result = redirected_output.getvalue()
            
            # Check if there are any figures to show
            figures = [os.path.join(fig_dir, f) for f in os.listdir(fig_dir) if f.endswith('.png')]
            if figures:
                result += "\n\n[Generated visualizations]"
                for fig_path in figures:
                    img = plt.imread(fig_path)
                    plt.figure(figsize=(10, 6))
                    plt.imshow(img)
                    plt.axis('off')
                    with st.expander(f"Visualization: {os.path.basename(fig_path)}"):
                        st.image(fig_path)
        
        except Exception as e:
            error = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
        
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            
            # Clean up temporary files
            for filepath in file_paths.values():
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except:
                    pass
        
        return result, error

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable or add it to your .env file.")
    st.stop()
    
# Initialize with minimal configuration to avoid compatibility issues
client = OpenAI(api_key=api_key)

# Initialize storage configuration
storage_config = StorageConfig()

# Initialize session state for current conversation
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = 0

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

# Load documents from PostgreSQL
st.session_state.documents = storage_config.get_all_documents()

# Load conversations from PostgreSQL
conversations = storage_config.get_all_conversations()

# If there are no conversations in the database but current_conversation_id is set, create a new one
if not conversations and st.session_state.current_conversation_id == 0:
    # Create a default conversation
    conv_id, error = storage_config.create_conversation(
        system_prompt=DEFAULT_SYSTEM_PROMPT, 
        title="New Conversation"
    )
    if error:
        st.error(f"Error creating default conversation: {error}")
    else:
        st.session_state.current_conversation_id = conv_id

# Function to generate Python code for data analysis
def generate_analysis_code(user_query, available_files):
    try:
        file_info = "\n".join([f"- {doc['name']} ({doc['type']})" for doc in available_files])
        prompt = CODE_GENERATION_PROMPT.format(
            available_files=file_info,
            user_question=user_query
        )
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.1,
        )
        
        code = response.choices[0].message.content
        
        # Remove markdown code block formatting if present
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        elif code.startswith("```"):
            code = code[3:].strip()
        
        if code.endswith("```"):
            code = code[:-3].strip()
            
        return code
    except Exception as e:
        return f"Error generating analysis code: {str(e)}"

# Function to generate a response
def generate_response(system_prompt, messages, analysis_results=None):
    try:
        full_messages = [{"role": "system", "content": system_prompt}]
        
        # If we have analysis results, add them to the context
        if analysis_results and analysis_results.get("result"):
            analysis_context = f"""
Here are the results of a data analysis performed on the user's files:

{analysis_results.get('result')}

Please use these insights to inform your response to the user's question.
"""
            # Add analysis context as a system message
            full_messages.append({"role": "system", "content": analysis_context})
        
        full_messages.extend(messages)
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=full_messages,
            temperature=0.7,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar for system prompt, document management and conversation management
with st.sidebar:
    st.title("System Prompt Experimenter")
    
    # Tabs for different sidebar sections
    sidebar_tab = st.radio("Sidebar Options", ["System Prompt", "Documents", "Conversations"])
    
    if sidebar_tab == "System Prompt":
        # Get current conversation from database
        conversations = storage_config.get_all_conversations()
        curr_conv_id = st.session_state.current_conversation_id
        current_conv = next((conv for conv_id, conv in conversations.items() if conv_id == curr_conv_id), None)
        
        # System prompt input
        current_prompt = current_conv.get('system_prompt', DEFAULT_SYSTEM_PROMPT) if current_conv else DEFAULT_SYSTEM_PROMPT
        system_prompt = st.text_area(
            "System Prompt",
            value=current_prompt,
            height=150
        )
        
        # Save system prompt to database
        if system_prompt != current_prompt and curr_conv_id:
            if st.button("Update System Prompt"):
                success, error = storage_config.update_conversation_system_prompt(curr_conv_id, system_prompt)
                if error:
                    st.error(error)
                else:
                    st.success("System prompt updated successfully")
        
        # Save current system prompt as a preset
        preset_name = st.text_input("Preset Name")
        if st.button("Save Preset") and preset_name:
            if "presets" not in st.session_state:
                st.session_state.presets = {}
            st.session_state.presets[preset_name] = system_prompt
            st.success(f"Preset '{preset_name}' saved!")
        
        # Load preset
        if "presets" in st.session_state and st.session_state.presets:
            selected_preset = st.selectbox("Load Preset", options=list(st.session_state.presets.keys()))
            if st.button("Load"):
                preset_prompt = st.session_state.presets[selected_preset]
                if curr_conv_id:
                    success, error = storage_config.update_conversation_system_prompt(curr_conv_id, preset_prompt)
                    if error:
                        st.error(error)
                    else:
                        st.success("System prompt updated from preset")
                        st.rerun()
    
    elif sidebar_tab == "Documents":
        if not storage_config.minio_ready or not storage_config.db_ready:
            st.error("Storage services are not properly configured. Please check your environment variables.")
        else:
            st.subheader("Upload Document")
            
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "csv", "xlsx", "xls"])
            
            if uploaded_file is not None:
                file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, 
                                "filesize": uploaded_file.size}
                st.write(file_details)
                
                file_id = str(uuid.uuid4())
                file_extension = uploaded_file.name.split(".")[-1].lower()
                
                # Extract preview text from the file
                file_text = extract_text(uploaded_file, file_extension)
                
                # Upload to MinIO and save to PostgreSQL
                result_id, error = storage_config.upload_file(
                    uploaded_file, 
                    uploaded_file.name, 
                    file_id, 
                    file_extension, 
                    file_text
                )
                
                if error:
                    st.error(error)
                else:
                    # Reload documents
                    st.session_state.documents = storage_config.get_all_documents()
                    st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # List uploaded documents
            st.subheader("Uploaded Documents")
            if st.session_state.documents:
                for doc_id, doc in st.session_state.documents.items():
                    with st.expander(f"{doc['name']} ({doc['upload_date']})"):
                        st.text(f"Type: {doc['type']}")
                        st.text("Preview:")
                        st.text(doc.get('preview', 'No preview available'))
                        
                        if st.button("Delete", key=f"del_{doc_id}"):
                            # Delete from MinIO and PostgreSQL
                            success, error = storage_config.delete_file(doc_id)
                            if error:
                                st.error(error)
                            else:
                                # Reload documents
                                st.session_state.documents = storage_config.get_all_documents()
                                st.success(f"File deleted successfully")
                                st.rerun()
            else:
                st.info("No documents uploaded yet.")
    
    elif sidebar_tab == "Conversations":
        # New conversation button
        if st.button("New Conversation"):
            conv_id, error = storage_config.create_conversation(
                system_prompt=DEFAULT_SYSTEM_PROMPT, 
                title=f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            if error:
                st.error(error)
            else:
                st.session_state.current_conversation_id = conv_id
                st.rerun()
        
        # Existing conversations
        st.subheader("Conversations")
        # Get updated list of conversations
        conversations = storage_config.get_all_conversations()
        for conv_id, conv in sorted(conversations.items(), key=lambda x: x[0], reverse=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"{conv['title']}", key=f"btn_conv_{conv_id}"):
                    st.session_state.current_conversation_id = conv_id
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_conv_{conv_id}"):
                    success, error = storage_config.delete_conversation(conv_id)
                    if error:
                        st.error(error)
                    else:
                        # If we deleted the current conversation, set current to first available
                        if st.session_state.current_conversation_id == conv_id:
                            remaining_convs = storage_config.get_all_conversations()
                            if remaining_convs:
                                st.session_state.current_conversation_id = next(iter(remaining_convs.keys()))
                            else:
                                # Create a new conversation if none left
                                new_id, _ = storage_config.create_conversation(
                                    system_prompt=DEFAULT_SYSTEM_PROMPT, 
                                    title="New Conversation"
                                )
                                st.session_state.current_conversation_id = new_id
                        st.rerun()

# Main chat area
st.title("Chat")

# Storage status
if not storage_config.minio_ready:
    st.warning("⚠️ MinIO storage is not available. Document functions will be limited.")
if not storage_config.db_ready:
    st.warning("⚠️ PostgreSQL database is not available. Document functions will be limited.")

# Document selection for the current query
if st.session_state.documents:
    # Allow user to select which documents to analyze
    doc_options = {doc['name']: doc_id for doc_id, doc in st.session_state.documents.items()}
    selected_docs = st.multiselect("Select files to analyze", options=list(doc_options.keys()))
    
    # Get the selected document data
    selected_doc_data = []
    for doc_name in selected_docs:
        doc_id = doc_options[doc_name]
        selected_doc_data.append(st.session_state.documents[doc_id])
    
    # Set analysis mode
    analysis_mode = st.radio("Analysis Mode", ["Generate Python Code", "Use Document Content Directly"], 
                             help="Choose how to analyze your documents. Generate Python Code mode will create and run custom code based on your question. Use Document Content Directly will send document text to the model.")
else:
    selected_doc_data = []
    analysis_mode = "Use Document Content Directly"  # Default

# Get current conversation ID
current_id = st.session_state.current_conversation_id

# Get conversation and messages from the database
conversations = storage_config.get_all_conversations()
current_conv = next((conv for conv_id, conv in conversations.items() if conv_id == current_id), None)

if not current_conv:
    # Create a new conversation if the current one doesn't exist
    conv_id, error = storage_config.create_conversation(
        system_prompt=DEFAULT_SYSTEM_PROMPT, 
        title="New Conversation"
    )
    if error:
        st.error(f"Error creating conversation: {error}")
    else:
        st.session_state.current_conversation_id = conv_id
        current_id = conv_id
        st.rerun()

# Get messages for current conversation
messages = storage_config.get_conversation_messages(current_id)

# Display messages
for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get the current system prompt
system_prompt = current_conv.get('system_prompt', DEFAULT_SYSTEM_PROMPT) if current_conv else DEFAULT_SYSTEM_PROMPT

# Chat input
prompt = st.chat_input("Type your message here...")
if prompt:
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Add to conversation history in the database
    success, error = storage_config.add_message(current_id, "user", prompt)
    if error:
        st.error(f"Error saving message: {error}")
    
    # Reset analysis results
    st.session_state.analysis_results = {}
    
    # If documents are selected and we're in code generation mode, generate and execute analysis code
    if selected_doc_data and analysis_mode == "Generate Python Code":
        with st.spinner("Generating and executing analysis code..."):
            # Generate Python code for data analysis
            st.subheader("Generated Analysis Code")
            code = generate_analysis_code(prompt, selected_doc_data)
            with st.expander("View Generated Code"):
                st.code(code, language="python")
            
            # Execute the generated code
            st.subheader("Analysis Results")
            result, error = execute_python_code(code, selected_doc_data, storage_config)
            
            if error:
                st.error("Error during analysis")
                st.code(error, language="python")
                st.session_state.analysis_results = {"error": error}
            else:
                st.text_area("Output", result, height=200)
                st.session_state.analysis_results = {"result": result}
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get the updated messages for the prompt context
            messages = storage_config.get_conversation_messages(current_id)
            
            if analysis_mode == "Generate Python Code":
                # Pass analysis results to the model
                response = generate_response(
                    system_prompt, 
                    messages,
                    st.session_state.analysis_results
                )
            else:
                # Use the document content directly approach
                doc_content = []
                for doc in selected_doc_data:
                    # Get document content from MinIO
                    if doc["type"] in ["csv", "xlsx", "xls"]:
                        # For tabular data, just use the preview
                        doc_content.append({
                            "name": doc["name"],
                            "content": doc.get("preview", "No preview available")
                        })
                    else:
                        # For text documents, get more content
                        temp_path = storage_config.get_document_content(doc['id'], doc)
                        if not temp_path.startswith("Error"):
                            try:
                                with open(temp_path, "r", errors="ignore") as f:
                                    content = f.read(10000)  # Read first 10K characters
                                doc_content.append({
                                    "name": doc["name"],
                                    "content": content
                                })
                                # Clean up temp file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                            except Exception as e:
                                doc_content.append({
                                    "name": doc["name"],
                                    "content": f"Error reading file: {str(e)}"
                                })
                        else:
                            doc_content.append({
                                "name": doc["name"],
                                "content": temp_path
                            })
                
                response = generate_response(
                    system_prompt, 
                    messages,
                    {"result": "\n\n".join([f"--- {doc['name']} ---\n{doc['content']}" for doc in doc_content])} if doc_content else None
                )
            
            st.write(response)
    
    # Add assistant response to conversation history in the database
    success, error = storage_config.add_message(current_id, "assistant", response)
    if error:
        st.error(f"Error saving response: {error}")

# Display conversation metadata
st.caption(f"Current conversation: {current_id} - {current_conv.get('title', 'Untitled')}")
