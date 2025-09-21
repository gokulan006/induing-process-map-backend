from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Body

import pdfplumber
import re
from transformers import GPT2TokenizerFast
import openai
import sqlite3
from datetime import datetime

DB_PATH = Path("./processmapper.db")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize database with proper tables"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create saved processes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_processes (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            name TEXT NOT NULL,
            process_data TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create tasks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_size INTEGER,
            status TEXT NOT NULL,
            process_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME,
            result_path TEXT
        )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_saved_processes_task_id ON saved_processes(task_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_saved_processes_created_at ON saved_processes(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
         
        raise

 
init_db()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FastAPI app initialization
app = FastAPI(
    title="ProcessMapper AI Backend",
    description="Complete backend API for SOP Processing and Visualization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000", 
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:5173",  # Vite default
    "http://localhost:5174",  # Vite alternate
    "http://127.0.0.1:5174",  # Vite alternate
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
UPLOAD_DIR = Path("./uploads")
RESULTS_DIR = Path("./results") 
VISUALIZATIONS_DIR = Path("./visualizations")
HISTORY_DIR = Path("./history")
STATIC_DIR = Path("./static")

for directory in [UPLOAD_DIR, RESULTS_DIR, VISUALIZATIONS_DIR, HISTORY_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# In-memory storage for demo (use Redis/PostgreSQL in production)
tasks: Dict[str, Dict[str, Any]] = {}
chat_sessions: Dict[str, List[Dict]] = defaultdict(list)
document_history: List[Dict[str, Any]] = []

# OpenAI client setup - FIXED MODEL NAME
openai_client = openai.OpenAI(
    api_key= os.environ.get("OPENAI_API_KEY")
)

class SaveProcessRequest(BaseModel):
    name: str


# FIXED: Integrated FlexibleSOPProcessor directly into main.py
class FlexibleSOPProcessor:
    """
    Smart SOP processor that automatically detects any number of processes
    and adapts the output schema accordingly.
    """
    
    def __init__(self, target_model="gpt-5"):  # FIXED: Use valid model name
        try:
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # Fallback: simple token counting
            self.tokenizer = None
        
        # FIXED: Updated model limits with correct model names
        self.model_limits = {
            "gpt-3.5-turbo": 4000,
            "gpt-4": 8000,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-5-mini": 160000
        }
        
        self.max_tokens = self.model_limits.get(target_model, 4000)
        self.target_model = target_model
        
        # Process detection patterns - flexible for any format
        self.process_patterns = [
            # Standard format: "2.0 AP-001: Invoice Processing"
            r'^(\d+\.\d+)\s+([A-Z]+-\d+):\s*(.+)$',
            # Alternative format: "Process 1: Invoice Management"
            r'^Process\s+(\d+):\s*(.+)$',
            # Section format: "Section A - Vendor Onboarding"
            r'^Section\s+([A-Z]+)\s*[-–]\s*(.+)$',
            # Simple numbered: "1. Invoice Processing Workflow"
            r'^(\d+)\s*\.\s*(.+?(?:Process|Workflow|Procedure|Operation)).*$',
            # Header format: "INVOICE PROCESSING AND APPROVAL"
            r'^([A-Z][A-Z\s]+)(?:PROCESS|PROCEDURE|WORKFLOW|OPERATION).*$'
        ]
    
    def get_token_count(self, text: str) -> int:
        """Get token count with fallback"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimation (4 chars per token)
            return len(text) // 4
    
    def process_document(self, file_path: str) -> Dict:
        """
        Main processing function that automatically detects processes
        """
        # Extract text from PDF
        raw_text = self.extract_pdf_text(file_path)
        
        # Analyze document size
        token_count = self.get_token_count(raw_text)
        
        # Detect processes dynamically
        detected_processes = self.detect_processes(raw_text)
        
        logger.info(f"📄 Document Analysis:")
        logger.info(f"   Tokens: {token_count:,}")
        logger.info(f"   Detected Processes: {len(detected_processes)}")
        for i, process in enumerate(detected_processes, 1):
            logger.info(f"     {i}. {process['name']}")
        
        if token_count <= self.max_tokens:
            logger.info("✅ Using Direct Processing")
            return self.direct_processing(raw_text, detected_processes)
        else:
            logger.info("⚙️ Using Intelligent Chunking")
            return self.chunked_processing(raw_text, detected_processes)
    
    def extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF with structure preservation"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")
        
        return text
    
    def detect_processes(self, text: str) -> List[Dict]:
        """
        Dynamically detect all processes in the document
        """
        processes = []
        lines = text.split('\n')
        current_process = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check against all process patterns
            for pattern in self.process_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Finalize previous process
                    if current_process:
                        processes.append(current_process)
                    
                    # Extract process information
                    if len(match.groups()) >= 3:  # Pattern with code
                        section_num, process_code, process_name = match.groups()
                        process_id = process_code
                    elif len(match.groups()) == 2:  # Pattern without code
                        section_identifier, process_name = match.groups()
                        process_id = f"PROC-{section_identifier}"
                    else:
                        process_name = match.groups()[0]
                        process_id = f"PROC-{len(processes) + 1}"
                    
                    # Start new process
                    current_process = {
                        'id': process_id,
                        'name': process_name.strip(),
                        'section_header': line,
                        'start_line': i,
                        'content_lines': [],
                        'raw_content': ""
                    }
                    break
            else:
                # Add content to current process
                if current_process:
                    current_process['content_lines'].append(line)
        
        # Add final process
        if current_process:
            processes.append(current_process)
        
        # Extract detailed content for each process
        for process in processes:
            process['raw_content'] = '\n'.join(process['content_lines'])
            process['steps'] = self.extract_process_steps(process['raw_content'])
            process['estimated_steps'] = len(process['steps'])
        
        # Handle case where no processes detected
        if not processes:
            # Treat entire document as single process
            processes = [{
                'id': 'PROC-001',
                'name': 'Main Process',
                'section_header': 'Main Process',
                'raw_content': text,
                'steps': self.extract_process_steps(text),
                'estimated_steps': len(self.extract_process_steps(text))
            }]
        
        return processes
    
    def extract_process_steps(self, content: str) -> List[Dict]:
        """Extract numbered steps from process content"""
        steps = []
        
        # Pattern for numbered steps
        step_pattern = r'^\s*(\d+)\s*\.\s*(.+?)(?=\n\s*\d+\s*\.|\n\s*[a-z]\)|$)'
        
        matches = re.finditer(step_pattern, content, re.MULTILINE | re.DOTALL)
        for match in matches:
            step_num, step_content = match.groups()
            steps.append({
                'number': int(step_num),
                'content': step_content.strip()
            })
        
        return steps
    
    def direct_processing(self, text: str, detected_processes: List[Dict]) -> Dict:
        """Process document directly without chunking"""
        # Create enhanced text with process annotations
        enhanced_text = self.enhance_text_for_llm(text, detected_processes)
        
        # Create dynamic prompt based on detected processes
        prompt = self.create_flexible_prompt(enhanced_text, detected_processes)
        
        return {
            "strategy": "direct_processing",
            "detected_processes": detected_processes,
            "chunks": [{
                "chunk_id": 1,
                "content": enhanced_text,
                "prompt": prompt,
                "token_count": self.get_token_count(prompt)
            }]
        }
    
    def enhance_text_for_llm(self, text: str, processes: List[Dict]) -> str:
        """Add intelligent annotations based on detected processes"""
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Page \d+', '', text)
        
        enhanced_lines = []
        lines = text.split('\n')
        current_process_id = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new process
            for process in processes:
                if process['section_header'] in line:
                    current_process_id = process['id']
                    enhanced_lines.append(f"[PROCESS_START: {process['id']}] {line}")
                    break
            else:
                # Annotate content based on type
                if re.match(r'^\d+\.\s+', line):
                    enhanced_lines.append(f"[PROCESS_STEP] {line}")
                elif any(word in line.lower() for word in ['control', 'risk', 'fraud', 'duplicate', 'compliance']):
                    enhanced_lines.append(f"[RISK_CONTROL] {line}")
                elif re.match(r'^\s*o\s+', line) or re.match(r'^\s*[a-z]\)\s+', line):
                    enhanced_lines.append(f"[SUB_STEP] {line}")
                elif any(word in line.lower() for word in ['overview', 'description', 'purpose']):
                    enhanced_lines.append(f"[PROCESS_DESCRIPTION] {line}")
                else:
                    enhanced_lines.append(line)
        
        return "\n".join(enhanced_lines)
    
    def create_flexible_prompt(self, enhanced_text: str, detected_processes: List[Dict]) -> str:
        """Create adaptive prompt based on detected processes"""
        
        # Build process-specific instructions
        process_instructions = []
        for i, process in enumerate(detected_processes, 1):
            process_instructions.append(f"Process {i}: '{process['name']}' (ID: {process['id']}) - Extract ~{process['estimated_steps']} steps")
        
        processes_description = "\n".join(process_instructions)
        
        return f"""
You are an expert business process analyst. Extract comprehensive process information from this SOP document.

DOCUMENT ANALYSIS:
- Detected {len(detected_processes)} process(es) in this document
- Processes to extract:
{processes_description}

DOCUMENT CONTENT:
{enhanced_text}

EXTRACTION REQUIREMENTS:
1. Identify ALL business processes mentioned in the document
2. For EACH process, extract:
   - Process name and description, the description should be concise yet comprehensive
   - Associated risks with categories and descriptions
   - Related controls with types and effectiveness
   - Generate BPMN-compatible XML structure with also diagram elements with proper aligned structured with enough spacing mentioning the direction of flow.
   - The flowchart should be very neat and organized, don't overlap any elements and maintain proper spacing and make professional looking flowchart
   - Subprocesses (If a subprocess is detected within a main process otherwise ignore this step):
   - Create a separate flowchart for the subprocess and connect it to the main process flowchart 

3. IMPORTANT: The number of processes can be 1, 2, or more - extract exactly what exists in the document

DYNAMIC OUTPUT SCHEMA:
Return valid JSON with this flexible structure:

{{
  "found_processes": true,
  "process_count": {len(detected_processes)},
  "processes": [
    {{
      "process_name": "Exact process name from document",
      "process_description": "Comprehensive description of what this process does",
      "process_map_bpmn_xml": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<bpmn:definitions xmlns:bpmn=\"http://www.omg.org/spec/BPMN/20100524/MODEL\" xmlns:bpmndi=\"http://www.omg.org/spec/BPMN/20100524/DI\">\n  <bpmn:process id=\"Process_1\">\n    <bpmn:startEvent id=\"StartEvent_1\" />\n    <bpmn:task id=\"Task_1\" name=\"Step 1\" />\n    <bpmn:endEvent id=\"EndEvent_1\" />\n  </bpmn:process>\n</bpmn:definitions>",
      "risk_taxonomy": [
        {{
          "category": "Risk Category (e.g., Fraud Risk, Operational Risk, Compliance Risk)",
          "risk_name": "Specific Risk Name",
          "description": "Detailed description of what this risk involves"
        }}
      ],
      "controls": [
        {{
          "control_name": "Specific Control Name",
          "control_type": "Control Type (e.g., Preventive, Detective, Corrective, Automated)",
          "description": "Detailed description of how this control works"
        }}
      ]
    }}
  ]
}}

CRITICAL INSTRUCTIONS:
- Extract ONLY the processes that actually exist in the document
- If document has 1 process, return 1 process
- If document has 3 processes, return 3 processes
- Do not assume a fixed number of processes
- Include ALL risks and controls mentioned for each process strictly from the document only
- Generate valid BPMN XML with proper start events, tasks, and end events
- Return ONLY valid JSON, no explanatory text

QUALITY CHECKS:
- Ensure process_count matches the actual number of processes returned
- Verify each process has meaningful steps, risks, and controls
- Validate BPMN XML structure is syntactically correct
"""
    
    def chunked_processing(self, text: str, detected_processes: List[Dict]) -> Dict:
        """Handle large documents with intelligent chunking by process"""
        chunks = []
        
        for process in detected_processes:
            # Create chunk for each process
            process_content = f"PROCESS: {process['name']}\n{process['section_header']}\n\n{process['raw_content']}"
            enhanced_content = self.enhance_text_for_llm(process_content, [process])
            prompt = self.create_single_process_prompt(enhanced_content, process)
            
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "process_id": process['id'],
                "process_name": process['name'],
                "content": enhanced_content,
                "prompt": prompt,
                "token_count": self.get_token_count(prompt)
            })
        
        return {
            "strategy": "chunked_processing",
            "detected_processes": detected_processes,
            "chunks": chunks
        }
    
    def create_single_process_prompt(self, enhanced_content: str, process: Dict) -> str:
        """Create prompt for individual process processing"""
        return f"""
Extract information for this single business process:

PROCESS: {process['name']} (ID: {process['id']})

CONTENT:
{enhanced_content}

Extract and return JSON for this ONE process only:
{{
  "process_name": "{process['name']}",
  "process_description": "CONCISE purpose description (max 2 sentences)",
  "process_map_bpmn_xml": "Complete BPMN XML with positioning data",
  "subprocesses": [],
  "risk_taxonomy": [...],
  "controls": [
    {{
      "control_name": "Control name",
      "control_type": "Type",
      "description": "Description",
      "addresses_risk": "Risk name",
      "source": "SOP_EXTRACTED or LLM_GENERATED"
    }}
  ]
}}

Return ONLY valid JSON for this single process.
"""

# Pydantic models
class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: int = 0
    detail: str = ""
    error_message: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime

class ProcessResult(BaseModel):
    found_processes: bool
    process_count: int
    processes: List[Dict[str, Any]]

class HistoryItem(BaseModel):
    id: str
    name: str
    processCount: int
    generatedDate: str
    status: str
    type: str
    size: Optional[int] = None
    task_id: str

class ProcessExplanationRequest(BaseModel):
    process_name: str
    process_description: str
    risk_taxonomy: List[Dict[str, Any]]
    controls: List[Dict[str, Any]]

class ProcessExplanationResponse(BaseModel):
    explanation: str
    key_insights: List[str]
    recommendations: List[str]
    complexity_score: int  # 1-10 scale

@app.post("/api/explain-process", response_model=ProcessExplanationResponse)
async def explain_process(request: ProcessExplanationRequest):
    """Generate AI explanation for a specific process"""
    try:
        # Build comprehensive prompt for LLM
        prompt = f"""
You are an expert business process analyst. Provide a comprehensive, easy-to-understand explanation of this business process.

PROCESS INFORMATION:
Process Name: {request.process_name}
Description: {request.process_description}

IDENTIFIED RISKS:
{chr(10).join([f"- {risk.get('risk_name', 'Unknown')}: {risk.get('description', 'No description')} (Category: {risk.get('category', 'Unknown')})" for risk in request.risk_taxonomy])}

CONTROL MEASURES:
{chr(10).join([f"- {control.get('control_name', 'Unknown')}: {control.get('description', 'No description')} (Type: {control.get('control_type', 'Unknown')}, Source: {control.get('source', 'Unknown')}) - Addresses: {control.get('addresses_risk', 'Unknown risk')}" for control in request.controls])}

ANALYSIS REQUIREMENTS:

1. EXECUTIVE EXPLANATION (2-3 paragraphs):
   - Explain what this process does in business terms
   - Why it's important to the organization
   - How it fits into the broader business context

2. KEY INSIGHTS (3-5 bullet points):
   - Most critical aspects of this process
   - Notable patterns in risks and controls
   - Efficiency considerations

3. STRATEGIC RECOMMENDATIONS (3-4 bullet points):
   - Process optimization opportunities
   - Risk mitigation improvements
   - Technology or automation suggestions
   - Compliance enhancements

4. COMPLEXITY ASSESSMENT:
   - Rate process complexity from 1-10 (1=very simple, 10=extremely complex)
   - Consider: number of steps, stakeholders, risks, controls, dependencies

FORMAT YOUR RESPONSE AS JSON:
{{
    "explanation": "Comprehensive 2-3 paragraph explanation in business language",
    "key_insights": ["Insight 1", "Insight 2", "Insight 3", ...],
    "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3", ...],
    "complexity_score": integer from 1-10
}}

IMPORTANT: Return ONLY valid JSON. No additional text or formatting.
"""

        # Call OpenAI API
        response = await openai_client.chat.completions.create(
            model="gpt-5",   
            messages=[
                {"role": "system", "content": "You are an expert business process analyst. Always return valid JSON only."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse response
        raw_response = response.choices[0].message.content
        logger.info(f"AI explanation response preview: {raw_response[:200]}...")
        
        try:
            explanation_data = json.loads(raw_response)
            
            return ProcessExplanationResponse(
                explanation=explanation_data.get("explanation", "Unable to generate explanation"),
                key_insights=explanation_data.get("key_insights", []),
                recommendations=explanation_data.get("recommendations", []),
                complexity_score=explanation_data.get("complexity_score", 5)
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for explanation: {e}")
            logger.error(f"Raw response: {raw_response}")
            
            # Fallback response
            return ProcessExplanationResponse(
                explanation=f"This is a {request.process_name.lower()} process with {len(request.risk_taxonomy)} identified risks and {len(request.controls)} control measures. The process involves systematic handling of business operations with appropriate risk management and control mechanisms in place.",
                key_insights=[
                    f"Process manages {len(request.risk_taxonomy)} distinct risk categories",
                    f"Has {len(request.controls)} control measures implemented",
                    "Requires systematic approach for optimal execution"
                ],
                recommendations=[
                    "Regular review of risk assessments",
                    "Periodic testing of control effectiveness",
                    "Consider automation opportunities",
                    "Ensure staff training on procedures"
                ],
                complexity_score=min(max(1, len(request.risk_taxonomy) + len(request.controls)), 10)
            )

    except Exception as e:
        logger.error(f"Process explanation failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate process explanation: {str(e)}"
        )
# =============================================
# FILE UPLOAD AND PROCESSING ENDPOINTS
# =============================================

@app.post("/api/upload", response_model=Dict[str, Any])
async def upload_file(file: UploadFile = File(...)):
    """Upload SOP document - matches UploadPanel.tsx requirements"""
    
    # Validate file type
    allowed_types = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/plain'
    ]

    
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
    
    if file.content_type not in allowed_types and not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload PDF, DOCX, DOC, or TXT files."
        )
    
    
     
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:   
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
    
     
    task_id = str(uuid.uuid4())
    
     
    filename = f"{task_id}_{file.filename}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)
    
     
    tasks[task_id] = {
        "status": "uploaded",
        "progress": 10,
        "file_path": str(file_path),
        "original_filename": file.filename,
        "file_size": len(file_content),
        "created_at": datetime.now().isoformat(),
        "detail": "File uploaded successfully"
    }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO tasks (id, filename, file_size, status) VALUES (?, ?, ?, ?)",
        (task_id, file.filename, len(file_content), "uploaded")
    )
    
    conn.commit()
    conn.close()

    
    history_item = {
        "id": task_id,
        "name": file.filename,
        "processCount": 0,   
        "generatedDate": datetime.now().isoformat(),
        "status": "uploaded",
        "type": file.filename.split('.')[-1].lower(),
        "size": len(file_content),
        "task_id": task_id
    }
    document_history.append(history_item)
    
    logger.info(f"File uploaded successfully: {filename} (Task: {task_id})")
    
    return {
        "task_id": task_id,
        "filename": file.filename,
        "file_size": len(file_content),
        "message": "File uploaded successfully",
        "status": "uploaded"
    }

@app.post("/api/process/{task_id}")
async def start_processing(task_id: str, background_tasks: BackgroundTasks):
    """Start SOP processing - matches UploadPanel.tsx generateProcessMap"""
    
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    if task.get("status") in ("processing", "completed"):
        return {
            "message": f"Task is already {task['status']}",
            "status": task["status"],
            "progress": task.get("progress", 0)
        }
    
    # Update task status
    tasks[task_id]["status"] = "processing"
    tasks[task_id]["progress"] = 20
    tasks[task_id]["detail"] = "Starting document analysis..."
    
    # Start background processing
    background_tasks.add_task(process_document_task, task_id)
    
    logger.info(f"Started processing task: {task_id}")
    
    return {
        "task_id": task_id,
        "message": "Processing started",
        "status": "processing"
    }

async def process_document_task(task_id: str):
    """Background task for processing documents - FIXED VERSION"""
    try:
        task = tasks[task_id]
        file_path = task["file_path"]
        
        # Update progress
        tasks[task_id]["progress"] = 30
        tasks[task_id]["detail"] = "Analyzing document structure..."
        
        # Initialize processor
        processor = FlexibleSOPProcessor(target_model="gpt-5")  
        
        # Process document
        tasks[task_id]["progress"] = 50
        tasks[task_id]["detail"] = "Extracting processes and generating analysis..."
        
        result = processor.process_document(file_path)
        
         
        tasks[task_id]["progress"] = 70
        tasks[task_id]["detail"] = "Generating AI analysis..."
        
        all_processes = []
        
        for chunk in result["chunks"]:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-5",   
                    messages=[
                        {"role": "system", "content": "You are an expert business process analyst. Always return valid JSON only."},
                        {"role": "user", "content": chunk["prompt"]}
                    ]
                )
                
                raw_response = response.choices[0].message.content
                logger.info(f"Raw response preview: {raw_response[:200]}...")
                
                if result['strategy'] == "direct_processing":
                    extracted_data = json.loads(raw_response)
                    all_processes = extracted_data.get("processes", [])
                    break
                else:
                    process_data = json.loads(raw_response)
                    all_processes.append(process_data)
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for task {task_id}: {e}")
                logger.error(f"Raw response: {raw_response}")
                continue
            except Exception as e:
                logger.error(f"OpenAI API error for task {task_id}: {e}")
                continue
        
        # Combine results
        final_result = {
            "found_processes": len(all_processes) > 0,
            "process_count": len(all_processes),
            "processes": all_processes,
            "metadata": {
                "strategy": result["strategy"],
                "detected_processes": len(result["detected_processes"]),
                "chunks_processed": len(result["chunks"]),
                "processing_time": datetime.now().isoformat()
            }
        }
        
        # Save results
        tasks[task_id]["progress"] = 90
        tasks[task_id]["detail"] = "Generating visualizations..."
        
        result_path = RESULTS_DIR / f"{task_id}_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
        "UPDATE tasks SET status = ?, process_count = ?, completed_at = ?, result_path = ? WHERE id = ?",
        ("completed", final_result["process_count"], datetime.now().isoformat(), str(result_path), task_id)
        )

        conn.commit()
        conn.close()

        # Generate HTML visualization
        html_content = generate_html_visualizer(final_result, task_id)
        html_path = VISUALIZATIONS_DIR / f"{task_id}_visualizer.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Update task completion
        tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "detail": "Processing completed successfully",
            "result_path": str(result_path),
            "html_path": str(html_path),
            "result": final_result,
            "completed_at": datetime.now().isoformat()
        })
        
        # Update history
        for item in document_history:
            if item["id"] == task_id:
                item["status"] = "completed"
                item["processCount"] = final_result["process_count"]
                break
        
        logger.info(f"Processing completed successfully for task: {task_id}")
        
    except Exception as e:
        logger.error(f"Processing failed for task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        
        tasks[task_id].update({
            "status": "error",
            "progress": 0,
            "detail": "Processing failed",
            "error_message": str(e),
            "failed_at": datetime.now().isoformat()
        })
        
        # Update history
        for item in document_history:
            if item["id"] == task_id:
                item["status"] = "error"
                break
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            ("error", task_id)
        )
        
        conn.commit()
        conn.close()


def generate_html_visualizer(result: Dict[str, Any], task_id: str) -> str:
    """Generate HTML visualizer for the frontend"""
    
    processes_json = json.dumps(result, indent=2)
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ProcessMapper AI - Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }}
        .content {{
            padding: 2rem;
        }}
        .process-card {{
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            margin-bottom: 2rem;
            overflow: hidden;
        }}
        .process-header {{
            background: #f3f4f6;
            padding: 1rem;
            border-bottom: 1px solid #e5e7eb;
        }}
        .process-content {{
            padding: 1.5rem;
        }}
        .risk-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }}
        .risk-item {{
            background: #fef2f2;
            border: 1px solid #fca5a5;
            border-radius: 8px;
            padding: 1rem;
        }}
        .control-item {{
            background: #f0fdf4;
            border: 1px solid #86efac;
            border-radius: 8px;
            padding: 1rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            background: #3b82f6;
            color: white;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔄 Process Map Visualization</h1>
            <p>Task ID: {task_id}</p>
            <p>Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="content">
            <div class="mb-4">
                <h2>Analysis Summary</h2>
                <p><strong>Processes Found:</strong> {result["process_count"]}</p>
                <p><strong>Total Risks:</strong> {sum(len(p.get("risk_taxonomy", [])) for p in result["processes"])}</p>
                <p><strong>Total Controls:</strong> {sum(len(p.get("controls", [])) for p in result["processes"])}</p>
            </div>
'''
    
    # Add process details
    for i, process in enumerate(result["processes"], 1):
        risks = process.get("risk_taxonomy", [])
        controls = process.get("controls", [])
        
        html_content += f'''
            <div class="process-card">
                <div class="process-header">
                    <h3>Process {i}: {process.get("process_name", "Unnamed Process")}</h3>
                    <span class="badge">BPMN Available</span>
                </div>
                <div class="process-content">
                    <p><strong>Description:</strong> {process.get("process_description", "No description available")}</p>
                    
                    <div class="risk-grid">
                        <div>
                            <h4>🚨 Risks ({len(risks)})</h4>
                            {''.join(f'<div class="risk-item"><strong>{risk.get("risk_name", "Unnamed Risk")}</strong><br><small>{risk.get("category", "Unknown Category")}</small><p>{risk.get("description", "No description")}</p></div>' for risk in risks)}
                        </div>
                        <div>
                            <h4>🛡️ Controls ({len(controls)})</h4>
                            {''.join(f'<div class="control-item"><strong>{control.get("control_name", "Unnamed Control")}</strong><br><small>{control.get("control_type", "Unknown Type")} - {control.get("source", "Unknown Source")}</small><p>{control.get("description", "No description")}</p></div>' for control in controls)}
                        </div>
                    </div>
                </div>
            </div>
        '''
    
    html_content += '''
        </div>
    </div>
    
    <script>
        console.log('ProcessMapper AI Visualization Loaded');
        // Add any interactive features here
    </script>
</body>
</html>
    '''
    return html_content

# Saving and loading processed results

# Replace the existing save-process endpoint with this corrected version
@app.post("/api/save-process/{task_id}")
async def save_process_result(task_id: str, request: dict):
    """Save processed result with a custom name"""
    try:
        # Extract name from request
        name = request.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Name is required")
        
        task = tasks.get(task_id)
        result_data = None
        
        if not task:
            # Try to find in database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("SELECT result_path FROM tasks WHERE id = ?", (task_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result and os.path.exists(result[0]):
                with open(result[0], "r", encoding="utf-8") as f:
                    result_data = json.load(f)
            else:
                raise HTTPException(status_code=404, detail="Task ID not found")
        else:
            if task.get("status") != "completed":
                raise HTTPException(status_code=400, detail="Processing not completed yet")
            
            result_data = task.get("result")
            if not result_data:
                 
                result_path = task.get("result_path")
                if result_path and os.path.exists(result_path):
                    with open(result_path, "r", encoding="utf-8") as f:
                        result_data = json.load(f)
                else:
                    raise HTTPException(status_code=404, detail="Result data not found")
        
         
        save_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO saved_processes (id, task_id, name, process_data) VALUES (?, ?, ?, ?)",
                (save_id, task_id, name, json.dumps(result_data))
            )
            
            conn.commit()
            logger.info(f"Process saved successfully: {save_id} for task {task_id}")
            
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Database error saving process: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()
        
        return {
            "id": save_id, 
            "name": name, 
            "message": "Process saved successfully",
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving process: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save process: {str(e)}")
      
@app.get("/api/saved-processes")
async def get_saved_processes():
    """Get all saved processes"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, task_id, created_at 
            FROM saved_processes 
            ORDER BY created_at DESC
        """)
        processes = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                "id": row[0],
                "name": row[1],
                "task_id": row[2],
                "created_at": row[3]
            }
            for row in processes
        ]
        
    except Exception as e:
        logger.error(f"Error fetching saved processes: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch saved processes")
    

@app.get("/api/saved-process/{save_id}")
async def get_saved_process(save_id: str):
    """Get a specific saved process"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT process_data FROM saved_processes WHERE id = ?", (save_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Saved process not found")
        
        return json.loads(result[0])
        
    except Exception as e:
        logger.error(f"Error fetching saved process {save_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch saved process")
    

@app.get("/api/load-history/{task_id}")
async def load_history_item(task_id: str):
    """Load a specific history item by task_id"""
    task = tasks.get(task_id)
    if not task:
         
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT result_path FROM tasks WHERE id = ?", (task_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result and os.path.exists(result[0]):
            with open(result[0], "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Task ID not found")
    
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    result = task.get("result")
    if not result:
        raise HTTPException(status_code=404, detail="Result data not found")
    
    return result

# =============================================
# STATUS AND RESULT ENDPOINTS
# =============================================

@app.get("/api/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get processing status - matches frontend polling requirements"""
    
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0),
        detail=task.get("detail", ""),
        error_message=task.get("error_message")
    )

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """Get processing results JSON - matches UploadPanel.tsx downloadJSON"""
    
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    result_path = task.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        result_path,
        media_type="application/json",
        filename=f"process_map_{task_id}.json"
    )

@app.get("/api/visualize/{task_id}", response_class=HTMLResponse)
async def get_visualization(task_id: str):
    """Get HTML visualization - matches UploadPanel.tsx showVisualization"""
    
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    if task.get("status") != "completed":
        return HTMLResponse(
            content="<h2>Visualization not ready yet. Processing in progress...</h2>",
            status_code=202
        )
    
    html_path = task.get("html_path")
    if not html_path or not os.path.exists(html_path):
        return HTMLResponse(
            content="<h2>Visualization file not found</h2>",
            status_code=404
        )
    
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return HTMLResponse(content=content)

@app.get("/api/result-data/{task_id}")
async def get_result_data(task_id: str):
    """Get result data as JSON object - for frontend display"""
    
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    result = task.get("result")
    if not result:
        raise HTTPException(status_code=404, detail="Result data not found")
    
    return result

# =============================================
# CHAT ASSISTANT ENDPOINTS
# =============================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_assistant(message: ChatMessage):
    """Chat assistant endpoint using GPT-5 - matches ChatAssistant.tsx"""
    
    session_id = message.session_id or "default"
    user_message = message.message
    
    try:
         
        prompt = f"""
You are an AI assistant for ProcessMapper, a platform that analyzes SOP documents to generate process maps, 
identify risks, and suggest controls. Respond to the user's question in a helpful, professional manner.

User question: {user_message}

Context about ProcessMapper:
- Users can upload PDF and DOCX files up to 10MB
- The system generates BPMN process maps with risk analysis
- Features include process visualization, risk taxonomy, and control frameworks
- Users can view history of previous analyses
- The platform provides REST APIs for integration

Instructions:
1. Answer questions about ProcessMapper's features and capabilities
2. Provide helpful guidance on using the platform
3. Be concise but thorough in responses
4. If the question is not related to ProcessMapper, politely redirect
5. Maintain a professional, helpful tone

Response:
"""
        
        # Call OpenAI API with GPT-5-mini
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for ProcessMapper, a business process analysis platform."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the response
        ai_response = response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"GPT-5-mini API error: {e}")
        # Fallback to predefined responses if API call fails
        user_message_lower = user_message.lower()
        
        if any(keyword in user_message_lower for keyword in ['upload', 'file']):
            ai_response = "To upload a document, click the 'Upload SOP Document' button on the homepage. We support PDF and DOCX files up to 10MB. After upload, you can generate a process map with AI analysis."
        
        elif any(keyword in user_message_lower for keyword in ['process map', 'generate', 'analysis']):
            ai_response = "Our AI analyzes your SOP documents to automatically identify processes, risks, and controls. The generated process map includes BPMN visualization, risk taxonomy, and control frameworks with complete positioning data."
        
        elif any(keyword in user_message_lower for keyword in ['history', 'previous', 'past']):
            ai_response = "You can view all your previously generated process maps by clicking the 'View History' button. From there, you can search, filter, download results, and view visualizations."
        
        elif any(keyword in user_message_lower for keyword in ['format', 'support', 'file type']):
            ai_response = "We support PDF (.pdf) and Microsoft Word (.docx, .doc) formats. Make sure your documents contain structured process information with clear steps, roles, and procedures for best results."
        
        elif any(keyword in user_message_lower for keyword in ['risk', 'control']):
            ai_response = "Our AI identifies potential risks within your processes and suggests appropriate controls. We categorize risks by type (operational, compliance, fraud, etc.) and provide detailed mitigation strategies with source attribution."
        
        elif any(keyword in user_message_lower for keyword in ['bpmn', 'visualization', 'diagram']):
            ai_response = "ProcessMapper generates industry-standard BPMN 2.0 XML with complete visual positioning data. You can view interactive process maps and export them for use in other process management tools."
        
        elif any(keyword in user_message_lower for keyword in ['error', 'problem', 'issue']):
            ai_response = "If you're experiencing issues: 1) Check your file format (PDF/DOCX only), 2) Ensure file size is under 10MB, 3) Wait for processing to complete, 4) Try refreshing the page. Contact support if problems persist."
        
        elif any(keyword in user_message_lower for keyword in ['api', 'integration', 'technical']):
            ai_response = "ProcessMapper provides REST APIs for integration: POST /api/upload for files, POST /api/process/{task_id} to start processing, GET /api/status/{task_id} for progress, and GET /api/result/{task_id} for results."
        
        else:
            ai_response = "I can help you with uploading documents, generating process maps, understanding risk analysis, viewing history, and navigating ProcessMapper AI. What specific question do you have about our platform?"

    # Store conversation
    chat_entry = {
        "user_message": user_message,
        "bot_response": ai_response,
        "timestamp": datetime.now().isoformat()
    }
    chat_sessions[session_id].append(chat_entry)
    
    return ChatResponse(
        response=ai_response,
        session_id=session_id,
        timestamp=datetime.now()
    )

@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    return {
        "session_id": session_id,
        "messages": chat_sessions.get(session_id, [])
    }

# =============================================
# HISTORY MANAGEMENT ENDPOINTS
# =============================================

@app.get("/api/history")
async def get_document_history(
    search: Optional[str] = Query(None, description="Search term for filtering"),
    filter_type: Optional[str] = Query("all", description="File type filter")
):
    """Get document history - matches HistoryPanel.tsx"""
    
    filtered_history = document_history.copy()
    
    # Apply search filter
    if search:
        filtered_history = [
            item for item in filtered_history 
            if search.lower() in item["name"].lower()
        ]
    
    # Apply type filter
    if filter_type and filter_type != "all":
        filtered_history = [
            item for item in filtered_history 
            if item["type"] == filter_type
        ]
    
    # Sort by date (newest first)
    filtered_history.sort(key=lambda x: x["generatedDate"], reverse=True)
    
    return {
        "history": filtered_history,
        "total": len(document_history),
        "filtered": len(filtered_history)
    }

@app.delete("/api/history/{task_id}")
async def delete_history_item(task_id: str):
    """Delete a history item and associated files"""
    
    # Remove from history
    global document_history
    document_history = [item for item in document_history if item["id"] != task_id]
    
     
    if task_id in tasks:
        task = tasks[task_id]
        
         
        for file_path in [task.get("file_path"), task.get("result_path"), task.get("html_path")]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Failed to delete file {file_path}: {e}")
        
        del tasks[task_id]
    
    return {"message": "History item deleted successfully"}

# =============================================
# UTILITY ENDPOINTS
# =============================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(tasks),
        "completed_tasks": len([t for t in tasks.values() if t.get("status") == "completed"]),
        "total_documents": len(document_history)
    }

@app.get("/api/stats")
async def get_statistics():
    """Get platform statistics"""
    
    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks.values() if t.get("status") == "completed"])
    processing_tasks = len([t for t in tasks.values() if t.get("status") == "processing"])
    error_tasks = len([t for t in tasks.values() if t.get("status") == "error"])
    
    total_processes = sum(
        t.get("result", {}).get("process_count", 0) 
        for t in tasks.values() 
        if t.get("status") == "completed"
    )
    
    return {
        "total_documents": total_tasks,
        "completed_documents": completed_tasks,
        "processing_documents": processing_tasks,
        "error_documents": error_tasks,
        "total_processes_extracted": total_processes,
        "success_rate": round((completed_tasks / max(total_tasks, 1)) * 100, 1)
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ProcessMapper AI Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "upload": "POST /api/upload",
            "process": "POST /api/process/{task_id}",
            "status": "GET /api/status/{task_id}",
            "result": "GET /api/result/{task_id}",
            "visualize": "GET /api/visualize/{task_id}",
            "chat": "POST /api/chat",
            "history": "GET /api/history"
        }
    }

# =============================================
# ERROR HANDLERS
# =============================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )

# =============================================
# STARTUP AND SHUTDOWN EVENTS
# =============================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("ProcessMapper AI Backend starting up...")
    init_db()
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Visualizations directory: {VISUALIZATIONS_DIR}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("ProcessMapper AI Backend shutting down...")

# =============================================
# MAIN APPLICATION RUNNER
# =============================================

if __name__ == "__main__":
    import uvicorn
    
     
    uvicorn.run(
        "main:app",   
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )