# Complete YAML-Based Workflow Orchestration System

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Workflow Definition Layer                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ YAML Files │  │  Schemas   │  │  Templates │                │
│  └────────────┘  └────────────┘  └────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Workflow Engine (Python)                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Parser → Validator → DAG Builder → Executor               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Node Executors                           │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │  Input   │  Agent   │  Merge   │  Batch   │  Output  │      │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘      │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     AWS Services (Runtime)                       │
│  S3 │ Bedrock │ Lambda │ Step Functions │ DynamoDB              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. YAML Workflow Schema

### 2.1 Complete Workflow Example

**`workflows/resume-shortlisting.yaml`:**

```yaml
workflow:
  name: resume_shortlisting
  version: 1.0
  description: Analyze resumes and shortlist candidates based on job requirements
  
  metadata:
    author: platform-team
    created_at: 2025-10-14
    tags: [recruitment, ai, automation]
  
  # Global configuration
  config:
    timeout: 1800  # 30 minutes
    retry_policy:
      max_attempts: 3
      backoff: exponential
    
  # Define workflow nodes
  nodes:
    # ============================================
    # 1. Input Node - File Upload
    # ============================================
    - id: upload_files
      type: input
      description: Upload resumes and job description
      
      config:
        fields:
          - name: resumes
            type: file_upload
            accept: [pdf, docx]
            multiple: true
            required: true
            max_size_mb: 10
            min_files: 1
            max_files: 50
            category: resume
            
          - name: job_description
            type: file_upload
            accept: [pdf, docx, txt]
            required: true
            category: jd
            
          - name: min_experience_years
            type: number
            default: 0
            min: 0
            max: 50
            description: Minimum years of experience required
      
      outputs:
        - name: resume_files
          type: file_array
        - name: jd_file
          type: file
        - name: requirements
          type: object
    
    # ============================================
    # 2. Agent Node - Extract JD Requirements
    # ============================================
    - id: extract_requirements
      type: agent
      depends_on: [upload_files]
      description: Extract key requirements from job description
      
      config:
        agent:
          name: jd_analyzer
          
          model:
            provider: bedrock
            model_id: anthropic.claude-3-sonnet-20240229-v1:0
            temperature: 0.1
            max_tokens: 2000
          
          prompt_template: |
            Extract key requirements from this job description.
            
            Job Description:
            {{job_description}}
            
            Provide a structured JSON response with:
            {
              "title": "Job title",
              "required_skills": ["skill1", "skill2", ...],
              "preferred_skills": ["skill1", "skill2", ...],
              "min_experience_years": number,
              "education_required": ["degree1", "degree2", ...],
              "responsibilities": ["resp1", "resp2", ...],
              "must_have_keywords": ["keyword1", "keyword2", ...]
            }
            
            Return ONLY valid JSON.
          
          input_mapping:
            job_description: $nodes.upload_files.outputs.jd_file.text
          
          output_schema:
            type: object
            required: [title, required_skills, min_experience_years]
            properties:
              title: {type: string}
              required_skills: {type: array, items: {type: string}}
              preferred_skills: {type: array, items: {type: string}}
              min_experience_years: {type: number}
              education_required: {type: array}
              responsibilities: {type: array}
              must_have_keywords: {type: array}
      
      outputs:
        - name: requirements
          type: object
    
    # ============================================
    # 3. Batch Node - Process Resumes in Batches
    # ============================================
    - id: batch_resumes
      type: batch
      depends_on: [upload_files]
      description: Group resumes into batches for parallel processing
      
      config:
        batch_size: 10
        strategy: even_distribution
        
        input_mapping:
          items: $nodes.upload_files.outputs.resume_files
      
      outputs:
        - name: batches
          type: array
    
    # ============================================
    # 4. Parallel Node - Analyze Each Resume
    # ============================================
    - id: analyze_resumes
      type: parallel
      depends_on: [extract_requirements, batch_resumes]
      description: Analyze each resume against requirements
      
      config:
        concurrency: 5  # Process 5 resumes at a time
        foreach: $nodes.batch_resumes.outputs.batches
        
        agent:
          name: resume_evaluator
          
          model:
            provider: bedrock
            model_id: anthropic.claude-3-haiku-20240307-v1:0
            temperature: 0.2
            max_tokens: 1500
          
          prompt_template: |
            Evaluate this resume against the job requirements.
            
            Job Requirements:
            {{requirements}}
            
            Resume:
            {{resume}}
            
            Provide evaluation in JSON:
            {
              "candidate_name": "Full name",
              "email": "email",
              "phone": "phone",
              "overall_score": 0-100,
              "experience_match": {
                "years": number,
                "score": 0-100,
                "relevant_roles": ["role1", "role2"]
              },
              "skills_match": {
                "matched_skills": ["skill1", "skill2"],
                "missing_skills": ["skill1", "skill2"],
                "score": 0-100
              },
              "education_match": {
                "degrees": ["degree1", "degree2"],
                "score": 0-100
              },
              "strengths": ["strength1", "strength2", "strength3"],
              "concerns": ["concern1", "concern2"],
              "recommendation": "strong_yes|yes|maybe|no|strong_no",
              "reasoning": "Brief explanation of recommendation"
            }
          
          input_mapping:
            requirements: $nodes.extract_requirements.outputs.requirements
            resume: $item
          
          retry:
            max_attempts: 2
            backoff_seconds: 5
      
      outputs:
        - name: evaluations
          type: array
    
    # ============================================
    # 5. Merge Node - Aggregate Results
    # ============================================
    - id: aggregate_results
      type: merge
      depends_on: [analyze_resumes]
      description: Combine all resume evaluations
      
      config:
        strategy: flatten
        
        input_mapping:
          items: $nodes.analyze_resumes.outputs.evaluations
      
      outputs:
        - name: all_evaluations
          type: array
    
    # ============================================
    # 6. Conditional Node - Filter Candidates
    # ============================================
    - id: categorize_candidates
      type: conditional
      depends_on: [aggregate_results]
      description: Categorize candidates based on scores
      
      config:
        conditions:
          - when: "$item.overall_score >= 80 && $item.recommendation in ['strong_yes', 'yes']"
            then: shortlisted
            
          - when: "$item.overall_score >= 60 && $item.overall_score < 80"
            then: maybe_list
            
          - otherwise: rejected
        
        input_mapping:
          items: $nodes.aggregate_results.outputs.all_evaluations
      
      outputs:
        - name: shortlisted
          type: array
        - name: maybe_list
          type: array
        - name: rejected
          type: array
    
    # ============================================
    # 7. Agent Node - Generate Interview Questions
    # ============================================
    - id: generate_questions
      type: agent
      depends_on: [categorize_candidates, extract_requirements]
      description: Generate interview questions for shortlisted candidates
      
      config:
        agent:
          name: question_generator
          
          model:
            provider: bedrock
            model_id: anthropic.claude-3-sonnet-20240229-v1:0
            temperature: 0.7
            max_tokens: 1000
          
          prompt_template: |
            Generate 5 targeted interview questions for this candidate.
            
            Job Requirements:
            {{requirements}}
            
            Candidate Profile:
            - Name: {{candidate.candidate_name}}
            - Strengths: {{candidate.strengths}}
            - Areas to probe: {{candidate.concerns}}
            
            Generate questions that:
            1. Validate their strengths
            2. Explore potential concerns
            3. Assess cultural fit
            4. Test technical depth
            5. Understand career motivation
            
            Return JSON:
            {
              "candidate_name": "name",
              "questions": [
                {"category": "technical", "question": "...", "why_asking": "..."},
                ...
              ]
            }
          
          input_mapping:
            requirements: $nodes.extract_requirements.outputs.requirements
            candidate: $item
          
          foreach: $nodes.categorize_candidates.outputs.shortlisted
      
      outputs:
        - name: interview_guides
          type: array
    
    # ============================================
    # 8. Output Node - Generate Report
    # ============================================
    - id: generate_report
      type: output
      depends_on: [categorize_candidates, generate_questions]
      description: Create final shortlisting report
      
      config:
        format: html
        template: shortlist_report.html
        
        data_mapping:
          job_title: $nodes.extract_requirements.outputs.requirements.title
          total_candidates: $nodes.upload_files.outputs.resume_files.length
          shortlisted: $nodes.categorize_candidates.outputs.shortlisted
          maybe_list: $nodes.categorize_candidates.outputs.maybe_list
          rejected: $nodes.categorize_candidates.outputs.rejected
          interview_guides: $nodes.generate_questions.outputs.interview_guides
          generated_at: $now
        
        delivery:
          - type: s3
            bucket: workflow-outputs
            key: "{execution_id}/shortlist_report.html"
            
          - type: email
            to: [hr@company.com]
            subject: "Resume Shortlist Report - {{job_title}}"
            template: email_notification.html
            
          - type: webhook
            url: https://api.company.com/recruitment/webhook
            method: POST
            headers:
              Authorization: Bearer ${SECRET_API_KEY}
      
      outputs:
        - name: report_url
          type: string
        - name: summary
          type: object
```

---

## 3. Workflow Engine Implementation

### 3.1 Core Engine Structure

```
workflow-engine/
├── engine/
│   ├── __init__.py
│   ├── parser.py           # YAML parser and validator
│   ├── dag.py              # DAG builder and executor
│   ├── context.py          # Execution context manager
│   ├── evaluator.py        # Expression evaluator ($nodes.x.y)
│   └── executor.py         # Main workflow executor
├── nodes/
│   ├── __init__.py
│   ├── base.py             # Base node class
│   ├── input_node.py
│   ├── agent_node.py
│   ├── batch_node.py
│   ├── parallel_node.py
│   ├── merge_node.py
│   ├── conditional_node.py
│   └── output_node.py
├── utils/
│   ├── __init__.py
│   ├── storage.py          # S3 operations
│   ├── ai_client.py        # Bedrock/OpenAI client
│   └── file_processor.py   # PDF/DOCX processing
├── schemas/
│   ├── workflow_schema.json
│   └── node_schemas/
│       ├── input.json
│       ├── agent.json
│       └── ...
└── templates/
    ├── shortlist_report.html
    └── email_notification.html
```

### 3.2 Workflow Parser

**`engine/parser.py`:**

```python
import yaml
import jsonschema
from typing import Dict, Any, List
from pathlib import Path

class WorkflowParser:
    """Parse and validate YAML workflows"""
    
    def __init__(self, schema_dir: str = "schemas"):
        self.schema_dir = Path(schema_dir)
        self.workflow_schema = self._load_schema("workflow_schema.json")
    
    def parse(self, yaml_file: str) -> Dict[str, Any]:
        """Parse YAML workflow file"""
        with open(yaml_file, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Validate against schema
        self.validate(workflow)
        
        return workflow
    
    def validate(self, workflow: Dict[str, Any]) -> None:
        """Validate workflow structure"""
        try:
            jsonschema.validate(workflow, self.workflow_schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid workflow: {e.message}")
        
        # Custom validations
        self._validate_dependencies(workflow)
        self._validate_output_references(workflow)
    
    def _validate_dependencies(self, workflow: Dict[str, Any]) -> None:
        """Ensure all node dependencies exist"""
        node_ids = {node['id'] for node in workflow['workflow']['nodes']}
        
        for node in workflow['workflow']['nodes']:
            depends_on = node.get('depends_on', [])
            for dep in depends_on:
                if dep not in node_ids:
                    raise ValueError(
                        f"Node '{node['id']}' depends on non-existent node '{dep}'"
                    )
    
    def _validate_output_references(self, workflow: Dict[str, Any]) -> None:
        """Validate $nodes.x.y references"""
        # Check that all referenced nodes exist
        # Check that referenced outputs are declared
        pass
    
    def _load_schema(self, filename: str) -> Dict[str, Any]:
        """Load JSON schema"""
        schema_path = self.schema_dir / filename
        with open(schema_path, 'r') as f:
            return json.load(f)
```

### 3.3 DAG Builder and Executor

**`engine/dag.py`:**

```python
from typing import Dict, Any, List, Set
from collections import defaultdict, deque
import networkx as nx

class WorkflowDAG:
    """Build and execute workflow DAG"""
    
    def __init__(self, workflow: Dict[str, Any]):
        self.workflow = workflow
        self.graph = nx.DiGraph()
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build directed acyclic graph from workflow"""
        nodes = self.workflow['workflow']['nodes']
        
        # Add nodes
        for node in nodes:
            self.graph.add_node(
                node['id'],
                type=node['type'],
                config=node.get('config', {}),
                outputs=node.get('outputs', [])
            )
        
        # Add edges (dependencies)
        for node in nodes:
            depends_on = node.get('depends_on', [])
            for dep in depends_on:
                self.graph.add_edge(dep, node['id'])
        
        # Validate DAG (no cycles)
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Workflow contains cycles")
    
    def get_execution_order(self) -> List[str]:
        """Get topological sort of nodes"""
        return list(nx.topological_sort(self.graph))
    
    def get_parallel_groups(self) -> List[List[str]]:
        """Group nodes that can execute in parallel"""
        execution_order = self.get_execution_order()
        groups = []
        processed = set()
        
        while processed != set(execution_order):
            # Find nodes whose dependencies are all processed
            group = []
            for node_id in execution_order:
                if node_id in processed:
                    continue
                
                deps = set(self.graph.predecessors(node_id))
                if deps.issubset(processed):
                    group.append(node_id)
            
            groups.append(group)
            processed.update(group)
        
        return groups
    
    def visualize(self, output_file: str = "workflow_dag.png") -> None:
        """Generate visual representation of DAG"""
        import matplotlib.pyplot as plt
        
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=1500,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20
        )
        plt.savefig(output_file)
        print(f"DAG visualization saved to {output_file}")
```

### 3.4 Expression Evaluator

**`engine/evaluator.py`:**

```python
import re
from typing import Any, Dict
from datetime import datetime

class ExpressionEvaluator:
    """Evaluate expressions like $nodes.x.y, $item, etc."""
    
    def __init__(self, context: Dict[str, Any]):
        self.context = context
    
    def evaluate(self, expression: str) -> Any:
        """Evaluate an expression"""
        if not isinstance(expression, str):
            return expression
        
        # Handle $now
        if expression == "$now":
            return datetime.utcnow().isoformat()
        
        # Handle $item (for foreach loops)
        if expression == "$item":
            return self.context.get('current_item')
        
        # Handle $nodes.x.y.z
        if expression.startswith("$nodes."):
            return self._eval_node_reference(expression)
        
        # Handle $env.VAR
        if expression.startswith("$env."):
            var_name = expression[5:]
            return os.getenv(var_name)
        
        # Regular string, return as-is
        return expression
    
    def _eval_node_reference(self, expression: str) -> Any:
        """Evaluate $nodes.node_id.outputs.field"""
        # Parse: $nodes.upload_files.outputs.resume_files
        parts = expression.split('.')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid node reference: {expression}")
        
        node_id = parts[1]
        output_path = parts[2:]  # e.g., ['outputs', 'resume_files']
        
        # Get node output from context
        node_outputs = self.context.get('nodes', {}).get(node_id, {})
        
        # Navigate path
        result = node_outputs
        for part in output_path:
            if isinstance(result, dict):
                result = result.get(part)
            elif isinstance(result, list):
                # Handle array access: outputs.items[0]
                match = re.match(r'(\w+)\[(\d+)\]', part)
                if match:
                    field, index = match.groups()
                    result = result[int(index)].get(field)
                else:
                    result = result.get(part)
            else:
                break
        
        return result
    
    def evaluate_condition(self, condition: str) -> bool:
        """Evaluate boolean condition"""
        # Replace references with actual values
        evaluated = condition
        
        # Find all $item.field references
        for match in re.finditer(r'\$item\.(\w+)', condition):
            field = match.group(1)
            value = self.context.get('current_item', {}).get(field)
            evaluated = evaluated.replace(match.group(0), repr(value))
        
        # Find all $nodes.x.y references
        for match in re.finditer(r'\$nodes\.[\w\.]+', condition):
            ref = match.group(0)
            value = self._eval_node_reference(ref)
            evaluated = evaluated.replace(ref, repr(value))
        
        # Safely evaluate boolean expression
        try:
            return eval(evaluated, {"__builtins__": {}}, {})
        except Exception as e:
            raise ValueError(f"Invalid condition '{condition}': {e}")
```

### 3.5 Main Workflow Executor

**`engine/executor.py`:**

```python
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import uuid

from .parser import WorkflowParser
from .dag import WorkflowDAG
from .context import ExecutionContext
from .evaluator import ExpressionEvaluator
from nodes import NodeFactory

class WorkflowExecutor:
    """Main workflow execution engine"""
    
    def __init__(self, workflow_file: str):
        self.parser = WorkflowParser()
        self.workflow = self.parser.parse(workflow_file)
        self.dag = WorkflowDAG(self.workflow)
        self.node_factory = NodeFactory()
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow"""
        execution_id = str(uuid.uuid4())
        
        # Initialize execution context
        context = ExecutionContext(
            execution_id=execution_id,
            workflow=self.workflow,
            input_data=input_data
        )
        
        print(f"Starting workflow execution: {execution_id}")
        print(f"Workflow: {self.workflow['workflow']['name']}")
        
        try:
            # Get execution order
            parallel_groups = self.dag.get_parallel_groups()
            
            # Execute nodes in topological order
            for group in parallel_groups:
                if len(group) == 1:
                    # Single node - execute sequentially
                    await self._execute_node(group[0], context)
                else:
                    # Multiple nodes - execute in parallel
                    await self._execute_parallel(group, context)
            
            # Mark completion
            context.status = 'completed'
            context.completed_at = datetime.utcnow()
            
            print(f"Workflow completed: {execution_id}")
            
            return context.get_results()
            
        except Exception as e:
            context.status = 'failed'
            context.error = str(e)
            print(f"Workflow failed: {e}")
            raise
    
    async def _execute_node(
        self, 
        node_id: str, 
        context: ExecutionContext
    ) -> None:
        """Execute a single node"""
        node_def = self._get_node_definition(node_id)
        
        print(f"  Executing node: {node_id} ({node_def['type']})")
        
        # Create node executor
        node = self.node_factory.create(node_def['type'], node_def, context)
        
        # Execute node
        start_time = datetime.utcnow()
        result = await node.execute()
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Store result in context
        context.set_node_output(node_id, result)
        
        print(f"  ✓ Node completed: {node_id} ({duration:.2f}s)")
    
    async def _execute_parallel(
        self,
        node_ids: List[str],
        context: ExecutionContext
    ) -> None:
        """Execute multiple nodes in parallel"""
        print(f"  Executing {len(node_ids)} nodes in parallel: {node_ids}")
        
        # Create tasks
        tasks = [
            self._execute_node(node_id, context)
            for node_id in node_ids
        ]
        
        # Execute all in parallel
        await asyncio.gather(*tasks)
    
    def _get_node_definition(self, node_id: str) -> Dict[str, Any]:
        """Get node definition from workflow"""
        for node in self.workflow['workflow']['nodes']:
            if node['id'] == node_id:
                return node
        raise ValueError(f"Node not found: {node_id}")
```

---

## 4. Node Executors

### 4.1 Base Node Class

**`nodes/base.py`:**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from engine.context import ExecutionContext
from engine.evaluator import ExpressionEvaluator

class BaseNode(ABC):
    """Base class for all node executors"""
    
    def __init__(
        self,
        node_def: Dict[str, Any],
        context: ExecutionContext
    ):
        self.node_def = node_def
        self.context = context
        self.config = node_def.get('config', {})
        self.evaluator = ExpressionEvaluator(context.data)
    
    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Execute the node and return outputs"""
        pass
    
    def resolve_input(self, input_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Resolve input values from context"""
        resolved = {}
        for key, expression in input_mapping.items():
            resolved[key] = self.evaluator.evaluate(expression)
        return resolved
    
    def log(self, message: str) -> None:
        """Log message"""
        print(f"    [{self.node_def['id']}] {message}")
```

### 4.2 Agent Node Executor

**`nodes/agent_node.py`:**

```python
from typing import Dict, Any
import json
import boto3
from .base import BaseNode

class AgentNode(BaseNode):
    """Execute AI agent node"""
    
    async def execute(self) -> Dict[str, Any]:
        """Execute agent"""
        agent_config = self.config['agent']
        
        # Resolve input mapping
        input_mapping = agent_config.get('input_mapping', {})
        inputs = self.resolve_input(input_mapping)
        
        # Build prompt from template
        prompt_template = agent_config['prompt_template']
        prompt = self._render_prompt(prompt_template, inputs)
        
        # Call AI model
        model_config = agent_config['model']
        response = await self._call_ai_model(prompt, model_config)
        
        # Parse response
        result = self._parse_response(response, agent_config.get('output_schema'))
        
        return result
    
    def _render_prompt(self, template: str, inputs: Dict[str, Any]) -> str:
        """Render prompt template with inputs"""
        import jinja2
        
        # Use Jinja2 for template rendering
        jinja_template = jinja2.Template(template)
        return jinja_template.render(**inputs)
    
    async def _call_ai_model(
        self,
        prompt: str,
        model_config: Dict[str, Any]
    ) -> str:
        """Call AI model (Bedrock, OpenAI, etc.)"""
        provider = model_config['provider']
        
        if provider == 'bedrock':
            return await self._call_bedrock(prompt, model_config)
        elif provider == 'openai':
            return await self._call_openai(prompt, model_config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _call_bedrock(
        self,
        prompt: str,
        model_config: Dict[str, Any]
    ) -> str:
        """Call AWS Bedrock"""
        bedrock = boto3.client('bedrock-runtime')
        
        response = bedrock.invoke_model(
            modelId=model_config['model_id'],
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": model_config.get('max_tokens', 2000),
                "temperature": model_config.get('temperature', 0.7),
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    
    def _parse_response(
        self,
        response: str,
        schema: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Parse AI response (extract JSON if needed)"""
        try:
            # Try direct JSON parse
            return json.loads(response)
        except:
            # Extract JSON from markdown code block
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
                return json.loads(json_str)
            
            # Return raw response
            return {"raw_response": response}
```

### 4.3 Parallel Node Executor

**`nodes/parallel_node.py`:**

```python
import asyncio
from typing import Dict, Any, List
from .base import BaseNode
from .agent_node import AgentNode

class ParallelNode(BaseNode):
    """Execute nodes in parallel (foreach pattern)"""
    
    async def execute(self) -> Dict[str, Any]:
        """Execute parallel processing"""
        # Get items to process
        foreach_expr = self.config.get('foreach')
        items = self.evaluator.evaluate(foreach_expr)
        
        if not isinstance(items, list):
            raise ValueError(f"foreach must evaluate to array, got {type(items)}")
        
        self.log(f"Processing {len(items)} items in parallel")
        
        # Get concurrency limit
        concurrency = self.config.get('concurrency', 5)
        
        # Process in batches
        results = []
        for i in range(0, len(items), concurrency):
            batch = items[i:i+concurrency]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
        
        return {'evaluations': results}
    
    async def _process_batch(self, items: List[Any]) -> List[Any]:
        """Process a batch of items in parallel"""
        # Create tasks
        tasks = [self._process_item(item) for item in items]
        
        # Execute in parallel
        return await asyncio.gather(*tasks)
    
    async def _process_item(self, item: Any) -> Any:
        """Process a single item"""
        # Update context with current item
        old_item = self.context.data.get('current_item')
        self.context.data['current_item'] = item
        
        try:
            # Execute agent for this item
            agent_config = self.config.get('agent', {})
            
            # Create temporary agent node
            temp_node_def = {
                'id': f"{self.node_def['id']}_item",
                'type': 'agent',
                'config': {'agent': agent_config}
            }
            
            agent = AgentNode(temp_node_def, self.context)
            result = await agent.execute()
            
            return result
            
        finally:
            # Restore context
            if old_item is not None:
                self.context.data['current_item'] = old_item
            else:
                self.context.data.pop('current_item', None)
```

---

## 5. Simple Example Workflow

**`workflows/simple-resume-analysis.yaml`:**

```yaml
workflow:
  name: simple_resume_analysis
  version: 1.0
  description: Basic resume analysis workflow
  
  nodes:
    - id: upload
      type: input
      config:
        fields:
          - name: resume
            type: file_upload
            accept: [pdf]
            required: true
      outputs:
        - name: resume_file
    
    - id: analyze
      type: agent
      depends_on: [upload]
      config:
        agent:
          model:
            provider: bedrock
            model_id: anthropic.claude-3-haiku-20240307-v1:0
          
          prompt_template: |
            Extract key information from this resume:
            {{resume_text}}
            
            Return JSON with: name, email, skills, experience_years
          
          input_mapping:
            resume_text: $nodes.upload.outputs.resume_file.text
      
      outputs:
        - name: analysis
    
    - id: output
      type: output
      depends_on: [analyze]
      config:
        format: json
        delivery:
          - type: s3
            bucket: workflow-outputs
            key: "{execution_id}/result.json"
```

---

## 6. Usage Example

**`examples/run_workflow.py`:**

```python
import asyncio
from engine.executor import WorkflowExecutor

async def main():
    # Initialize executor
    executor = WorkflowExecutor('workflows/resume-shortlisting.yaml')
    
    # Prepare input data
    input_data = {
        'resume_files': [
            {'path': 's3://bucket/resume1.pdf'},
            {'path': 's3://bucket/resume2.pdf'},
        ],
        'jd_file': {'path': 's3://bucket/jd.pdf'}
    }
    
    # Execute workflow
    results = await executor.execute(input_data)
    
    print("Workflow Results:")
    print(f"  Shortlisted: {len(results['shortlisted'])} candidates")
    print(f"  Report: {results['report_url']}")

if __name__ == '__main__':
    asyncio.run(main())
```