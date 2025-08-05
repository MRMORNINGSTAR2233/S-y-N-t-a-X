"""
Advanced Autonomous Agent for S-y-N-t-a-X AI CLI
Intelligent agent that can understand tasks, break them down, and execute them autonomously
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from ai_cli.llms.base import ChatMessage, LLMResponse
from ai_cli.llms.manager import LLMManager
from ai_cli.tools.file_operations import FileOperations
from ai_cli.memory.vector_store import VectorStore


class TaskType(Enum):
    """Types of tasks the agent can perform"""
    CREATE_FILE = "create_file"
    READ_FILE = "read_file"
    UPDATE_FILE = "update_file"
    DELETE_FILE = "delete_file"
    SEARCH_CODE = "search_code"
    GENERATE_CODE = "generate_code"
    ANALYZE_CODE = "analyze_code"
    EXPLAIN_CODE = "explain_code"
    DEBUG_CODE = "debug_code"
    REFACTOR_CODE = "refactor_code"
    CREATE_API = "create_api"
    CREATE_DATABASE = "create_database"
    RUN_COMMAND = "run_command"
    GENERAL_QUERY = "general_query"


@dataclass
class Task:
    """Represents a task to be executed by the agent"""
    task_type: TaskType
    description: str
    parameters: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ExecutionResult:
    """Result of task execution"""
    success: bool
    result: Any
    error_message: str = ""
    artifacts: List[str] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


class AdvancedAutonomousAgent:
    """Advanced AI agent that can understand and execute complex tasks autonomously"""
    
    def __init__(self, llm_manager: LLMManager, file_ops: FileOperations, 
                 vector_store: VectorStore, console: Console):
        self.llm_manager = llm_manager
        self.file_ops = file_ops
        self.vector_store = vector_store
        self.console = console
        self.current_project_path = Path.cwd()
        self.conversation_history = []
        self.execution_history = []
        
        # Enhanced system prompt for intelligent task understanding
        self.system_prompt = """You are an advanced AI coding assistant with the ability to understand, plan, and execute complex software development tasks autonomously.

Your capabilities include:
- Understanding natural language requests and breaking them into actionable tasks
- Creating, reading, updating, and deleting files
- Generating code in multiple programming languages
- Analyzing and debugging existing code
- Creating APIs, databases, and complete applications
- Executing system commands when needed

When given a task, you should:
1. Analyze the request and understand the goal
2. Break down complex tasks into smaller, manageable steps
3. Identify the specific actions needed (file operations, code generation, etc.)
4. Execute the steps in the correct order
5. Provide clear feedback on progress and results

Always respond with structured JSON when planning tasks, but provide natural language explanations for users.

Available task types:
- create_file: Create a new file with specified content
- read_file: Read and analyze existing files
- update_file: Modify existing files
- delete_file: Remove files or directories
- search_code: Search through codebase
- generate_code: Generate new code based on requirements
- analyze_code: Analyze code for issues, patterns, etc.
- explain_code: Explain how code works
- debug_code: Find and fix bugs in code
- refactor_code: Improve code structure and quality
- create_api: Create REST APIs or web services
- create_database: Set up database schemas and connections
- run_command: Execute system commands
- general_query: Answer questions or provide information

Be proactive, intelligent, and always aim to complete the user's request fully."""

    async def process_request(self, user_input: str) -> ExecutionResult:
        """Process a user request and execute the necessary tasks"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user", 
                "content": user_input,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Analyze the request and create execution plan
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Analyzing request...", total=None)
                
                execution_plan = await self._analyze_and_plan(user_input)
                
                if not execution_plan or not execution_plan.get('tasks'):
                    progress.stop()
                    return ExecutionResult(
                        success=False,
                        result="Could not understand the request or create execution plan",
                        error_message="Failed to parse request"
                    )
                
                progress.update(task, description="Executing tasks...")
                
                # Execute the plan
                result = await self._execute_plan(execution_plan, progress)
                
                progress.stop()
                return result
                
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                result="Request timed out",
                error_message="Operation timed out after 60 seconds"
            )
        except Exception as e:
            self.console.print(f"[red]Agent error: {e}[/red]")
            return ExecutionResult(
                success=False,
                result=f"Agent execution failed: {str(e)}",
                error_message=str(e)
            )

    async def _analyze_and_plan(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input and create a structured execution plan"""
        analysis_prompt = f"""
Analyze this user request and create a detailed execution plan:

User Request: "{user_input}"

Current working directory: {self.current_project_path}

Please provide a JSON response with the following structure:
{{
    "understanding": "Brief summary of what the user wants",
    "goal": "The main objective",
    "tasks": [
        {{
            "task_type": "create_file|read_file|update_file|delete_file|search_code|generate_code|analyze_code|explain_code|debug_code|refactor_code|create_api|create_database|run_command|general_query",
            "description": "What this task does",
            "parameters": {{
                "file_path": "path/to/file",
                "content": "file content or code",
                "language": "programming language",
                "search_query": "search terms",
                "command": "system command",
                "other_params": "as needed"
            }},
            "priority": 1-5,
            "dependencies": ["task_ids that must complete first"]
        }}
    ],
    "expected_outcome": "What should be achieved"
}}

Make sure to:
1. Break complex requests into smaller, actionable tasks
2. Identify the correct file paths and names
3. Determine the appropriate programming language
4. Consider dependencies between tasks
5. Be specific about what needs to be created or modified
"""
        
        try:
            # Use timeout to prevent hanging
            response = await asyncio.wait_for(
                self._get_llm_response(analysis_prompt),
                timeout=30.0
            )
            
            # Extract JSON from response
            plan = self._extract_json_from_response(response.content)
            return plan
            
        except asyncio.TimeoutError:
            self.console.print("[yellow]Analysis timed out, using fallback planning[/yellow]")
            return self._create_fallback_plan(user_input)
        except Exception as e:
            self.console.print(f"[yellow]Analysis failed: {e}, using fallback[/yellow]")
            return self._create_fallback_plan(user_input)

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text"""
        try:
            # Try to find JSON block in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find JSON without code blocks
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON found, create a simple plan
            return self._create_simple_plan(response_text)
            
        except json.JSONDecodeError:
            return self._create_simple_plan(response_text)

    def _create_fallback_plan(self, user_input: str) -> Dict[str, Any]:
        """Create a simple fallback plan when analysis fails"""
        # Analyze the input for common patterns
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['create', 'make', 'build', 'generate']):
            if 'api' in input_lower:
                return self._create_api_plan(user_input)
            elif any(ext in input_lower for ext in ['.py', 'python', 'fastapi', 'flask']):
                return self._create_python_file_plan(user_input)
            else:
                return self._create_general_creation_plan(user_input)
        elif any(word in input_lower for word in ['read', 'show', 'display', 'view']):
            return self._create_read_plan(user_input)
        elif any(word in input_lower for word in ['update', 'modify', 'change', 'edit']):
            return self._create_update_plan(user_input)
        elif any(word in input_lower for word in ['delete', 'remove', 'rm']):
            return self._create_delete_plan(user_input)
        else:
            return self._create_general_query_plan(user_input)

    def _create_api_plan(self, user_input: str) -> Dict[str, Any]:
        """Create a plan for API creation"""
        return {
            "understanding": "User wants to create a REST API",
            "goal": "Create a FastAPI application with authentication",
            "tasks": [
                {
                    "task_type": "create_file",
                    "description": "Create main FastAPI application file",
                    "parameters": {
                        "file_path": "main.py",
                        "language": "python",
                        "content_type": "fastapi_app"
                    },
                    "priority": 1,
                    "dependencies": []
                },
                {
                    "task_type": "create_file", 
                    "description": "Create authentication module",
                    "parameters": {
                        "file_path": "auth.py",
                        "language": "python",
                        "content_type": "auth_module"
                    },
                    "priority": 2,
                    "dependencies": ["main.py"]
                },
                {
                    "task_type": "create_file",
                    "description": "Create requirements file",
                    "parameters": {
                        "file_path": "requirements.txt",
                        "content_type": "requirements"
                    },
                    "priority": 3,
                    "dependencies": []
                }
            ],
            "expected_outcome": "A working FastAPI application with authentication"
        }

    def _create_python_file_plan(self, user_input: str) -> Dict[str, Any]:
        """Create a plan for Python file creation"""
        return {
            "understanding": "User wants to create a Python file",
            "goal": "Create a Python file based on requirements",
            "tasks": [
                {
                    "task_type": "create_file",
                    "description": "Create Python file",
                    "parameters": {
                        "file_path": "new_file.py",
                        "language": "python",
                        "content_type": "python_script"
                    },
                    "priority": 1,
                    "dependencies": []
                }
            ],
            "expected_outcome": "A Python file with the requested functionality"
        }

    def _create_general_creation_plan(self, user_input: str) -> Dict[str, Any]:
        """Create a general creation plan"""
        return {
            "understanding": "User wants to create something",
            "goal": "Create requested item",
            "tasks": [
                {
                    "task_type": "generate_code",
                    "description": "Generate code based on request",
                    "parameters": {
                        "description": user_input,
                        "language": "python"
                    },
                    "priority": 1,
                    "dependencies": []
                }
            ],
            "expected_outcome": "Created item as requested"
        }

    def _create_read_plan(self, user_input: str) -> Dict[str, Any]:
        """Create a plan for reading files"""
        return {
            "understanding": "User wants to read or view files",
            "goal": "Display file content",
            "tasks": [
                {
                    "task_type": "read_file",
                    "description": "Read and display file",
                    "parameters": {
                        "file_path": self._extract_file_path(user_input)
                    },
                    "priority": 1,
                    "dependencies": []
                }
            ],
            "expected_outcome": "File content displayed"
        }

    def _create_update_plan(self, user_input: str) -> Dict[str, Any]:
        """Create a plan for updating files"""
        return {
            "understanding": "User wants to update/modify files",
            "goal": "Update specified file",
            "tasks": [
                {
                    "task_type": "update_file",
                    "description": "Update file content",
                    "parameters": {
                        "file_path": self._extract_file_path(user_input),
                        "description": user_input
                    },
                    "priority": 1,
                    "dependencies": []
                }
            ],
            "expected_outcome": "File updated as requested"
        }

    def _create_delete_plan(self, user_input: str) -> Dict[str, Any]:
        """Create a plan for deleting files"""
        return {
            "understanding": "User wants to delete files",
            "goal": "Delete specified file",
            "tasks": [
                {
                    "task_type": "delete_file",
                    "description": "Delete file",
                    "parameters": {
                        "file_path": self._extract_file_path(user_input)
                    },
                    "priority": 1,
                    "dependencies": []
                }
            ],
            "expected_outcome": "File deleted"
        }

    def _create_general_query_plan(self, user_input: str) -> Dict[str, Any]:
        """Create a plan for general queries"""
        return {
            "understanding": "User has a general question or request",
            "goal": "Provide helpful response",
            "tasks": [
                {
                    "task_type": "general_query",
                    "description": "Answer user question",
                    "parameters": {
                        "query": user_input
                    },
                    "priority": 1,
                    "dependencies": []
                }
            ],
            "expected_outcome": "Helpful response provided"
        }

    def _create_simple_plan(self, response_text: str) -> Dict[str, Any]:
        """Create a simple plan from response text"""
        return {
            "understanding": "Process user request",
            "goal": "Complete the requested task",
            "tasks": [
                {
                    "task_type": "general_query",
                    "description": "Handle user request",
                    "parameters": {
                        "response": response_text
                    },
                    "priority": 1,
                    "dependencies": []
                }
            ],
            "expected_outcome": "Task completed"
        }

    def _extract_file_path(self, text: str) -> str:
        """Extract file path from text"""
        # Look for common file extensions
        patterns = [
            r'([a-zA-Z0-9_/\\.-]+\.py)',
            r'([a-zA-Z0-9_/\\.-]+\.js)',
            r'([a-zA-Z0-9_/\\.-]+\.ts)',
            r'([a-zA-Z0-9_/\\.-]+\.json)',
            r'([a-zA-Z0-9_/\\.-]+\.txt)',
            r'([a-zA-Z0-9_/\\.-]+\.md)',
            r'([a-zA-Z0-9_/\\.-]+\.[a-zA-Z]{2,4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return "file.py"  # Default fallback

    async def _execute_plan(self, plan: Dict[str, Any], progress: Progress) -> ExecutionResult:
        """Execute the planned tasks"""
        tasks = plan.get('tasks', [])
        results = []
        artifacts = []
        
        self.console.print(f"[blue]🎯 Goal:[/blue] {plan.get('goal', 'Complete user request')}")
        
        for i, task_data in enumerate(tasks):
            task_desc = task_data.get('description', f'Task {i+1}')
            progress.update(progress.task_ids[0], description=f"Executing: {task_desc}")
            
            try:
                result = await self._execute_task(task_data)
                results.append(result)
                
                if result.success:
                    self.console.print(f"[green]✓[/green] {task_desc}")
                    artifacts.extend(result.artifacts)
                else:
                    self.console.print(f"[red]❌[/red] {task_desc}: {result.error_message}")
                    
            except Exception as e:
                error_msg = f"Task execution failed: {str(e)}"
                self.console.print(f"[red]❌[/red] {task_desc}: {error_msg}")
                results.append(ExecutionResult(
                    success=False,
                    result=None,
                    error_message=error_msg
                ))
        
        # Determine overall success
        successful_tasks = [r for r in results if r.success]
        overall_success = len(successful_tasks) > 0
        
        if overall_success:
            self.console.print(f"[green]✅ Completed {len(successful_tasks)}/{len(tasks)} tasks successfully[/green]")
        else:
            self.console.print(f"[red]❌ All tasks failed[/red]")
        
        # Combine results
        combined_result = self._combine_results(results)
        
        return ExecutionResult(
            success=overall_success,
            result=combined_result,
            artifacts=artifacts
        )

    async def _execute_task(self, task_data: Dict[str, Any]) -> ExecutionResult:
        """Execute a single task"""
        task_type = task_data.get('task_type')
        parameters = task_data.get('parameters', {})
        
        try:
            if task_type == 'create_file':
                return await self._execute_create_file(parameters)
            elif task_type == 'read_file':
                return await self._execute_read_file(parameters)
            elif task_type == 'update_file':
                return await self._execute_update_file(parameters)
            elif task_type == 'delete_file':
                return await self._execute_delete_file(parameters)
            elif task_type == 'generate_code':
                return await self._execute_generate_code(parameters)
            elif task_type == 'search_code':
                return await self._execute_search_code(parameters)
            elif task_type == 'general_query':
                return await self._execute_general_query(parameters)
            else:
                return ExecutionResult(
                    success=False,
                    result=None,
                    error_message=f"Unknown task type: {task_type}"
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"Task execution error: {str(e)}"
            )

    async def _execute_create_file(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute file creation task"""
        file_path = parameters.get('file_path', 'new_file.py')
        content_type = parameters.get('content_type', 'python_script')
        language = parameters.get('language', 'python')
        
        # Generate content based on type
        if content_type == 'fastapi_app':
            content = self._generate_fastapi_content()
        elif content_type == 'auth_module':
            content = self._generate_auth_module_content()
        elif content_type == 'requirements':
            content = self._generate_requirements_content()
        else:
            # Generate content using LLM
            content = await self._generate_content_with_llm(parameters)
        
        # Create the file
        full_path = self.current_project_path / file_path
        success = self.file_ops.write_file(full_path, content)
        
        if success:
            return ExecutionResult(
                success=True,
                result=f"Created file: {file_path}",
                artifacts=[str(full_path)]
            )
        else:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"Failed to create file: {file_path}"
            )

    async def _execute_read_file(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute file reading task"""
        file_path = parameters.get('file_path', '')
        
        if not file_path:
            return ExecutionResult(
                success=False,
                result=None,
                error_message="No file path specified"
            )
        
        full_path = self.current_project_path / file_path
        
        if not full_path.exists():
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"File not found: {file_path}"
            )
        
        content = self.file_ops.read_file(full_path)
        
        if content is not None:
            # Display the file content
            self._display_file_content(full_path, content)
            return ExecutionResult(
                success=True,
                result=f"Read file: {file_path}",
                artifacts=[str(full_path)]
            )
        else:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"Could not read file: {file_path}"
            )

    async def _execute_update_file(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute file update task"""
        file_path = parameters.get('file_path', '')
        description = parameters.get('description', '')
        
        if not file_path:
            return ExecutionResult(
                success=False,
                result=None,
                error_message="No file path specified"
            )
        
        full_path = self.current_project_path / file_path
        
        if not full_path.exists():
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"File not found: {file_path}"
            )
        
        # Read current content
        current_content = self.file_ops.read_file(full_path)
        if current_content is None:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"Could not read file for updating: {file_path}"
            )
        
        # Generate updated content using LLM
        updated_content = await self._generate_updated_content(current_content, description, file_path)
        
        # Write updated content
        success = self.file_ops.write_file(full_path, updated_content, backup=True)
        
        if success:
            return ExecutionResult(
                success=True,
                result=f"Updated file: {file_path}",
                artifacts=[str(full_path)]
            )
        else:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"Failed to update file: {file_path}"
            )

    async def _execute_delete_file(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute file deletion task"""
        file_path = parameters.get('file_path', '')
        
        if not file_path:
            return ExecutionResult(
                success=False,
                result=None,
                error_message="No file path specified"
            )
        
        full_path = self.current_project_path / file_path
        
        if not full_path.exists():
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"File not found: {file_path}"
            )
        
        success = self.file_ops.delete_file(full_path)
        
        if success:
            return ExecutionResult(
                success=True,
                result=f"Deleted file: {file_path}",
                artifacts=[]
            )
        else:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"Failed to delete file: {file_path}"
            )

    async def _execute_generate_code(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute code generation task"""
        description = parameters.get('description', '')
        language = parameters.get('language', 'python')
        
        try:
            # Generate code using LLM
            code = await self._generate_code_with_llm(description, language)
            
            # Display the generated code
            self._display_generated_code(code, language)
            
            return ExecutionResult(
                success=True,
                result=f"Generated {language} code",
                artifacts=[]
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"Code generation failed: {str(e)}"
            )

    async def _execute_search_code(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute code search task"""
        query = parameters.get('search_query', parameters.get('query', ''))
        
        if not query:
            return ExecutionResult(
                success=False,
                result=None,
                error_message="No search query specified"
            )
        
        try:
            # Search using vector store
            results = self.vector_store.search_code(query, n_results=5)
            
            if results:
                self._display_search_results(query, results)
                return ExecutionResult(
                    success=True,
                    result=f"Found {len(results)} search results",
                    artifacts=[]
                )
            else:
                return ExecutionResult(
                    success=False,
                    result="No search results found",
                    error_message=""
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"Search failed: {str(e)}"
            )

    async def _execute_general_query(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute general query task"""
        query = parameters.get('query', parameters.get('response', ''))
        
        try:
            if 'response' in parameters:
                # Use pre-generated response
                response = parameters['response']
            else:
                # Generate response using LLM
                response_obj = await self._get_llm_response(query)
                response = response_obj.content
            
            # Display the response
            self._display_response(response)
            
            return ExecutionResult(
                success=True,
                result="Response provided",
                artifacts=[]
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"Query processing failed: {str(e)}"
            )

    async def _get_llm_response(self, prompt: str) -> LLMResponse:
        """Get response from LLM with timeout and error handling"""
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=prompt)
        ]
        
        # Show current configuration for debugging
        config = self.llm_manager.get_current_configuration()
        self.console.print(f"[dim]🤖 Using model: {config['default_model']} ({config['default_provider']})[/dim]")
        
        # First try the default model/provider (user's selected configuration)
        try:
            response = await asyncio.wait_for(
                self.llm_manager.chat_completion(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096
                ),
                timeout=30.0
            )
            return response
        except asyncio.TimeoutError:
            self.console.print(f"[yellow]Timeout with default model {self.llm_manager.default_model}, trying fallbacks...[/yellow]")
        except Exception as e:
            self.console.print(f"[yellow]Error with default model {self.llm_manager.default_model}: {e}, trying fallbacks...[/yellow]")
        
        # If default fails, try fallback providers in order of preference
        # Prioritize providers that are more likely to work well for autonomous tasks
        fallback_providers = ["gemini", "anthropic", "openai", "groq", "ollama"]
        available_providers = self.llm_manager.get_available_providers()
        
        # Filter to only available providers and maintain preference order
        providers_to_try = [p for p in fallback_providers if p in available_providers]
        
        for provider in providers_to_try:
            try:
                # Get the best model for this provider
                available_models = self.llm_manager.get_available_models(provider)
                if not available_models:
                    continue
                    
                model = available_models[0].name  # Use first available model for this provider
                
                response = await asyncio.wait_for(
                    self.llm_manager.chat_completion(
                        messages=messages,
                        model=model,
                        provider=provider,
                        temperature=0.7,
                        max_tokens=4096
                    ),
                    timeout=30.0
                )
                self.console.print(f"[blue]✓ Using fallback: {provider}/{model}[/blue]")
                return response
            except asyncio.TimeoutError:
                self.console.print(f"[yellow]Timeout with {provider}, trying next...[/yellow]")
                continue
            except Exception as e:
                self.console.print(f"[yellow]Error with {provider}: {e}, trying next...[/yellow]")
                continue
        
        # If all providers fail, return a fallback response
        return LLMResponse(
            content="I'm sorry, I'm having trouble processing your request right now. Please try again or use specific commands like /read, /write, /update, etc.",
            model="fallback",
            provider="system",
            tokens_used=0,
            cost_usd=0.0,
            function_calls=[],
            metadata={"error": "all_providers_failed"}
        )

    def _generate_fastapi_content(self) -> str:
        """Generate FastAPI application content"""
        return '''"""
FastAPI Application with Authentication
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(
    title="S-y-N-t-a-X API",
    description="REST API with authentication",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Models
class User(BaseModel):
    username: str
    email: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Mock user database
users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": "admin123"  # In production, use proper hashing
    }
}

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from token"""
    token = credentials.credentials
    # In production, properly validate JWT token
    if token == "valid-token":
        return users_db["admin"]
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "S-y-N-t-a-X API is running!", "status": "active"}

@app.post("/auth/login", response_model=Token)
async def login(login_data: LoginRequest):
    """Authenticate user and return token"""
    user = users_db.get(login_data.username)
    if not user or user["hashed_password"] != login_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    return {
        "access_token": "valid-token",  # In production, generate proper JWT
        "token_type": "bearer"
    }

@app.get("/auth/me", response_model=User)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return User(username=current_user["username"], email=current_user["email"])

@app.get("/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    """Protected route example"""
    return {"message": f"Hello {current_user['username']}, this is a protected route!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    def _generate_auth_module_content(self) -> str:
        """Generate authentication module content"""
        return '''"""
Authentication module for FastAPI application
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status

# Security configuration
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify access token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
'''

    def _generate_requirements_content(self) -> str:
        """Generate requirements.txt content"""
        return '''fastapi==0.104.1
uvicorn[standard]==0.24.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
pydantic==2.5.0
'''

    async def _generate_content_with_llm(self, parameters: Dict[str, Any]) -> str:
        """Generate file content using LLM"""
        language = parameters.get('language', 'python')
        description = parameters.get('description', 'Create a simple file')
        
        prompt = f"""
Generate {language} code for the following request:
{description}

Please provide clean, well-commented code that follows best practices.
Only return the code, no explanations or markdown formatting.
"""
        
        try:
            response = await self._get_llm_response(prompt)
            return response.content.strip()
        except Exception:
            # Fallback content
            if language == 'python':
                return '#!/usr/bin/env python3\n"""\nGenerated Python file\n"""\n\ndef main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()\n'
            else:
                return f'// Generated {language} file\nconsole.log("Hello, World!");\n'

    async def _generate_updated_content(self, current_content: str, description: str, file_path: str) -> str:
        """Generate updated file content using LLM"""
        prompt = f"""
Update the following code based on this request: {description}

Current file content ({file_path}):
```
{current_content}
```

Please provide the complete updated file content. Only return the code, no explanations.
"""
        
        try:
            response = await self._get_llm_response(prompt)
            return response.content.strip()
        except Exception:
            # Fallback: just append a comment
            return current_content + f'\n\n# Updated: {description}\n'

    async def _generate_code_with_llm(self, description: str, language: str) -> str:
        """Generate code using LLM"""
        prompt = f"""
Generate {language} code for: {description}

Please provide clean, well-commented code that follows best practices.
Only return the code, no explanations or markdown formatting.
"""
        
        try:
            response = await self._get_llm_response(prompt)
            return response.content.strip()
        except Exception:
            return f'// {language} code for: {description}\n// Code generation failed, please try again\n'

    def _display_file_content(self, file_path: Path, content: str):
        """Display file content with syntax highlighting"""
        # Determine language for syntax highlighting
        language = "text"
        if file_path.suffix:
            ext_to_lang = {
                '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
                '.css': 'css', '.html': 'html', '.xml': 'xml',
                '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml',
                '.md': 'markdown', '.sql': 'sql', '.sh': 'bash',
                '.go': 'go', '.rs': 'rust', '.php': 'php'
            }
            language = ext_to_lang.get(file_path.suffix.lower(), "text")
        
        # Display content with syntax highlighting
        content_panel = Panel(
            Syntax(content, language, theme="monokai", line_numbers=True),
            title=f"[bold green]📖 {file_path.name}[/bold green]",
            border_style="green"
        )
        self.console.print(content_panel)

    def _display_generated_code(self, code: str, language: str):
        """Display generated code with syntax highlighting"""
        code_panel = Panel(
            Syntax(code, language, theme="monokai", line_numbers=True),
            title=f"[bold blue]🔧 Generated {language.title()} Code[/bold blue]",
            border_style="blue"
        )
        self.console.print(code_panel)

    def _display_search_results(self, query: str, results: List[Dict[str, Any]]):
        """Display search results"""
        from rich.table import Table
        
        results_table = Table(title=f"🔍 Search Results for: {query}", box=None)
        results_table.add_column("File", style="cyan")
        results_table.add_column("Content", style="white")
        results_table.add_column("Score", style="green")
        
        for result in results:
            file_name = result.get('metadata', {}).get('file_path', 'Unknown')
            content = result.get('code', '')[:100] + "..." if len(result.get('code', '')) > 100 else result.get('code', '')
            similarity = f"{result.get('similarity', 0):.2f}"
            results_table.add_row(file_name, content, similarity)
        
        self.console.print(results_table)

    def _display_response(self, response: str):
        """Display AI response"""
        from rich.markdown import Markdown
        
        response_panel = Panel(
            Markdown(response),
            title="[bold blue]🤖 AI Response[/bold blue]",
            border_style="blue"
        )
        self.console.print(response_panel)

    def _combine_results(self, results: List[ExecutionResult]) -> str:
        """Combine multiple execution results into a summary"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        summary = []
        
        if successful:
            summary.append(f"✅ {len(successful)} tasks completed successfully:")
            for result in successful:
                summary.append(f"  • {result.result}")
        
        if failed:
            summary.append(f"❌ {len(failed)} tasks failed:")
            for result in failed:
                summary.append(f"  • {result.error_message}")
        
        return "\n".join(summary)
