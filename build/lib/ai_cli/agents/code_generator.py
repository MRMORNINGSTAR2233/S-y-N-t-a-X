"""Code generation agent using LangGraph for agentic workflows."""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, SystemMessage

from ai_cli.llms.manager import LLMManager, ChatMessage
from ai_cli.memory.vector_store import VectorStore
from ai_cli.tools.git_integration import GitManager
from ai_cli.tools.file_operations import FileOperations
from ai_cli.tools.code_analysis import CodeAnalyzer


@dataclass
class CodeGenerationState:
    """State for code generation workflow."""
    description: str
    target_file: Optional[str] = None
    language: Optional[str] = None
    framework: Optional[str] = None
    existing_code: str = ""
    generated_code: str = ""
    analysis_results: Dict[str, Any] = None
    context: Dict[str, Any] = None
    errors: List[str] = None
    final_result: Dict[str, Any] = None


class CodeGeneratorAgent:
    """Agent for generating code from natural language descriptions."""
    
    def __init__(self, llm_manager: LLMManager, vector_store: VectorStore, git_manager: GitManager):
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.git_manager = git_manager
        self.file_ops = FileOperations()
        self.code_analyzer = CodeAnalyzer()
        
        # Create the agent workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for code generation."""
        
        workflow = StateGraph(CodeGenerationState)
        
        # Add nodes
        workflow.add_node("analyze_context", self._analyze_context)
        workflow.add_node("gather_context", self._gather_context)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("validate_code", self._validate_code)
        workflow.add_node("refine_code", self._refine_code)
        workflow.add_node("finalize", self._finalize)
        
        # Add edges
        workflow.add_edge("analyze_context", "gather_context")
        workflow.add_edge("gather_context", "generate_code")
        workflow.add_edge("generate_code", "validate_code")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "validate_code",
            self._should_refine,
            {
                "refine": "refine_code",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("refine_code", "validate_code")
        workflow.add_edge("finalize", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_context")
        
        return workflow.compile()
    
    async def _analyze_context(self, state: CodeGenerationState) -> CodeGenerationState:
        """Analyze the request context and determine approach."""
        
        # Detect language if not specified
        if not state.language and state.target_file:
            state.language = self.code_analyzer.detect_language(state.target_file)
        
        # Load existing file if specified
        if state.target_file and Path(state.target_file).exists():
            state.existing_code = self.file_ops.read_file(state.target_file)
        
        # Initialize context
        state.context = {
            "project_type": self.code_analyzer.detect_project_type(),
            "existing_patterns": [],
            "dependencies": [],
            "style_guide": {}
        }
        
        return state
    
    async def _gather_context(self, state: CodeGenerationState) -> CodeGenerationState:
        """Gather relevant context from codebase and memory."""
        
        # Search for similar code patterns
        similar_code = self.vector_store.search_code(
            query=state.description,
            language=state.language,
            n_results=5
        )
        
        # Search for relevant project context
        project_context = self.vector_store.search_project_context(
            query=state.description,
            n_results=3
        )
        
        # Analyze existing codebase patterns
        if state.target_file:
            codebase_analysis = self.code_analyzer.analyze_codebase_patterns(
                Path(state.target_file).parent
            )
            state.context.update(codebase_analysis)
        
        # Update context with findings
        state.context["similar_code"] = similar_code
        state.context["project_context"] = project_context
        
        return state
    
    async def _generate_code(self, state: CodeGenerationState) -> CodeGenerationState:
        """Generate code based on description and context."""
        
        # Build system prompt
        system_prompt = self._build_system_prompt(state)
        
        # Build user prompt
        user_prompt = self._build_user_prompt(state)
        
        # Get best model for code generation
        model = await self.llm_manager.get_best_model_for_task("code_generation")
        
        # Generate code
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        response = await self.llm_manager.chat_completion(
            messages=messages,
            model=model,
            temperature=0.2  # Lower temperature for more deterministic code
        )
        
        state.generated_code = self._extract_code_from_response(response.content)
        
        return state
    
    async def _validate_code(self, state: CodeGenerationState) -> CodeGenerationState:
        """Validate the generated code for syntax and logic errors."""
        
        validation_results = self.code_analyzer.validate_code(
            code=state.generated_code,
            language=state.language,
            existing_code=state.existing_code
        )
        
        state.analysis_results = validation_results
        state.errors = validation_results.get("errors", [])
        
        return state
    
    def _should_refine(self, state: CodeGenerationState) -> str:
        """Decide whether to refine the code or finalize."""
        
        if state.errors and len(state.errors) > 0:
            return "refine"
        
        # Check code quality metrics
        if state.analysis_results:
            quality_score = state.analysis_results.get("quality_score", 0.8)
            if quality_score < 0.7:
                return "refine"
        
        return "finalize"
    
    async def _refine_code(self, state: CodeGenerationState) -> CodeGenerationState:
        """Refine the generated code based on validation feedback."""
        
        # Build refinement prompt
        refinement_prompt = self._build_refinement_prompt(state)
        
        model = await self.llm_manager.get_best_model_for_task("debugging")
        
        messages = [
            ChatMessage(role="system", content="You are an expert code reviewer and debugger."),
            ChatMessage(role="user", content=refinement_prompt)
        ]
        
        response = await self.llm_manager.chat_completion(
            messages=messages,
            model=model,
            temperature=0.1
        )
        
        state.generated_code = self._extract_code_from_response(response.content)
        
        return state
    
    async def _finalize(self, state: CodeGenerationState) -> CodeGenerationState:
        """Finalize the code generation process."""
        
        state.final_result = {
            "success": True,
            "generated_code": state.generated_code,
            "target_file": state.target_file,
            "language": state.language,
            "analysis": state.analysis_results,
            "context_used": len(state.context.get("similar_code", [])),
        }
        
        # Store the generated code in vector memory for future reference
        if state.generated_code and state.target_file:
            await self._store_in_memory(state)
        
        return state
    
    def _build_system_prompt(self, state: CodeGenerationState) -> str:
        """Build the system prompt for code generation."""
        
        base_prompt = f"""You are an expert software engineer specializing in {state.language or 'multiple languages'}.
Your task is to generate high-quality, production-ready code based on natural language descriptions.

Guidelines:
- Write clean, readable, and maintainable code
- Follow best practices and conventions for {state.language or 'the target language'}
- Include appropriate error handling
- Add clear comments for complex logic
- Ensure code is secure and efficient
- Make code testable and modular"""

        if state.framework:
            base_prompt += f"\n- Use {state.framework} framework conventions and patterns"
        
        if state.context.get("style_guide"):
            base_prompt += f"\n- Follow the project's coding style: {state.context['style_guide']}"
        
        return base_prompt
    
    def _build_user_prompt(self, state: CodeGenerationState) -> str:
        """Build the user prompt with context and requirements."""
        
        prompt_parts = [
            f"Generate code for: {state.description}",
        ]
        
        if state.target_file:
            prompt_parts.append(f"Target file: {state.target_file}")
        
        if state.existing_code:
            prompt_parts.append(f"Existing code in file:\n```{state.language}\n{state.existing_code}\n```")
        
        # Add similar code examples
        similar_code = state.context.get("similar_code", [])
        if similar_code:
            prompt_parts.append("Similar code patterns in the codebase:")
            for i, code_match in enumerate(similar_code[:2]):  # Limit to top 2
                prompt_parts.append(f"Example {i+1}:\n```{state.language}\n{code_match['code']}\n```")
        
        # Add project context
        project_context = state.context.get("project_context", [])
        if project_context:
            prompt_parts.append("Project context:")
            for context in project_context[:2]:
                prompt_parts.append(f"- {context['content']}")
        
        prompt_parts.append(f"\nGenerate the {state.language or 'appropriate'} code. Return only the code without explanations.")
        
        return "\n\n".join(prompt_parts)
    
    def _build_refinement_prompt(self, state: CodeGenerationState) -> str:
        """Build prompt for code refinement."""
        
        prompt_parts = [
            "The following code has issues that need to be fixed:",
            f"```{state.language}\n{state.generated_code}\n```",
            "",
            "Issues found:"
        ]
        
        for error in state.errors:
            prompt_parts.append(f"- {error}")
        
        if state.analysis_results:
            suggestions = state.analysis_results.get("suggestions", [])
            if suggestions:
                prompt_parts.append("\nSuggestions for improvement:")
                for suggestion in suggestions:
                    prompt_parts.append(f"- {suggestion}")
        
        prompt_parts.append(f"\nPlease fix these issues and return the corrected {state.language} code.")
        
        return "\n".join(prompt_parts)
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response, removing markdown and explanations."""
        
        # Remove markdown code blocks
        lines = response.split('\n')
        in_code_block = False
        code_lines = []
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # If no code blocks found, return the whole response
        return response.strip()
    
    async def _store_in_memory(self, state: CodeGenerationState) -> None:
        """Store generated code in vector memory."""
        
        self.vector_store.add_code_snippet(
            code=state.generated_code,
            file_path=state.target_file,
            language=state.language,
            description=state.description
        )
        
        # Store the generation context
        self.vector_store.add_project_context(
            content=f"Generated: {state.description}",
            context_type="code_generation",
            file_paths=[state.target_file] if state.target_file else [],
            tags=["generated", state.language or "unknown"]
        )
    
    async def generate_feature(self, description: str, target_file: Optional[str] = None,
                             language: Optional[str] = None, framework: Optional[str] = None,
                             dry_run: bool = False) -> Dict[str, Any]:
        """Generate a feature from description."""
        
        initial_state = CodeGenerationState(
            description=description,
            target_file=target_file,
            language=language,
            framework=framework
        )
        
        # Run the workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        # Apply changes if not dry run
        if not dry_run and final_state.final_result["success"] and target_file:
            success = self.file_ops.write_file(target_file, final_state.generated_code)
            final_state.final_result["file_written"] = success
        
        return final_state.final_result
    
    async def edit_code(self, instruction: str, target_file: Optional[str] = None,
                       scope: Optional[str] = None) -> Dict[str, Any]:
        """Edit existing code with natural language instructions."""
        
        if not target_file or not Path(target_file).exists():
            return {"success": False, "error": "Target file not found"}
        
        # Read existing code
        existing_code = self.file_ops.read_file(target_file)
        language = self.code_analyzer.detect_language(target_file)
        
        # Create edit-specific prompt
        edit_description = f"Edit the existing code to: {instruction}"
        if scope:
            edit_description += f" (scope: {scope})"
        
        # Use the same workflow but with edit context
        initial_state = CodeGenerationState(
            description=edit_description,
            target_file=target_file,
            language=language,
            existing_code=existing_code
        )
        
        final_state = await self.workflow.ainvoke(initial_state)
        
        return final_state.final_result
