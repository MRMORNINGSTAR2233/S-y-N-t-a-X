"""Debugger agent for analyzing and fixing code issues."""

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
class DebugState:
    """State for debugging workflow."""
    target: str
    trace_file: Optional[str] = None
    context_files: List[str] = None
    error_analysis: Dict[str, Any] = None
    similar_errors: List[Dict[str, Any]] = None
    fix_suggestions: List[Dict[str, Any]] = None
    applied_fixes: List[str] = None
    final_result: Dict[str, Any] = None


class DebuggerAgent:
    """Agent for debugging and fixing code issues."""
    
    def __init__(self, llm_manager: LLMManager, vector_store: VectorStore, git_manager: GitManager):
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.git_manager = git_manager
        self.file_ops = FileOperations()
        self.code_analyzer = CodeAnalyzer()
        
        # Create the debugging workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for debugging."""
        
        workflow = StateGraph(DebugState)
        
        # Add nodes
        workflow.add_node("analyze_error", self._analyze_error)
        workflow.add_node("search_similar_errors", self._search_similar_errors)
        workflow.add_node("generate_fixes", self._generate_fixes)
        workflow.add_node("validate_fixes", self._validate_fixes)
        workflow.add_node("apply_fixes", self._apply_fixes)
        workflow.add_node("finalize", self._finalize)
        
        # Add edges
        workflow.add_edge("analyze_error", "search_similar_errors")
        workflow.add_edge("search_similar_errors", "generate_fixes")
        workflow.add_edge("generate_fixes", "validate_fixes")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "validate_fixes",
            self._should_apply_fixes,
            {
                "apply": "apply_fixes",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("apply_fixes", "finalize")
        workflow.add_edge("finalize", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_error")
        
        return workflow.compile()
    
    async def _analyze_error(self, state: DebugState) -> DebugState:
        """Analyze the error or issue in the target."""
        
        error_info = {
            'type': 'unknown',
            'description': '',
            'file_path': state.target,
            'line_number': None,
            'stack_trace': '',
            'error_message': '',
            'context': {}
        }
        
        # If target is a file, analyze it
        if Path(state.target).exists() and Path(state.target).is_file():
            # Read the file
            content = self.file_ops.read_file(state.target)
            if content:
                # Analyze for syntax errors
                language = self.code_analyzer.detect_language(state.target)
                validation = self.code_analyzer.validate_code(content, language or 'text')
                
                error_info.update({
                    'type': 'syntax_error' if not validation.get('syntax_valid', True) else 'code_analysis',
                    'description': 'Code analysis and potential issues',
                    'syntax_errors': validation.get('errors', []),
                    'warnings': validation.get('warnings', []),
                    'suggestions': validation.get('suggestions', []),
                    'quality_score': validation.get('quality_score', 0.8)
                })
        
        # If trace file is provided, analyze it
        if state.trace_file and Path(state.trace_file).exists():
            trace_content = self.file_ops.read_file(state.trace_file)
            if trace_content:
                error_info.update(self._parse_error_trace(trace_content))
        
        # Analyze git context
        git_info = self.git_manager.get_file_status(state.target)
        error_info['git_status'] = git_info
        
        state.error_analysis = error_info
        return state
    
    def _parse_error_trace(self, trace_content: str) -> Dict[str, Any]:
        """Parse error trace to extract useful information."""
        trace_info = {
            'stack_trace': trace_content,
            'error_message': '',
            'file_references': [],
            'line_numbers': []
        }
        
        lines = trace_content.split('\n')
        
        # Extract error message (usually the last non-empty line)
        for line in reversed(lines):
            if line.strip():
                trace_info['error_message'] = line.strip()
                break
        
        # Extract file references and line numbers
        import re
        file_pattern = r'File "([^"]+)", line (\d+)'
        for line in lines:
            match = re.search(file_pattern, line)
            if match:
                file_path, line_no = match.groups()
                trace_info['file_references'].append(file_path)
                trace_info['line_numbers'].append(int(line_no))
        
        return trace_info
    
    async def _search_similar_errors(self, state: DebugState) -> DebugState:
        """Search for similar errors in vector memory."""
        
        # Build search query from error analysis
        search_query = state.error_analysis.get('error_message', '')
        if not search_query:
            # Use description and type as fallback
            search_query = f"{state.error_analysis.get('type', '')} {state.error_analysis.get('description', '')}"
        
        # Search for similar error patterns
        similar_errors = self.vector_store.search_error_patterns(
            error_query=search_query,
            language=self.code_analyzer.detect_language(state.target),
            n_results=5
        )
        
        state.similar_errors = similar_errors
        return state
    
    async def _generate_fixes(self, state: DebugState) -> DebugState:
        """Generate fix suggestions using LLM."""
        
        # Build comprehensive prompt
        system_prompt = self._build_debug_system_prompt(state)
        user_prompt = self._build_debug_user_prompt(state)
        
        # Get best model for debugging
        model = await self.llm_manager.get_best_model_for_task("debugging")
        
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        response = await self.llm_manager.chat_completion(
            messages=messages,
            model=model,
            temperature=0.1  # Low temperature for precise debugging
        )
        
        # Parse fix suggestions from response
        fix_suggestions = self._parse_fix_suggestions(response.content)
        state.fix_suggestions = fix_suggestions
        
        return state
    
    def _build_debug_system_prompt(self, state: DebugState) -> str:
        """Build system prompt for debugging."""
        
        language = self.code_analyzer.detect_language(state.target) or 'unknown'
        
        return f"""You are an expert debugging assistant specializing in {language} code.
Your task is to analyze errors, understand their root causes, and provide precise fix suggestions.

Guidelines:
- Analyze the error thoroughly before suggesting fixes
- Provide step-by-step explanations for each fix
- Consider edge cases and potential side effects
- Suggest preventive measures to avoid similar issues
- Prioritize fixes by impact and safety
- Include code examples for complex fixes
- Maintain code style and project conventions"""
    
    def _build_debug_user_prompt(self, state: DebugState) -> str:
        """Build user prompt for debugging."""
        
        prompt_parts = [
            f"Please analyze and fix the following issue in: {state.target}",
            "",
            "Error Analysis:"
        ]
        
        error_analysis = state.error_analysis
        prompt_parts.append(f"Type: {error_analysis.get('type', 'Unknown')}")
        prompt_parts.append(f"Description: {error_analysis.get('description', 'No description')}")
        
        if error_analysis.get('error_message'):
            prompt_parts.append(f"Error Message: {error_analysis['error_message']}")
        
        if error_analysis.get('syntax_errors'):
            prompt_parts.append("Syntax Errors:")
            for error in error_analysis['syntax_errors']:
                prompt_parts.append(f"  - {error}")
        
        if error_analysis.get('stack_trace'):
            prompt_parts.append(f"Stack Trace:\n{error_analysis['stack_trace']}")
        
        # Add file content
        if Path(state.target).exists():
            content = self.file_ops.read_file(state.target)
            if content:
                language = self.code_analyzer.detect_language(state.target) or 'text'
                prompt_parts.append(f"File Content:\n```{language}\n{content}\n```")
        
        # Add similar errors if found
        if state.similar_errors:
            prompt_parts.append("Similar errors found in codebase:")
            for i, similar in enumerate(state.similar_errors[:2], 1):
                prompt_parts.append(f"{i}. {similar['metadata'].get('solution', 'No solution available')}")
        
        # Add context files if provided
        if state.context_files:
            prompt_parts.append("Additional context files:")
            for context_file in state.context_files[:3]:  # Limit context
                if Path(context_file).exists():
                    context_content = self.file_ops.read_file(context_file)
                    if context_content:
                        prompt_parts.append(f"\n{context_file}:\n```\n{context_content[:1000]}...\n```")
        
        prompt_parts.append("\nPlease provide:")
        prompt_parts.append("1. Root cause analysis")
        prompt_parts.append("2. Step-by-step fix suggestions")
        prompt_parts.append("3. Updated code if applicable")
        prompt_parts.append("4. Prevention recommendations")
        
        return "\n".join(prompt_parts)
    
    def _parse_fix_suggestions(self, response: str) -> List[Dict[str, Any]]:
        """Parse fix suggestions from LLM response."""
        
        suggestions = []
        
        # Split response into sections
        sections = response.split('\n\n')
        
        current_suggestion = {
            'title': 'General Fix',
            'description': '',
            'code_changes': '',
            'priority': 'medium',
            'explanation': ''
        }
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Look for numbered fixes or bullet points
            if section.startswith(('1.', '2.', '3.', '-', '*')):
                if current_suggestion['description']:
                    suggestions.append(current_suggestion.copy())
                
                current_suggestion = {
                    'title': section[:50] + '...' if len(section) > 50 else section,
                    'description': section,
                    'code_changes': '',
                    'priority': 'medium',
                    'explanation': ''
                }
            
            # Look for code blocks
            elif '```' in section:
                current_suggestion['code_changes'] = section
            
            else:
                current_suggestion['explanation'] += section + '\n'
        
        # Add the last suggestion
        if current_suggestion['description']:
            suggestions.append(current_suggestion)
        
        return suggestions
    
    async def _validate_fixes(self, state: DebugState) -> DebugState:
        """Validate the proposed fixes."""
        
        if not state.fix_suggestions:
            return state
        
        # For each fix suggestion, try to validate if it would work
        validated_fixes = []
        
        for fix in state.fix_suggestions:
            validation = {
                'fix': fix,
                'is_safe': True,
                'potential_issues': [],
                'confidence': 0.8
            }
            
            # Basic validation checks
            code_changes = fix.get('code_changes', '')
            if code_changes:
                # Extract code from markdown blocks
                code = self._extract_code_from_markdown(code_changes)
                if code:
                    language = self.code_analyzer.detect_language(state.target)
                    if language:
                        code_validation = self.code_analyzer.validate_code(code, language)
                        if not code_validation.get('syntax_valid', True):
                            validation['is_safe'] = False
                            validation['potential_issues'].extend(code_validation.get('errors', []))
                            validation['confidence'] *= 0.5
            
            validated_fixes.append(validation)
        
        state.fix_suggestions = [v['fix'] for v in validated_fixes if v['is_safe']]
        return state
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        import re
        
        # Find code blocks
        code_pattern = r'```(?:\w+)?\n(.*?)\n```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return ""
    
    def _should_apply_fixes(self, state: DebugState) -> str:
        """Decide whether to apply fixes automatically."""
        
        # Only apply if we have validated fixes and they're safe
        if state.fix_suggestions and len(state.fix_suggestions) > 0:
            return "apply"
        
        return "finalize"
    
    async def _apply_fixes(self, state: DebugState) -> DebugState:
        """Apply the validated fixes."""
        
        applied_fixes = []
        
        for fix in state.fix_suggestions:
            try:
                # Extract and apply code changes
                code_changes = fix.get('code_changes', '')
                if code_changes and Path(state.target).exists():
                    new_code = self._extract_code_from_markdown(code_changes)
                    if new_code:
                        # Create backup
                        backup_path = f"{state.target}.debug_backup"
                        self.file_ops.copy_file(state.target, backup_path)
                        
                        # Apply fix
                        success = self.file_ops.write_file(state.target, new_code)
                        if success:
                            applied_fixes.append(fix['title'])
                        else:
                            # Restore backup if write failed
                            self.file_ops.copy_file(backup_path, state.target)
                
            except Exception as e:
                print(f"Error applying fix: {e}")
        
        state.applied_fixes = applied_fixes
        return state
    
    async def _finalize(self, state: DebugState) -> DebugState:
        """Finalize the debugging process."""
        
        state.final_result = {
            'success': True,
            'target': state.target,
            'error_type': state.error_analysis.get('type', 'unknown'),
            'fixes_suggested': len(state.fix_suggestions or []),
            'fixes_applied': len(state.applied_fixes or []),
            'similar_errors_found': len(state.similar_errors or []),
            'analysis': state.error_analysis,
            'suggestions': state.fix_suggestions
        }
        
        # Store error pattern in memory for future reference
        if state.error_analysis and state.fix_suggestions:
            await self._store_error_pattern(state)
        
        return state
    
    async def _store_error_pattern(self, state: DebugState) -> None:
        """Store the error pattern and solution in vector memory."""
        
        error_message = state.error_analysis.get('error_message', 'Unknown error')
        
        # Create solution summary
        solutions = []
        for fix in state.fix_suggestions:
            solutions.append(fix.get('description', fix.get('title', 'Fix applied')))
        
        solution_text = "\n".join(solutions)
        
        self.vector_store.add_error_pattern(
            error_message=error_message,
            solution=solution_text,
            language=self.code_analyzer.detect_language(state.target) or 'unknown',
            error_type=state.error_analysis.get('type', 'unknown'),
            file_path=state.target
        )
    
    async def debug_issue(self, target: str, trace_file: Optional[str] = None,
                         context_files: Optional[List[str]] = None,
                         apply_fixes: bool = False) -> Dict[str, Any]:
        """Debug an issue in the target file or trace."""
        
        initial_state = DebugState(
            target=target,
            trace_file=trace_file,
            context_files=context_files or []
        )
        
        # Run the debugging workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        return final_state.final_result
