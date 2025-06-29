"""
Graph visualization utilities for the Veritas AI fact-checking engine.

This module provides functions to generate visual representations of the
LangGraph workflow for debugging and documentation purposes.
"""

from typing import Dict, Any, List, Optional
import json

from .graph import VeritasGraph, CompiledVeritasGraph


class GraphVisualizer:
    """Utilities for visualizing Veritas AI graphs."""
    
    @staticmethod
    def generate_mermaid(graph: VeritasGraph) -> str:
        """
        Generate a Mermaid diagram representation of the graph.
        
        Args:
            graph: VeritasGraph instance to visualize
        
        Returns:
            Mermaid diagram syntax as string
        """
        mermaid = "graph TD;\n"
        
        # Add START and END nodes
        mermaid += "    START([START]);\n"
        mermaid += "    END([END]);\n"
        
        # Add registered nodes
        for node_name, node_info in graph.nodes.items():
            safe_name = node_name.replace("-", "_").replace(".", "_")
            description = node_info.get("description", "")
            
            if description:
                mermaid += f"    {safe_name}[\"{node_name}<br/>{description}\"];\n"
            else:
                mermaid += f"    {safe_name}[{node_name}];\n"
        
        # Add direct edges
        for edge in graph.edges:
            from_safe = edge["from"].replace("-", "_").replace(".", "_")
            to_safe = edge["to"].replace("-", "_").replace(".", "_")
            
            if edge["from"] == "START":
                from_safe = "START"
            if edge["to"] == "END":
                to_safe = "END"
            
            mermaid += f"    {from_safe} --> {to_safe};\n"
        
        # Add conditional edges
        for edge in graph.conditional_edges:
            from_safe = edge["from"].replace("-", "_").replace(".", "_")
            
            # Add decision diamond for conditional logic
            decision_name = f"{from_safe}_decision"
            mermaid += f"    {decision_name}{{{edge['from']}<br/>Decision}};\n"
            mermaid += f"    {from_safe} --> {decision_name};\n"
            
            # Add paths from decision
            for condition, target in edge["path_map"].items():
                target_safe = target.replace("-", "_").replace(".", "_")
                if target == "END":
                    target_safe = "END"
                mermaid += f"    {decision_name} -->|{condition}| {target_safe};\n"
        
        return mermaid
    
    @staticmethod
    def generate_json_representation(graph: VeritasGraph) -> Dict[str, Any]:
        """
        Generate a JSON representation of the graph structure.
        
        Args:
            graph: VeritasGraph instance to represent
        
        Returns:
            Dictionary containing graph structure
        """
        return {
            "metadata": graph.metadata,
            "nodes": {
                name: {
                    "description": info.get("description"),
                    "registered_at": info.get("registered_at")
                }
                for name, info in graph.nodes.items()
            },
            "edges": graph.edges,
            "conditional_edges": [
                {
                    "from": edge["from"],
                    "path_map": edge["path_map"],
                    "description": edge.get("description"),
                    "added_at": edge.get("added_at")
                }
                for edge in graph.conditional_edges
            ]
        }
    
    @staticmethod
    def generate_execution_flow(
        compiled_graph: CompiledVeritasGraph,
        execution_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a visualization of an execution flow.
        
        Args:
            compiled_graph: Compiled graph with execution history
            execution_id: Specific execution to visualize (latest if None)
        
        Returns:
            Execution flow visualization data
        """
        history = compiled_graph.get_execution_history()
        if not history:
            return None
        
        if execution_id:
            execution = next(
                (exec for exec in history if exec["execution_id"] == execution_id),
                None
            )
        else:
            execution = history[-1]  # Latest execution
        
        if not execution:
            return None
        
        return {
            "execution_id": execution["execution_id"],
            "started_at": execution["started_at"],
            "completed_at": execution.get("completed_at"),
            "status": execution.get("status"),
            "initial_stage": execution.get("initial_stage"),
            "final_stage": execution.get("final_stage"),
            "error": execution.get("error")
        }
    
    @staticmethod
    def create_state_transition_diagram(states: List[str]) -> str:
        """
        Create a Mermaid diagram showing state transitions.
        
        Args:
            states: List of workflow states in order
        
        Returns:
            Mermaid diagram showing state flow
        """
        mermaid = "graph LR;\n"
        
        for i, state in enumerate(states):
            safe_state = state.replace("-", "_")
            mermaid += f"    {safe_state}[{state}];\n"
            
            if i < len(states) - 1:
                next_safe = states[i + 1].replace("-", "_")
                mermaid += f"    {safe_state} --> {next_safe};\n"
        
        return mermaid


def visualize_veritas_workflow() -> str:
    """
    Generate a standard visualization of the complete Veritas AI workflow.
    
    Returns:
        Mermaid diagram of the full fact-checking workflow
    """
    workflow_states = [
        "initialized",
        "content_extracted", 
        "claims_identified",
        "evidence_gathered",
        "response_drafted",
        "response_reviewed",
        "user_collaborating",
        "completed"
    ]
    
    mermaid = "graph TD;\n"
    mermaid += "    START([User Submits TikTok URL]);\n"
    
    # Add workflow states
    state_descriptions = {
        "initialized": "Initialize<br/>Investigation",
        "content_extracted": "Extract Content<br/>(Audio, Text, OCR)",
        "claims_identified": "Identify<br/>Factual Claims",
        "evidence_gathered": "Research Claims<br/>& Gather Evidence",
        "response_drafted": "Generate<br/>Draft Response",
        "response_reviewed": "Internal<br/>Quality Review",
        "user_collaborating": "Human-AI<br/>Collaboration",
        "completed": "Final Response<br/>Ready"
    }
    
    for state in workflow_states:
        safe_state = state.replace("-", "_")
        description = state_descriptions.get(state, state)
        mermaid += f"    {safe_state}[{description}];\n"
    
    # Add connections
    mermaid += "    START --> initialized;\n"
    for i, state in enumerate(workflow_states[:-1]):
        current_safe = state.replace("-", "_")
        next_safe = workflow_states[i + 1].replace("-", "_")
        mermaid += f"    {current_safe} --> {next_safe};\n"
    
    # Add revision loop
    mermaid += "    response_reviewed -->|Needs Revision| response_drafted;\n"
    mermaid += "    user_collaborating -->|User Feedback| response_drafted;\n"
    
    # Add end
    mermaid += "    completed --> END([Response Published]);\n"
    
    # Add error handling
    mermaid += "    initialized -->|Error| ERROR[Error State];\n"
    mermaid += "    content_extracted -->|Error| ERROR;\n"
    mermaid += "    claims_identified -->|Error| ERROR;\n"
    mermaid += "    evidence_gathered -->|Error| ERROR;\n"
    
    # Style the diagram
    mermaid += "\n    classDef startEnd fill:#e1f5fe;\n"
    mermaid += "    classDef process fill:#f3e5f5;\n"
    mermaid += "    classDef review fill:#fff3e0;\n"
    mermaid += "    classDef error fill:#ffebee;\n"
    
    mermaid += "    class START,END startEnd;\n"
    mermaid += "    class initialized,content_extracted,claims_identified,evidence_gathered process;\n"
    mermaid += "    class response_drafted,response_reviewed,user_collaborating,completed review;\n"
    mermaid += "    class ERROR error;\n"
    
    return mermaid


def export_graph_documentation(graph: VeritasGraph, filename: str) -> None:
    """
    Export comprehensive graph documentation to a file.
    
    Args:
        graph: VeritasGraph to document
        filename: Output filename (without extension)
    """
    docs = {
        "title": "Veritas AI Fact-Checking Graph",
        "generated_at": graph.metadata["created_at"],
        "metadata": graph.metadata,
        "mermaid_diagram": GraphVisualizer.generate_mermaid(graph),
        "json_structure": GraphVisualizer.generate_json_representation(graph),
        "workflow_diagram": visualize_veritas_workflow()
    }
    
    # Write as JSON
    with open(f"{filename}.json", "w") as f:
        json.dump(docs, f, indent=2)
    
    # Write Mermaid diagram
    with open(f"{filename}_diagram.md", "w") as f:
        f.write("# Veritas AI Graph Structure\n\n")
        f.write("```mermaid\n")
        f.write(docs["mermaid_diagram"])
        f.write("\n```\n\n")
        f.write("# Workflow Overview\n\n")
        f.write("```mermaid\n")
        f.write(docs["workflow_diagram"])
        f.write("\n```\n") 