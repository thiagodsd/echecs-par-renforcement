from typing import TypeVar, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from custom_types import DevOpsState

def create_devops_graph(agents: Dict[str, Any]) -> StateGraph:
    # Create the graph
    workflow = StateGraph(DevOpsState)
    
    # Define nodes
    
    # Analysis node
    def analyze_project(state: DevOpsState) -> DevOpsState:
        # Get response from the junior engineer for initial analysis
        response = agents["junior_web_software_engineer"].invoke({
            "context": state.context,
            "dev_context": state.ux_ui_context,
            "query": state.query
        })
        
        # Update state with analysis results
        state.result_analysis = response
        return state
    
    # Implementation node
    def implement_solutions(state: DevOpsState) -> DevOpsState:
        # Get implementation details from senior engineer
        response = agents["senior_web_software_engineer"].invoke({
            "context": state.context,
            "analysis": state.result_analysis,
            "query": state.query
        })
        
        # Update state with implementation results
        state.result_implementation = response
        return state
    
    # Review node
    def review_implementation(state: DevOpsState) -> Dict[str, Any]:
        # Review by frontend senior dev
        review = agents["front_end_senior_dev"].invoke({
            "implementation": state.result_implementation,
            "context": state.context
        })
        
        # Decide if we need another iteration
        if state.refine_count < 3 and "needs_refinement" in review:
            return {"should_refine": True}
        return {"should_refine": False}
    
    # Add nodes to graph
    workflow.add_node("analyze", analyze_project)
    workflow.add_node("implement", implement_solutions)
    workflow.add_node("review", review_implementation)
    
    # Define edges
    workflow.add_edge("analyze", "implement")
    workflow.add_edge("implement", "review")
    
    # Conditional edges based on review
    workflow.add_conditional_edges(
        "review",
        lambda x: "should_refine" if x.get("should_refine") else END
    )
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    return workflow.compile()