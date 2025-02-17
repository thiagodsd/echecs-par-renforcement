from typing import List, Optional
from pydantic import BaseModel

class FileAugmentedContext(BaseModel):
    filename: str = "The complete path of the file"
    content: str = "A copy of the file content"

class RetrieveAugmentedContext(BaseModel):
    filename: str
    content: str
    summary: str

class AugmentedContent(BaseModel):
    file: List[FileAugmentedContext]
    rag: List[RetrieveAugmentedContext]

class CodeImplementation(BaseModel):
    title: str
    files: List[str]
    current_state: str
    action_needed: str
    reasoning: str
    full_pseudo_code: str

class ListCodeImplementation(BaseModel):
    code_implementations: List[CodeImplementation]

class DevOpsState(BaseModel):
    """State management for the DevOps workflow"""
    refine_count: int = 0
    result_analysis: str = ""
    result_implementation: str = ""
    context: str = ""
    ux_ui_context: str = ""
    dev_context: str = ""
    query: str = ""