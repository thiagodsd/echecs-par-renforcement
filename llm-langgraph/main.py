from typing import TypeVar, Dict, Any
from langchain_core.messages import BaseMessage
from graph import create_devops_graph
from agents import setup_agents
from custom_types import DevOpsState

def main():
    # Initialize agents
    agents = setup_agents()
    
    # Create the graph
    graph = create_devops_graph(agents)
    
    # Initial state
    initial_state = DevOpsState(
        refine_count=0,
        result_analysis="",
        result_implementation="",
        context="""
# PROJECT CONTEXT
**Project Concept**
A platform to help divorced parents organize their children's lives, featuring a shared calendar, expenses, tasks, etc. It is expected that users will prefer to use the platform on their mobile devices.
**Status**
Under development, single-developer project, aiming to launch beta version next month.
**Tech Stack**
NextJS 14, TypeScript, DaisyUI, Framer Motion, Firebase, Firestore, and GCP services when needed.
""",
        ux_ui_context="""
## UX/UI GUIDELINES
Regarding the UX/UI, the following guidelines should be followed:
- **Responsiveness**: the platform must be responsive and work well on mobile devices.
- **User Interface**: the interface must be user-friendly and intuitive.
- **Styles Consistency and Coherence**: the styles must be consistent and coherent throughout the platform.
""",
        dev_context="""
The final code needs to follow the following structure:
- **TITLE**: the title must be created in a way that the developer can understand what is the suggestion about.
- **FILES**: a list of files that are relevant to the suggestion.
- **CURRENT STATE**: a description of the current state of the file and the probable issues.
- **ACTION**: a overview of the necessary changes.
- **REASONING**: an explanation of the reasons for the changes.
- **FULL SOLUTION CODE**: for each file involved, a full implementation is provided.
""",
        query="""
Please, consider my /home/dusoudeth/Documentos/github/compartilar/app/(user)/[username]/meu-lar/page.tsx page
Do a deep research in order to evaluate if the UI follow the best practices for a mobile application.
"""
    )

    # Execute the graph
    for output in graph.stream(initial_state):
        print(f"Current output: {output}")

if __name__ == "__main__":
    main()