from os import getenv

from agno.playground import Playground

from agents.sage import get_sage
from workspace.dev_resources import dev_fastapi

######################################################
## Router for the Agent Playground
######################################################

sage_agent = get_sage(debug_mode=True)

# Create a playground instance
playground = Playground(agents=[sage_agent])

# Register the endpoint where playground routes are served with agno.com
if getenv("RUNTIME_ENV") == "dev":
    playground.create_endpoint(f"http://localhost:{dev_fastapi.host_port}")

playground_router = playground.get_router()
