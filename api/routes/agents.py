from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from agno.agent import Agent
from agno.storage.agent.session import AgentSession
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.operator import AgentType, get_agent, get_available_agents
from utils.log import logger

######################################################
## Router for Serving Agents
######################################################

agents_router = APIRouter(prefix="/agents", tags=["Agents"])


class Model(str, Enum):
    gpt_4o = "gpt-4o"
    o3_mini = "o3-mini"


@agents_router.get("", response_model=List[str])
async def list_agents():
    """
    GET /agents

    Returns a list of all available agent IDs.

    Returns:
        List[str]: List of agent identifiers
    """
    return get_available_agents()


async def chat_response_streamer(agent: Agent, message: str) -> AsyncGenerator:
    """
    Stream agent responses chunk by chunk.

    Args:
        agent: The agent instance to interact with
        message: User message to process

    Yields:
        Text chunks from the agent response
    """
    run_response = await agent.arun(message, stream=True)
    async for chunk in run_response:
        # chunk.content only contains the text response from the Agent.
        # For advanced use cases, we should yield the entire chunk
        # that contains the tool calls and intermediate steps.
        yield chunk.content


class RunRequest(BaseModel):
    """Request model for an running an agent"""

    message: str
    stream: bool = True
    model: Model = Model.gpt_4o
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@agents_router.post("/{agent_id}/run", status_code=status.HTTP_200_OK)
async def run_agent(agent_id: AgentType, body: RunRequest):
    """
    POST /agents/{agent_id}/run

    Sends a message to a specific agent and returns the response.

    Args:
        agent_id: The ID of the agent to interact with
        body: Request parameters including the message

    Returns:
        Either a streaming response or the complete agent response
    """
    logger.debug(f"RunRequest: {body}")

    try:
        agent: Agent = get_agent(
            model_id=body.model.value,
            agent_id=agent_id,
            user_id=body.user_id,
            session_id=body.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent not found: {str(e)}")

    if body.stream:
        return StreamingResponse(
            chat_response_streamer(agent, body.message),
            media_type="text/event-stream",
        )
    else:
        response = await agent.arun(body.message, stream=False)
        # response.content only contains the text response from the Agent.
        # For advanced use cases, we should yield the entire response
        # that contains the tool calls and intermediate steps.
        return response.content


@agents_router.get("/{agent_id}/sessions", response_model=List[str])
async def get_agent_session_ids(agent_id: AgentType, user_id: Optional[str] = None):
    """
    GET /agents/{agent_id}/sessions

    Returns all session IDs for a specific agent and user.

    Args:
        agent_id: The agent type
        user_id: Optional user ID

    Returns:
        List[str]: List of session identifiers
    """
    logger.debug(f"GetAgentSessionsRequest: agent_id={agent_id}, user_id={user_id}")
    return get_agent(user_id=user_id, agent_id=agent_id).storage.get_all_session_ids()


@agents_router.get("/{agent_id}/sessions/{session_id}", response_model=Optional[AgentSession])
async def get_agent_session(agent_id: AgentType, session_id: str, user_id: Optional[str] = None):
    """
    GET /agents/{agent_id}/sessions/{session_id}

    Retrieves details about a specific agent session.

    Args:
        agent_id: Agent ID
        session_id: The session ID to retrieve
        user_id: Optional user ID

    Returns:
        AgentSession or None if not found
    """
    logger.debug(f"GetAgentSessionRequest: agent_id={agent_id}, session_id={session_id}, user_id={user_id}")

    try:
        agent: Agent = get_agent(session_id=session_id, user_id=user_id, agent_id=agent_id)
        return agent.read_from_storage()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session not found: {str(e)}")


@agents_router.get("/{agent_id}/sessions/{session_id}/messages", response_model=List[Dict[str, Any]])
async def get_session_messages(agent_id: AgentType, session_id: str, user_id: Optional[str] = None):
    """
    GET /agents/{agent_id}/sessions/{session_id}/messages

    Retrieves the messages for a specific agent session.

    Args:
        agent_id: Agent ID
        session_id: The session ID to retrieve history for
        user_id: Optional user ID

    Returns:
        List of message objects representing the conversation history
    """
    logger.debug(f"GetSessionHistoryRequest: agent_id={agent_id}, session_id={session_id}, user_id={user_id}")

    try:
        agent: Agent = get_agent(session_id=session_id, user_id=user_id, agent_id=agent_id)
        # Load the agent from the database
        agent.read_from_storage()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session not found: {str(e)}")

    if agent.memory:
        return agent.memory.get_messages()
    else:
        return []


@agents_router.delete("/{agent_id}/sessions/{session_id}", response_model=dict)
async def delete_session(agent_id: AgentType, session_id: str, user_id: Optional[str] = None):
    """
    DELETE /agents/{agent_id}/sessions/{session_id}

    Deletes a specific agent session.
    """
    logger.debug(f"DeleteSessionRequest: agent_id={agent_id}, session_id={session_id}")

    try:
        agent: Agent = get_agent(user_id=user_id, agent_id=agent_id, session_id=session_id)
        agent.delete_session(session_id=session_id)
        return {"message": "Session deleted"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session not found: {str(e)}")


class RenameSessionRequest(BaseModel):
    """Request model for renaming a session"""

    session_name: str


@agents_router.patch("/{agent_id}/sessions/{session_id}/rename", response_model=dict)
async def rename_session(
    agent_id: AgentType, session_id: str, body: RenameSessionRequest, user_id: Optional[str] = None
):
    """
    PATCH /agents/{agent_id}/sessions/{session_id}/rename

    Renames a specific agent session.

    Args:
        agent_id: Agent ID
        session_id: The session ID to rename
        body: Request containing the new session name
        user_id: Optional user ID

    Returns:
        Updated session information
    """
    logger.debug(
        f"RenameSessionRequest: agent_id={agent_id}, session_id={session_id}, session_name={body.session_name}"
    )

    try:
        agent: Agent = get_agent(user_id=user_id, agent_id=agent_id, session_id=session_id)
        agent.rename_session(session_name=body.session_name)

        return {
            "session_id": agent.session_id,
            "session_name": agent.session_name,
        }
    except Exception as e:
        # Use a more appropriate status code for errors that might not be "not found"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to rename session: {str(e)}")


@agents_router.post("/{agent_id}/sessions/{session_id}/auto-rename", response_model=dict)
async def auto_rename_session(agent_id: AgentType, session_id: str, user_id: Optional[str] = None):
    """
    POST /agents/{agent_id}/sessions/{session_id}/auto-rename

    Automatically renames a session using the LLM based on conversation context.

    Args:
        session_id: The session ID to auto-rename
        user_id: Optional user ID
        agent_id: Optional agent type

    Returns:
        Updated session information with the auto-generated name
    """
    logger.debug(f"AutoRenameSessionRequest: agent_id={agent_id}, session_id={session_id}")

    try:
        agent: Agent = get_agent(user_id=user_id, agent_id=agent_id, session_id=session_id)
        agent.auto_rename_session()

        return {
            "session_id": agent.session_id,
            "session_name": agent.session_name,
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Failed to auto-rename session: {str(e)}")


@agents_router.post("/{agent_id}/load-knowledge", status_code=status.HTTP_200_OK)
async def load_knowledge(agent_id: AgentType, user_id: Optional[str] = None, recreate: bool = False):
    """
    POST /agents/{agent_id}/load-knowledge

    Loads the knowledge base for a specific agent. Please update the Agent Knowledge Base with the required data before calling this endpoint. Example: PDFUrlKnowledgeBase, CSVKnowledgeBase, etc.

    Args:
        agent_id: The agent type
        user_id: Optional user ID
        recreate: Whether to recreate the knowledge base
    Returns:
        Confirmation message
    """
    logger.debug(f"LoadKnowledgeRequest: agent_id={agent_id}, user_id={user_id}")

    try:
        agent: Agent = get_agent(user_id=user_id, agent_id=agent_id)
        logger.debug(f"Agent: {agent.knowledge}")
        if agent.knowledge is not None:
            agent.knowledge.load(recreate=recreate)
        return {"message": "Knowledge Base Loaded"}
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {str(e)}")
        # Consider more specific error handling here - not all exceptions should be 404
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Failed to load knowledge base: {str(e)}")
