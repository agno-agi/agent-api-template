from typing import Any, Callable, Dict, List, Optional

import streamlit as st
from agno.agent import Agent
from agno.document import Document
from agno.document.reader import Reader
from agno.document.reader.csv_reader import CSVReader
from agno.document.reader.docx_reader import DocxReader
from agno.document.reader.pdf_reader import PDFReader
from agno.document.reader.text_reader import TextReader
from agno.document.reader.website_reader import WebsiteReader
from agno.utils.log import logger


async def initialize_agent_session_state(agent_name: str):
    logger.info(f"---*--- Initializing session state for {agent_name} ---*---")
    if agent_name not in st.session_state:
        st.session_state[agent_name] = {
            "agent": None,
            "session_id": None,
            "messages": [],
        }


async def selected_model() -> str:
    """Display a model selector in the sidebar."""
    model_options = {
        "gpt-4o": "gpt-4o",
        "o3-mini": "o3-mini",
    }
    selected_model = st.sidebar.selectbox(
        "Choose a model",
        options=list(model_options.keys()),
        index=0,
        key="model_selector",
    )
    return model_options[selected_model]


async def add_message(
    agent_name: str, role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Safely add a message to the Agent's session state."""
    # if role == "user":
    #     logger.info(f"👤  {role} → {agent_name}: {content}")
    # else:
    #     logger.info(f"🤖  {agent_name} → user: {content}")
    st.session_state[agent_name]["messages"].append({"role": role, "content": content, "tool_calls": tool_calls})


def display_tool_calls(tool_calls_container, tools):
    """Display tool calls in a streamlit container with expandable sections.

    Args:
        tool_calls_container: Streamlit container to display the tool calls
        tools: List of tool call dictionaries containing name, args, content, and metrics
    """
    if not tools:
        return

    try:
        with tool_calls_container.container():
            for tool_call in tools:
                tool_name = tool_call.get("tool_name", "Unknown Tool")
                tool_args = tool_call.get("tool_args", {})
                content = tool_call.get("content")
                metrics = tool_call.get("metrics", {})

                # Add timing information
                execution_time_str = "N/A"
                try:
                    if metrics:
                        execution_time = metrics.time
                        if execution_time is not None:
                            execution_time_str = f"{execution_time:.2f}s"
                except Exception as e:
                    logger.error(f"Error displaying tool calls: {str(e)}")
                    pass

                with st.expander(
                    f"🛠️ {tool_name.replace('_', ' ').title()} ({execution_time_str})",
                    expanded=False,
                ):
                    # Show query with syntax highlighting
                    if isinstance(tool_args, dict) and tool_args.get("query"):
                        st.code(tool_args["query"], language="sql")

                    # Display arguments in a more readable format
                    if tool_args and tool_args != {"query": None}:
                        st.markdown("**Arguments:**")
                        st.json(tool_args)

                    if content:
                        st.markdown("**Results:**")
                        try:
                            # Check if content is already a dictionary or can be parsed as JSON
                            if isinstance(content, dict) or (
                                isinstance(content, str) and content.strip().startswith(("{", "["))
                            ):
                                st.json(content)
                            else:
                                # If not JSON, show as markdown
                                st.markdown(content)
                        except Exception:
                            # If JSON display fails, show as markdown
                            st.markdown(content)
    except Exception as e:
        logger.error(f"Error displaying tool calls: {str(e)}")
        tool_calls_container.error(f"Failed to display tool results: {str(e)}")


async def example_inputs(agent_name: str) -> None:
    """Show example inputs for an Agent."""
    with st.sidebar:
        st.markdown("#### :thinking_face: Try me!")
        if st.button("Who are you?"):
            await add_message(
                agent_name,
                "user",
                "Who are you?",
            )
        if st.button("What is your purpose?"):
            await add_message(
                agent_name,
                "user",
                "What is your purpose?",
            )

        # Agent-specific examples
        if agent_name == "sage":
            if st.button("Tell me about Agno"):
                await add_message(
                    agent_name,
                    "user",
                    "Tell me about Agno. Github repo: https://github.com/agno-agi/agno. Documentation: https://docs.agno.com",
                )
        elif agent_name == "scholar":
            if st.button("Tell me about the US tariffs"):
                await add_message(
                    agent_name,
                    "user",
                    "Tell me about the US tariffs",
                )


async def knowledge_widget(agent_name: str, agent: Agent) -> None:
    """Display a knowledge widget in the sidebar."""

    if agent is not None and agent.knowledge is not None:
        # Add websites to knowledge base
        if "url_scrape_key" not in st.session_state:
            st.session_state[agent_name]["url_scrape_key"] = 0
        input_url = st.sidebar.text_input(
            "Add URL to Knowledge Base", type="default", key=st.session_state[agent_name]["url_scrape_key"]
        )
        add_url_button = st.sidebar.button("Add URL")
        if add_url_button:
            if input_url is not None:
                alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
                if f"{input_url}_scraped" not in st.session_state:
                    scraper = WebsiteReader(max_links=2, max_depth=1)
                    web_documents: List[Document] = scraper.read(input_url)
                    if web_documents:
                        agent.knowledge.load_documents(web_documents, upsert=True)
                    else:
                        st.sidebar.error("Could not read website")
                    st.session_state[f"{input_url}_uploaded"] = True
                alert.empty()

        # Add documents to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state[agent_name]["file_uploader_key"] = 100
        uploaded_file = st.sidebar.file_uploader(
            "Add a Document (.pdf, .csv, .txt, or .docx)",
            key=st.session_state[agent_name]["file_uploader_key"],
        )
        if uploaded_file is not None:
            alert = st.sidebar.info("Processing document...", icon="🧠")
            document_name = uploaded_file.name.split(".")[0]
            if f"{document_name}_uploaded" not in st.session_state:
                file_type = uploaded_file.name.split(".")[-1].lower()

                reader: Reader
                if file_type == "pdf":
                    reader = PDFReader()
                elif file_type == "csv":
                    reader = CSVReader()
                elif file_type == "txt":
                    reader = TextReader()
                elif file_type == "docx":
                    reader = DocxReader()
                uploaded_file_documents: List[Document] = reader.read(uploaded_file)
                if uploaded_file_documents:
                    agent.knowledge.load_documents(uploaded_file_documents, upsert=True)
                else:
                    st.sidebar.error("Could not read document")
                st.session_state[f"{document_name}_uploaded"] = True
            alert.empty()

        # Load and delete knowledge
        if st.sidebar.button("🗑️ Delete Knowledge"):
            agent.knowledge.delete()
            st.sidebar.success("Knowledge deleted!")


async def session_selector(agent_name: str, agent: Agent, get_agent: Callable, user_id: str, model_id: str) -> None:
    """Display a session selector in the sidebar, if a new session is selected, the agent is restarted with the new session."""

    if not agent.storage:
        return

    try:
        # Get all agent sessions.
        agent_sessions = agent.storage.get_all_sessions()
        if not agent_sessions:
            st.sidebar.info("No saved sessions found.")
            return

        # Get session names if available, otherwise use IDs.
        sessions_list = []
        for session in agent_sessions:
            session_id = session.session_id
            session_name = session.session_data.get("session_name", None) if session.session_data else None
            display_name = session_name if session_name else session_id
            sessions_list.append({"id": session_id, "display_name": display_name})

        # Display session selector.
        st.sidebar.markdown("#### 💬 Session")
        selected_session = st.sidebar.selectbox(
            "Session",
            options=[s["display_name"] for s in sessions_list],
            key="session_selector",
            label_visibility="collapsed",
        )
        # Find the selected session ID.
        selected_session_id = next(s["id"] for s in sessions_list if s["display_name"] == selected_session)
        # Update the agent session if it has changed.
        if st.session_state[agent_name]["session_id"] != selected_session_id:
            logger.info(f"---*--- Loading {agent_name} session: {selected_session_id} ---*---")
            st.session_state[agent_name]["agent"] = get_agent(
                user_id=user_id,
                model_id=model_id,
                session_id=selected_session_id,
            )
            st.rerun()

        # Show the rename session widget.
        container = st.sidebar.container()
        session_row = container.columns([3, 1], vertical_alignment="center")

        # Initialize session_edit_mode if needed.
        if "session_edit_mode" not in st.session_state:
            st.session_state.session_edit_mode = False

        # Show the session name.
        with session_row[0]:
            if st.session_state.session_edit_mode:
                new_session_name = st.text_input(
                    "Session Name",
                    value=agent.session_name,
                    key="session_name_input",
                    label_visibility="collapsed",
                )
            else:
                st.markdown(f"Session Name: **{agent.session_name}**")

        # Show the rename session button.
        with session_row[1]:
            if st.session_state.session_edit_mode:
                if st.button("✓", key="save_session_name", type="primary"):
                    if new_session_name:
                        agent.rename_session(new_session_name)
                        st.session_state.session_edit_mode = False
                        container.success("Renamed!")
                        # Trigger a rerun to refresh the sessions list
                        st.rerun()
            else:
                if st.button("✎", key="edit_session_name"):
                    st.session_state.session_edit_mode = True
    except Exception as e:
        logger.error(f"Error in session selector: {str(e)}")
        st.sidebar.error("Failed to load sessions")


def export_chat_history(agent_name: str):
    """Export chat history in markdown format.

    Returns:
        str: Formatted markdown string of the chat history
    """
    if "messages" not in st.session_state[agent_name] or not st.session_state[agent_name]["messages"]:
        return f"# {agent_name} - Chat History\n\nNo messages to export."

    chat_text = f"# {agent_name} - Chat History\n\n"
    for msg in st.session_state[agent_name]["messages"]:
        role_label = "🤖 Assistant" if msg["role"] == "assistant" else "👤 User"
        chat_text += f"### {role_label}\n{msg['content']}\n\n"

        # Include tool calls if present
        if msg.get("tool_calls"):
            chat_text += "#### Tool Calls:\n"
            for i, tool_call in enumerate(msg["tool_calls"]):
                tool_name = tool_call.get("name", "Unknown Tool")
                chat_text += f"**{i + 1}. {tool_name}**\n\n"
                if "arguments" in tool_call:
                    chat_text += f"Arguments: ```json\n{tool_call['arguments']}\n```\n\n"
                if "content" in tool_call:
                    chat_text += f"Results: ```\n{tool_call['content']}\n```\n\n"

    return chat_text


async def utilities_widget(agent_name: str, agent: Agent) -> None:
    """Display a utilities widget in the sidebar."""
    st.sidebar.markdown("#### 🛠️ Utilities")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Start New Chat"):
            restart_agent(agent_name)
    with col2:
        fn = f"{agent_name}_chat_history.md"
        if "session_id" in st.session_state[agent_name]:
            fn = f"{agent_name}_{st.session_state[agent_name]['session_id']}.md"
        if st.download_button(
            ":file_folder: Export Chat History",
            export_chat_history(agent_name),
            file_name=fn,
            mime="text/markdown",
        ):
            st.sidebar.success("Chat history exported!")


def restart_agent(agent_name: str):
    logger.debug("---*--- Restarting Agent ---*---")
    st.session_state[agent_name]["agent"] = None
    st.session_state[agent_name]["session_id"] = None
    st.session_state[agent_name]["messages"] = []
    if "url_scrape_key" in st.session_state[agent_name]:
        st.session_state[agent_name]["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state[agent_name]:
        st.session_state[agent_name]["file_uploader_key"] += 1
    st.rerun()


async def about_agno():
    """Show information about Agno in the sidebar"""
    with st.sidebar:
        st.markdown("### About Agno ✨")
        st.markdown("""
        Agno is an open-source library for building Multimodal Agents.

        [GitHub Repository](https://github.com/agno-agi/agno) | [Documentation](https://docs.agno.com)
        """)

        st.markdown("### Need Help?")
        st.markdown(
            "If you have any questions, catch us on [discord](https://agno.link/discord) or post in the community [forum](https://agno.link/community)."
        )


async def footer():
    st.markdown("---")
    st.markdown(
        "<p style='text-align: right; color: gray;'>Built using <a href='https://github.com/agno-agi/agno'>Agno</a></p>",
        unsafe_allow_html=True,
    )
