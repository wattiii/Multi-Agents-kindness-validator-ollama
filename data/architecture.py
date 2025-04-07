from diagrams import Diagram
from diagrams.programming.language import Python, Bash
from diagrams.custom import Custom

with Diagram(direction="TB"):

    # Custom items with icons
    run_sh = Bash("Run.sh")
    gel_db = Custom("Gel DB", "./icons/gelDB_white.png")
    streamlit = Custom("Streamlit", "./icons/streamlit.png")

    # Python components
    main_py = Python("main.py")
    init_py = Python("__init__.py")
    manage_db = Python("manage_database.py")
    worker = Python("worker.py")
    ollama_client = Python("ollama_client.py")
    kindness_agent = Python("kindness_agent.py")
    agent_base = Python("agent_base.py")
    config = Python("config.py")

    # Diagram relationships
    run_sh >> main_py
    run_sh >> gel_db

    main_py >> streamlit
    main_py >> init_py
    main_py >> manage_db

    init_py >> worker
    init_py >> ollama_client
    init_py >> kindness_agent

    worker >> agent_base
    ollama_client >> agent_base
    ollama_client >> config

    kindness_agent >> manage_db
