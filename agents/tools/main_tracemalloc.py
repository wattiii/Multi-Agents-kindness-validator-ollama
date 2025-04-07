from agents.config import Config
import streamlit as st
import asyncio
from __init__ import AgentManager
from agents.kindness_agent import FastKindnessEvaluator, OllamaLLM
from manage_database import DatabaseManager
from config import Config
import tracemalloc
import pathlib

# --- Set page configuration as the first Streamlit command ---
st.set_page_config(
    page_title="✨ Kind Learning Assistant",
    page_icon="🌟",
    layout="wide"
)

# --- Start tracing memory allocations ---
tracemalloc.start()

def display_top_memory_usage(limit=10):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    st.write("[ Top memory-consuming lines ]")
    for stat in top_stats[:limit]:
        st.write(stat)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_manager" not in st.session_state:
        config = Config()
        st.session_state.agent_manager = AgentManager(config=config)

async def evaluate_response(evaluator, explanation):
    evaluation = await evaluator.generate_scores(explanation)
    return evaluation

# Optionally, include a button to check memory usage.
if st.button("Check Memory Usage"):
    display_top_memory_usage()

async def main():
    # Now that st.set_page_config() has been called at the top, 
    # we don't call it here again.
    initialize_session_state()

    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            border-radius: 15px;
        }
        .stButton > button {
            border-radius: 20px;
            background-color: #9DB5B2;
            color: white;
        }
        .output-box {
            border-radius: 15px;
            padding: 20px;
            background-color: #F3F6F5;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("✨ Kind Learning Assistant ✨")
        st.markdown("""
            <p style='text-align: center; color: #666666; font-size: 1.1em;'>
            I'm here to help you learn in a supportive and encouraging way! 
            What would you like to explore today?
            </p>
        """, unsafe_allow_html=True)

    with st.container():
        question = st.text_area(
            "What would you like to learn about?",
            height=100,
            placeholder="Feel free to ask anything! I'm here to help you understand in a clear and friendly way.",
            key="question_input"
        )
        # Configuration for the LLM
        config = {
            "model": "dolphin-llama3:latest",
            "base_url": "http://localhost:11434",
            "temperature": 0.2,
            "max_tokens": 256
        }

    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        if st.button("✨ Let's Learn Together ✨", use_container_width=True):
            if question:
                with st.spinner("I have finished searching the Internet. Preparing a thoughtful response..."):
                    writer = st.session_state.agent_manager.get_agent("writer")
                    explanation = writer.execute(question)
                    
                    evaluator = FastKindnessEvaluator(OllamaLLM(config))
                    if evaluator is None:
                        st.error("Error: Evaluator was not initialized.")
                    elif not isinstance(evaluator, FastKindnessEvaluator):
                        st.error(f"Error: Evaluator is of type {type(evaluator).__name__} instead of FastKindnessEvaluator.")
                    else:
                        evaluation = await evaluate_response(evaluator, explanation)
                        
                        new_message = {
                            "question": question,
                            "response": explanation,
                            "evaluation": evaluation
                        }
                        st.session_state.messages.append(new_message)
                        
                        db_manager = DatabaseManager()
                        await db_manager.save_results(
                            topic=question,
                            content=explanation,
                            scores=evaluation
                        )

    # Display messages from session state
    for msg in st.session_state.messages:
        st.markdown("---")
        st.markdown("### 🤔 Your Question")
        st.markdown(f"<div class='container'>{msg['question']}</div>", unsafe_allow_html=True)
        
        st.markdown("### 💡 Learning Response")
        st.markdown(f"<div class='container'>{msg['response']}</div>", unsafe_allow_html=True)
        
        with st.expander("✨ Response Quality Analysis"):
            eval_data = msg['evaluation']
            if eval_data:
                st.markdown("#### Quality Scores")
                for metric, score in eval_data.items():
                    metric_name = metric.replace("_score", "").title()
                    st.markdown(f"- **{metric_name}**: {'⭐' * int(score * 5)} ({score:.2f})")

# Run the asyncio loop when the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
