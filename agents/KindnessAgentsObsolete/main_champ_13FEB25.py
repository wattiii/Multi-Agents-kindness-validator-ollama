import streamlit as st
import asyncio
from __init__ import AgentManager
from agents.kindness_agent import FastKindnessEvaluator, OllamaLLM  # Ensure FastKindnessEvaluator is imported
from manage_database import DatabaseManager

def initialize_session_state():
    """
    The function `initialize_session_state` checks if certain keys exist in the session state and
    initializes them if they don't.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_manager" not in st.session_state:
        st.session_state.agent_manager = AgentManager()


async def main():
    """
    The `main` function sets up a Kind Learning Assistant web app using Streamlit, where users can ask
    questions, receive responses, and view quality analysis scores for each response.
    """
    st.set_page_config(
        page_title="✨ Kind Learning Assistant",
        page_icon="🌟",
        layout="wide"
    )

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

        col1, col2, col3 = st.columns([3, 2, 3])
        with col2:
            if st.button("✨ Let's Learn Together ✨", use_container_width=True):
                if question:
                    with st.spinner("I have finished searching the Internet. Preparing a thoughtful response..."):
                        writer = st.session_state.agent_manager.get_agent("writer")
                        response = writer.execute(question)

                        # Get evaluator instance (FastKindnessEvaluator)
                        evaluator = FastKindnessEvaluator(OllamaLLM())  # Initialize the evaluator
                        if evaluator is None:
                            st.error("Error: Evaluator was not initialized.")
                        elif not isinstance(evaluator, FastKindnessEvaluator):
                            st.error(f"Error: Evaluator is of type {type(evaluator).__name__} instead of FastKindnessEvaluator.")
                        if isinstance(evaluator, FastKindnessEvaluator):  # Ensure correct evaluator instance
                            evaluation = await evaluator.generate_scores(question, response)  # ✅ Await evaluation

                            # Add the evaluation data (scores) to the session state
                            st.session_state.messages.append({
                                "question": question,
                                "response": response,
                                "evaluation": evaluation  # Store the score data directly
                            })
                        else:
                            st.error("Error: Evaluator is not an instance of FastKindnessEvaluator.")

    for msg in st.session_state.messages:
        st.markdown("---")
        
        st.markdown(f"### 🤔 Your Question")
        st.markdown(f"<div class='container'>{msg['question']}</div>", unsafe_allow_html=True)
        
        st.markdown(f"### 💡 Learning Response")
        st.markdown(f"<div class='container'>{msg['response']}</div>", unsafe_allow_html=True)
        
        with st.expander("✨ Response Quality Analysis"):
            eval_data = msg['evaluation']  # Directly retrieve the evaluation data (scores)

            if eval_data:
                col1 = st.columns(1)[0]  # Only one column now for quality scores
                with col1:
                    st.markdown("#### Quality Scores")
                    # Ensure that 'scores' exist and iterate over them
                    for metric, score in eval_data.items():
                        metric_name = metric.replace("_score", "").title()  # Remove "_score"
                        st.markdown(f"- **{metric_name}**: {'⭐' * int(score * 5)} ({score:.2f})")
# Show both stars and score
# Instantiate DatabaseManager (no arguments needed)
    db_manager = DatabaseManager()  # The client is already available from the global config

    # Now call save_results with real data
    await db_manager.save_results(
        topic=prompt,
        content=response,
        scores=score
        }
    )

# Run the asyncio loop
if __name__ == "__main__":
    asyncio.run(main())