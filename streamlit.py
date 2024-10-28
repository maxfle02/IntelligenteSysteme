import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import time
import os
import dotenv
from dotenv import load_dotenv

load_dotenv()

# Set default section (Chat interface will be the main view)
st.sidebar.title("Options")
selected_tab = st.sidebar.radio("Go to", ["💬 Chat Interface", "⚙️ Update Info & Upload"])

# Sidebar for OpenAI API Key (accessible on both sections)
openai_api_key = os.getenv("OPENAI_KEY")

# Initialize session state for chat history and token count if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# Function to generate and store AI response
def generate_response(input_text):
    try:
        model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)

        # Create a complete message history to send to OpenAI
        messages = [{"role": "user", "content": input_text}]
        messages.extend(st.session_state.chat_history)  # Include chat history

        # print("Sending messages to OpenAI:", messages)  # For debugging
        
        # Get response from the model, which includes token usage
        response = model.invoke(messages)
        print(st.session_state.total_tokens)
        print(response.response_metadata['token_usage']['total_tokens'])
        st.session_state.total_tokens += response.response_metadata['token_usage']['total_tokens']  # Update total tokens
        return response.content  # Return the AI response
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None  # Return None on error

# Default section: Chat Interface
if selected_tab == "💬 Chat Interface":
    st.title("💬 Chat with AI Assistant")

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display the current token count
    st.sidebar.markdown(f"**Total Tokens Used:** {st.session_state.total_tokens}")

    # Accept user input
    if user_query := st.chat_input("What would you like to ask?"):
        # Check if API key is entered correctly

        if not openai_api_key.startswith("sk-"):
            st.warning("Please enter a valid OpenAI API key.", icon="⚠")
        else:
            # Display the user's message immediately
            with st.chat_message("user"):
                st.markdown(user_query)
                st.session_state.chat_history.append({"role": "user", "content": user_query})  # Store user input

            # Show loading message while waiting for the response
            with st.chat_message("assistant"):
                loading_message = st.markdown("Loading...")  # Placeholder for the AI response

                # Call the function to get the AI response
                assistant_response = generate_response(user_query)

                # Simulate a loading delay (if needed)
                time.sleep(1)  # Adjust the delay as necessary

                # Clear the loading message
                loading_message.empty()

                # If a valid response is returned, display it
                if assistant_response:
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})  # Store AI response
                    st.markdown(assistant_response)

# Optional section: Personal Info & PDF Upload (Settings/Configuration page)
elif selected_tab == "⚙️ Update Info & Upload":
    st.title("⚙️ Update Personal Info & Upload Manuals")

    # Form to collect user information and upload PDFs
    with st.form("personal_info_form"):
        # Personal Info Inputs
        name = st.text_input("Enter your Name:", value="John Doe")
        age = st.number_input("Enter your Age:", min_value=1, max_value=120, value=30, step=1)

        # File Uploader for PDFs
        uploaded_files = st.file_uploader(
            "Upload PDF User Manuals", accept_multiple_files=True, type=["pdf"]
        )
        
        # Submit Button
        submit_info = st.form_submit_button("Update Info")

        # Display submitted data
        if submit_info:
            if name and age:
                st.success(f"Hello {name}, aged {age}. Your information has been updated.")
            else:
                st.warning("Please fill out both name and age.")
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    st.write(f"Uploaded file: {uploaded_file.name}")
            else:
                st.warning("No files uploaded yet.")

