import streamlit as st
import requests
import json
import tempfile
import os
import base64

# Streamlit app configuration
st.set_page_config(page_title="Document Analyzer", page_icon="📄", layout="wide")

# Backend API URL
API_URL = "http://localhost:5000"

# Function to handle user authentication
def authenticate(email, password, is_login):
    endpoint = f"{API_URL}/login" if is_login else f"{API_URL}/register"
    response = requests.post(endpoint, json={"email": email, "password": password})
    return response.json()

# Function to analyze document
def analyze_document(file, token):
    headers = {"Authorization": token}
    files = {"file": file}
    response = requests.post(f"{API_URL}/analyze", headers=headers, files=files)
    return response

# Function to display PDF
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.title("Document Analyzer")

    # Sidebar for authentication
    with st.sidebar:
        st.header("Authentication")
        auth_option = st.radio("Choose an option:", ("Login", "Register"))
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        auth_button = st.button("Submit")

        if auth_button:
            response = authenticate(email, password, auth_option == "Login")
            if "token" in response:
                st.session_state.token = response["token"]
                st.success("Authentication successful!")
            else:
                st.error(response.get("message", "Authentication failed!"))

    # Main content
    if "token" in st.session_state:
        st.header("Upload and Analyze Document")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "doc", "docx", "jpg", "jpeg", "png", "mp4", "avi", "mov"])

        if uploaded_file:
            if st.button("Analyze"):
                with st.spinner("Analyzing document..."):
                    try:
                        response = analyze_document(uploaded_file, st.session_state.token)

                        if response.status_code == 200:
                            content_type = response.headers.get("content-type")
                            if "application/pdf" in content_type:
                                # Save and display PDF report
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                                    temp_file.write(response.content)
                                    temp_file_path = temp_file.name
                                st.success("Analysis complete! Displaying report:")
                                display_pdf(temp_file_path)
                                os.unlink(temp_file_path)
                            elif "application/json" in content_type:
                                # Display image analysis results
                                results = response.json()
                                st.json(results)
                            elif "video/mp4" in content_type:
                                # Save and display censored video
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                                    temp_file.write(response.content)
                                    temp_file_path = temp_file.name
                                st.success("Video processing complete!")
                                st.video(temp_file_path)
                                os.unlink(temp_file_path)
                        else:
                            error_message = response.json().get('message', 'Unknown error occurred')
                            st.error(f"Analysis failed: {error_message}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error connecting to the server: {str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
    else:
        st.info("Please login or register to use the Document Analyzer.")

if __name__ == "__main__":
    main()

