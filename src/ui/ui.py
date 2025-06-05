import streamlit as st
import os
import base64
import requests
from PIL import Image
import io
import toml

# create env dir
os.makedirs("./.streamlit", exist_ok=True)

# check and config api key path
CONFIG_PATH = "./.streamlit/secrets.toml"
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        toml.dump({"DASHSCOPE_API_KEY": "your-api-key-here"}, f)
    st.stop()

# read api key
try:
    config = toml.load(CONFIG_PATH)
    API_KEY = config.get("DASHSCOPE_API_KEY", "")
    
    if not API_KEY or API_KEY == "your-api-key-here":
        st.error("Please config API")
        st.error(f"Configuration Path: {os.path.abspath(CONFIG_PATH)}")
        st.stop()
except Exception as e:
    st.error(f"Read config file error: {str(e)}")
    st.stop()

# API config (compatible with OpenAI)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Page setting
st.set_page_config(
    page_title="450 AI Chat Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default sys prompt
DEFAULT_SYSTEM_PROMPT = """
To Add Default Prompt. For example:

You are an EV engineer responsible for disassembling batteries. You will be given a file containing the steps of assembling an EV and you need to design the process of disassembling it. 
"""

# init session status
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Sidebar setting
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.divider()
    st.subheader("Model Selection")
    model = st.selectbox(
        "qwen:",
        ["qwen-vl-plus", "qwen-vl-max", "qwen-turbo", "qwen-plus", "qwen-max"],
        index=0,
        help="Select from qwen models: vl for MLLM, others for LLM"
    )
    
    st.divider()
    st.subheader("System Settings")
    system_prompt = st.text_area(
        "System Prompt", 
        value=DEFAULT_SYSTEM_PROMPT, 
        height=200,
        help="Define AI's behavior and style"
    )
    
    st.divider()
    st.subheader("Model Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, help="the higher the answer more random")
        max_tokens = st.number_input("Max # tokens", 100, 4096, 1024)
    with col2:
        top_p = st.slider("Top P", 0.1, 1.0, 0.95, 0.05, help="Controls generation diversity")
        repetition_penalty = st.slider("Repetition Penalty", 0.1, 2.0, 1.0, 0.1, help="Reduces the likelihood of repeated content")
    
    st.divider()
    if st.button("🧹 Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploaded_image = None
        st.rerun()

# Main
st.title("🧠 450 AI Chat Agent")
st.caption("Input text or upload image")

# Check if the model is MLLM
is_multimodal = model.startswith("qwen-vl")

# Image
if is_multimodal:
    with st.expander("📷 Upload Images", expanded=True):
        uploaded_file = st.file_uploader(
            "Select Image File", 
            type=["jpg", "png", "jpeg", "webp"], 
            label_visibility="collapsed",
            key="file_uploader"
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                st.image(image, caption="Uploaded Images", use_column_width=True)
                
                # Remove image button
                if st.button("Remove Image(s)", key="remove_image"):
                    st.session_state.uploaded_image = None
                    st.rerun()
            except Exception as e:
                st.error(f"Handling Image Error: {str(e)}")
                st.session_state.uploaded_image = None
else:
    st.info("⚠️ The current model doesn't support image input, please select from qwen-vl series.")

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.write(message["content"])
        elif message["type"] == "image":
            st.image(message["content"], caption="Uploaded Image", use_column_width=True)

# User prompt
if prompt := st.chat_input("User Prompt...", key="user_input"):
    # add user prompt to history
    if is_multimodal and st.session_state.uploaded_image:
        # image + MLLM -> show MLLM message
        st.session_state.messages.append({
            "role": "user", 
            "type": "multimodal",
            "text": prompt,
            "image": st.session_state.uploaded_image
        })
    else:
        # pure LLM
        st.session_state.messages.append({
            "role": "user", 
            "type": "text",
            "content": prompt
        })
    
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)
        if is_multimodal and st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # prepare API request
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # build message
    messages = [{"role": "system", "content": system_prompt}]
    
    # add history
    for msg in st.session_state.messages:
        if msg["type"] == "text":
            messages.append({"role": msg["role"], "content": msg["content"]})
        elif msg["type"] == "multimodal" and msg["role"] == "user" and is_multimodal:
            # convert image to base64
            buffered = io.BytesIO()
            
            # RGBA -> RGB
            if msg["image"].mode == "RGBA":
                img = msg["image"].convert("RGB")
                img.save(buffered, format="JPEG")
            else:
                msg["image"].save(buffered, format="JPEG")
            
            image_content = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": msg["text"]},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_content}"}}
                ]
            })
    
    # build request body
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "stream": False
    }
    
    # call API
    with st.spinner("AI is thinking..."):
        try:                
            response = requests.post(
                f"{BASE_URL}/chat/completions", 
                headers=headers, 
                json=payload
            )
            
            if response.status_code != 200:
                st.error(f"API Request Failure: HTTP {response.status_code}")
                st.error(f"Error Message: {response.text}")
                st.stop()
                
            response_data = response.json()
            
            # check API return error
            if "error" in response_data:
                error_msg = response_data["error"].get("message", "Unknown Error")
                st.error(f"API Return Error: {error_msg}")
                st.stop()
            
            ai_response = response_data["choices"][0]["message"]["content"]
            
            # add AI response to history
            st.session_state.messages.append({
                "role": "assistant", 
                "type": "text",
                "content": ai_response
            })
            
            # show AI response
            with st.chat_message("assistant"):
                st.write(ai_response)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
