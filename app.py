from typing import Tuple, Dict, List
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import requests
import base64
import json
import os
import io

load_dotenv()

@st.cache_data
def get_chat_params() -> Tuple[str, str, str]:
    api_key: str = os.getenv("API_KEY")
    url: str = os.getenv("URL")
    model: str = "your-model-id"
    return api_key, url, model

class VisualChat(object):
    def __init__(self, api_key: str, url: str, model: str) -> None:
        self.api_key: str = api_key
        self.url: str = url
        self.model: str = model
    
    def init_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        return headers
    
    @st.fragment
    def chat_completion(
        self,
        query: str,
        t_session: List,
        i_session: List,
        inst: str | None,
        base64_images: List,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        frequency_penalty: float,
        presence_penalty: float
    ) -> None:
        headers: Dict[str, str] = self.init_headers()

        t_session.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        messages = [{"role": "system", "content": inst}] if inst is not None else []

        if len(t_session) == 1:
            i_session.append({"role": "user", "content": []})
            for base64_img in base64_images:
                img_url = f"data:image/png;base64,{base64_img}"
                image_url = {"url": img_url, "detail": "high"}
                i_session[0]["content"].append({"type": "image_url", "image_url": image_url})
            i_session[0]["content"].append({"type": "text", "text": query})
        elif len(t_session) > 1:
            i_session.append({"role": "user", "content": query})
        
        messages += i_session

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": True
        }

        buffer = io.StringIO()
        with st.chat_message("assistant"):
            placeholder = st.empty()

        try:
            response = requests.request("POST", self.url, headers=headers, json=payload, stream=True)
            if response.status_code == 200:
                for chunk in response.iter_lines():
                    if not chunk:
                        continue
                    decoded_chunk: str = chunk.decode("utf-8")
                    if decoded_chunk.startswith("data: [DONE]"):
                        break
                    if decoded_chunk.startswith("data:"):
                        json_chunk = json.loads(decoded_chunk.split("data:")[1].strip())
                        if not json_chunk["choices"]:
                            continue
                        delta = json_chunk["choices"][0]["delta"]
                        if "content" in delta and delta["content"] is not None:
                            buffer.write(delta["content"])
                            placeholder.markdown(buffer.getvalue())
                
                content = buffer.getvalue()
                t_session.append({"role": "assistant", "content": content})
                i_session.append({"role": "assistant", "content": content})
                buffer.close()

                st.rerun()
            else:
                st.warning(f"{response.status_code}:\n\n{response.text}")
        except Exception as e:
            st.error(f"Response Error:\n\n{e}")

@st.cache_resource
def init_client(api_key: str, url: str, model: str) -> VisualChat:
    client = VisualChat(api_key=api_key, url=url, model=model)
    return client

@st.fragment
def process_image(uploaded_image) -> str:
    max_size = 1024 * 1024
    img = Image.open(uploaded_image)

    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    output_buffer = io.BytesIO()
    quality = 95

    while True:
        output_buffer.seek(0)
        output_buffer.truncate()
        img.save(output_buffer, format='PNG', quality=quality, optimize=True)
        if len(output_buffer.getvalue()) <= max_size or quality <= 10:
            break
        quality -= 5
    
    base64_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
    return base64_image

@st.fragment
def display_conversation(t_session_state: List) -> None:
    for message in t_session_state:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main() -> None:
    api_key, url, model = get_chat_params()
    client = init_client(api_key=api_key, url=url, model=model)

    if "base64_img" not in st.session_state:
        st.session_state.base64_img = []
    if "t_content" not in st.session_state:
        st.session_state.t_content = []
    if "i_content" not in st.session_state:
        st.session_state.i_content = []

    @st.fragment
    def clear_conversation() -> None:
        if st.button("Clear", "_clear", type="primary", use_container_width=True):
            st.session_state.base64_img = []
            st.session_state.t_content = []
            st.session_state.i_content = []
            st.rerun()

    with st.sidebar:
        clear_conversation()
        system_prompt: str = st.text_area("System Prompt", "", key="_inst")
        max_tokens: int = st.slider("Max Tokens", 1, 4096, 4096, 1, key="_tokens")
        temperature: float = st.slider("Temperature", 0.00, 2.00, 0.70, 0.01, key="_temp")
        top_p: float = st.slider("Top P", 0.01, 1.00, 0.95, 0.01, key="_topp")
        top_k: int = st.slider("Top K", 1, 100, 50, 1, key="topk")
        frequency_penalty: float = st.slider("Frequency Penalty", -2.00, 2.00, 0.00, 0.01, key="_freq")
        presence_penalty: float = st.slider("Presence Penalty", -2.00, 2.00, 0.00, 0.01, key="_pres")
    
    types = ["JPG", "JPEG", "PNG"]
    uploaded_images = st.file_uploader("Image Uploader", types, True, "img_uploader")
    if uploaded_images is not None:
        with st.expander("Image", False):
            for uploaded_image in uploaded_images:
                base64_image: str = process_image(uploaded_image)
                st.session_state.base64_img.append(base64_image)
                st.image(uploaded_image)
    else:
        st.session_state.base64_img = []
    
    display_conversation(st.session_state.t_content)

    if query := st.chat_input("Say something...", key="_query"):
        if system_prompt:
            client.chat_completion(
                query=query,
                t_session=st.session_state.t_content,
                i_session=st.session_state.i_content,
                inst=system_prompt,
                base64_images=st.session_state.base64_img,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
        else:
            client.chat_completion(
                query=query,
                t_session=st.session_state.t_content,
                i_session=st.session_state.i_content,
                inst=None,
                base64_images=st.session_state.base64_img,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )

if __name__ == "__main__":
    main()
