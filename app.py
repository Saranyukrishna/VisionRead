import streamlit as st
import fitz
from docx import Document
from pptx import Presentation
from PIL import Image, ImageEnhance
import io
import os
from pathlib import Path
import base64
import numpy as np
import cohere
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import shutil
from langchain.schema import HumanMessage, AIMessage
from tavily import TavilyClient
from groq import Groq

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not cohere_api_key or not gemini_api_key or not tavily_api_key or not groq_api_key:
    st.error("API keys not found. Please check your .env file")
    st.stop()
try:
    co = cohere.Client(api_key=cohere_api_key)
    genai.configure(api_key=gemini_api_key)
    tavily = TavilyClient(api_key=tavily_api_key)
    groq = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize API clients: {str(e)}")
    st.stop()

MAX_PIXELS=1568 * 1568
SUPPORTED_TYPES=["pdf", "docx", "pptx"]
GEMINI_MODEL="gemini-1.5-flash"
IMAGE_QUALITY=95

# Temporarily store files in streamlit server
OUTPUT_DIR=Path(tempfile.mkdtemp())
IMAGES_DIR=OUTPUT_DIR /"images"
TEXT_FILE=OUTPUT_DIR /"extracted_text.txt"

# Remove the existing files after the clear of the session
def cleanup():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

BLANK_IMAGE_THRESHOLD=0.95
def is_blank_image(pil_image, threshold=BLANK_IMAGE_THRESHOLD):
    if pil_image.mode!='RGB':
        pil_image = pil_image.convert('RGB')
    img_array = np.array(pil_image)
    dark_pixels = np.sum((img_array[:,:,0] < 50) &
                        (img_array[:,:,1] < 50) &
                        (img_array[:,:,2] < 50))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    dark_ratio = dark_pixels / total_pixels

    white_pixels = np.sum((img_array[:,:,0] > 200) & 
                     (img_array[:,:,1] > 200) & 
                     (img_array[:,:,2] > 200))
    white_ratio = white_pixels / total_pixels

    return dark_ratio > threshold or white_ratio > threshold

def save_image(image_pil, image_count):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    img_path = IMAGES_DIR / f"image_{image_count}.png"
    if image_pil.mode in ('RGBA', 'LA'):
        image_pil = image_pil.convert('RGB')
    image_pil.save(img_path, quality=IMAGE_QUALITY, optimize=True)
    return str(img_path)

def resize_image(pil_image,max_pixels=MAX_PIXELS):
    org_width, org_height = pil_image.size
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        # Use high-quality resampling
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return pil_image

def base64_from_image(img_path):
    try:
        pil_image = Image.open(img_path)
        img_format = pil_image.format or "PNG"
        if pil_image.mode in ('RGBA', 'LA'):
            pil_image = pil_image.convert('RGB')
        pil_image = resize_image(pil_image)
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format=img_format, quality=IMAGE_QUALITY, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/{img_format.lower()};base64,{encoded}"
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Extracting text and images from pdf
def extract_pdf(file):
    text = ""
    image_paths = []
    image_count = 1
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    img_bytes = base_image["image"]
                    img_pil = Image.open(io.BytesIO(img_bytes))
                    if is_blank_image(img_pil):
                        continue
                    if img_pil.mode in ('RGBA', 'LA'):
                        img_pil = img_pil.convert('RGB')
                    img_path = save_image(img_pil, image_count)
                    image_paths.append(img_path)
                    image_count += 1
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
    return text, image_paths

# Extracting text and images from docx
def extract_docx(file):
    text = ""
    image_paths = []
    image_count = 1
    try:
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
        for rel in doc.part._rels:
            rel_obj = doc.part._rels[rel]
            if "image" in rel_obj.target_ref:
                img_data = rel_obj.target_part.blob
                img_pil = Image.open(io.BytesIO(img_data))
                if is_blank_image(img_pil):
                    continue
                if img_pil.mode in ('RGBA', 'LA'):
                    img_pil = img_pil.convert('RGB')
                img_path = save_image(img_pil, image_count)
                image_paths.append(img_path)
                image_count += 1
    except Exception as e:
        st.error(f"Error processing DOCX: {str(e)}")
    return text, image_paths

# Extracting text and images from ppt
def extract_pptx(file):
    text = ""
    image_paths = []
    image_count = 1
    try:
        prs = Presentation(file)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
                if shape.shape_type ==13:
                    img_stream = shape.image.blob
                    img_pil = Image.open(io.BytesIO(img_stream))
                    if is_blank_image(img_pil):
                        continue
                    if img_pil.mode in ('RGBA', 'LA'):
                        img_pil = img_pil.convert('RGB')
                    img_path = save_image(img_pil, image_count)
                    image_paths.append(img_path)
                    image_count += 1
    except Exception as e:
        st.error(f"Error processing PPTX: {str(e)}")
    return text, image_paths

# Gemini model
def ask_gemini(question, context=None, img_path=None):
    """Query Gemini with optional context and/or image"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        if img_path and context:
            prompt = f"""You are an expert assistant. Analyze the following question using both the image and the provided context if relevant.

* Use the context and image to answer the question with precision.

* If the image or context is not relevant to the question, provide a general answer.

* Keep the response clear, concise, and informative.
Context: {context}
Question: {question}"""
            img = Image.open(img_path)
            response = model.generate_content([prompt, img])
        elif img_path:
            prompt = f"""You are a knowledgeable assistant. Carefully analyze the provided image to answer the question below.

* Use the image to answer the question only if it's relevant.

* If the image is not related to the question, provide a general and accurate response based on your knowledge.

Question: {question}"""
            img = Image.open(img_path)
            response = model.generate_content([prompt, img])
        elif context:
            prompt = f"""You are an intelligent assistant. Use the following context to answer the question if it's relevant.

* If the context helps, incorporate it into your response.
* If the question is general or unrelated to the context, answer it independently.
Context: {context}
Question: {question}"""
            response = model.generate_content(prompt)
        else:
            response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"

# Tavily for web search results
def search_tavily(query,search_depth='advanced',max_results=5):
    """Search the web using Tavily with enhanced parameters"""
    try:
        response = tavily.search(query=query, include_answer=True, include_raw_content=True,include_sources=True,max_results=max_results,search_depth=search_depth)
        return response
    except Exception as e:
        st.error(f"Error searching with Tavily: {str(e)}")
        return None

# Groq for general chat
def ask_groq(question, context=None):
    """Query Groq with optional context"""
    try:
        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"Use this context if relevant: {context}"
            })
        messages.append({
            "role": "user",
            "content": question
        })

        response = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.8,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying Groq: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Document Q&A with Image Analysis")
if 'text_chat_history' not in st.session_state:
    st.session_state.text_chat_history = []
if 'image_chat_history' not in st.session_state:
    st.session_state.image_chat_history = []
if 'general_chat_history' not in st.session_state:
    st.session_state.general_chat_history = []
if 'scroll' not in st.session_state:
    st.session_state.scroll = False
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.text = ""
    st.session_state.image_paths = []
    st.session_state.selected_img = None
if 'prev_uploaded_file' not in st.session_state:
    st.session_state.prev_uploaded_file = None

with st.expander("Upload Document", expanded=True):
    uploaded_file = st.file_uploader("Choose a file", type=SUPPORTED_TYPES, key="file_uploader")

if uploaded_file != st.session_state.prev_uploaded_file:
    # Clear previous content
    st.session_state.processed = False
    st.session_state.text = ""
    st.session_state.image_paths = []
    st.session_state.selected_img = None
    st.session_state.text_chat_history = []
    st.session_state.image_chat_history = []
    cleanup()

if uploaded_file:
    with st.spinner("Extracting content from document..."):
        file_ext = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_ext == "pdf":
                st.session_state.text, st.session_state.image_paths = extract_pdf(uploaded_file)
            elif file_ext == "docx":
                st.session_state.text, st.session_state.image_paths = extract_docx(uploaded_file)
            elif file_ext == "pptx":
                st.session_state.text, st.session_state.image_paths = extract_pptx(uploaded_file)
            
            OUTPUT_DIR.mkdir(exist_ok=True)
            with open(TEXT_FILE, "w", encoding="utf-8") as f:
                f.write(st.session_state.text)
            
            st.session_state.processed = True
            st.session_state.prev_uploaded_file = uploaded_file
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Failed to process document: {str(e)}")
            cleanup()

def render_chat(container, chat_history):
    with container:
        for message in chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(
                    f"<div style='text-align: right; color: white; background-color: #0a84ff; padding: 8px; border-radius: 10px; margin: 5px 0; max-width: 80%; float: right; clear: both;'>{message.content}</div>",
                    unsafe_allow_html=True
                )
            elif isinstance(message, AIMessage):
                st.markdown(
                    f"<div style='text-align: left; color: black; background-color: #d1d1d1; padding: 8px; border-radius: 10px; margin: 5px 0; max-width: 80%; float: left; clear: both;'>{message.content}</div>",
                    unsafe_allow_html=True
                )

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "💬 General Chat"])

with tab1:
    st.subheader("Text Analysis")
    if st.session_state.processed:
        with st.expander("View Extracted Text"):
            st.text_area("Extracted Text", st.session_state.text, height=200, label_visibility="collapsed")
        text_chat_container = st.container()
        user_text_input = st.text_input("Ask about the text content:", key="text_input", label_visibility="collapsed")
        text_send_button = st.button("Send", key="text_send")
        if text_send_button and user_text_input:
            st.session_state.text_chat_history.append(HumanMessage(content=user_text_input))
            if user_text_input.lower() == 'close the chat':
                st.stop()
            with st.spinner("Analyzing text..."):
                answer = ask_gemini(user_text_input, context=st.session_state.text)
                st.session_state.text_chat_history.append(AIMessage(content=answer))
                st.session_state.scroll = True
                st.rerun()
        render_chat(text_chat_container, st.session_state.text_chat_history)

with tab2:
    st.subheader("Image Analysis")

    # Image selection and display at the top
    img_col, _ = st.columns([1, 3])
    with img_col:
        if st.session_state.selected_img:
            try:
                selected_img = Image.open(st.session_state.selected_img)
                
                with st.expander("Image Enhancement Options"):
                    enhance = st.checkbox("Enhance Image Quality", value=True)
                    contrast = st.slider("Adjust Contrast", 0.5, 2.0, 1.0)
                    sharpness = st.slider("Adjust Sharpness", 0.0, 2.0, 1.0)
                    
                    if enhance:
                        enhancer = ImageEnhance.Contrast(selected_img)
                        selected_img = enhancer.enhance(contrast)
                        enhancer = ImageEnhance.Sharpness(selected_img)
                        selected_img = enhancer.enhance(sharpness)
                st.image(selected_img, 
                         caption="Selected Image", 
                         use_container_width=True,
                         output_format="PNG")
                with io.BytesIO() as buffer:
                    selected_img.save(buffer, format="PNG", quality=IMAGE_QUALITY)
                    st.download_button(
                        label="Download Enhanced Image",
                        data=buffer.getvalue(),
                        file_name="enhanced_image.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"Error loading selected image: {str(e)}")
        else:
            st.info("No image selected")

    if st.session_state.selected_img:
        image_chat_container = st.container()
        render_chat(image_chat_container, st.session_state.image_chat_history)

        # Input section at the bottom
        input_col = st.container()
        with input_col:
            st.write("**Ask about the image**")
            user_image_input = st.text_input(
                "Ask about the image:", 
                key="image_input", 
                placeholder="Type your question here...",
                label_visibility="collapsed",
                disabled=not st.session_state.selected_img
            )
            image_send_button = st.button("Send", key="image_send", disabled=not st.session_state.selected_img)
        
        # Handle user input and generate responses
        if image_send_button and user_image_input:
            st.session_state.image_chat_history.append(HumanMessage(content=user_image_input))
            if user_image_input.lower() == 'close the chat':
                st.stop()
            
            with st.spinner("Analyzing image..."):
                answer = ask_gemini(
                    user_image_input, 
                    img_path=st.session_state.selected_img, 
                    context=st.session_state.text
                )
                st.session_state.image_chat_history.append(AIMessage(content=answer))
                st.session_state.scroll = True
                st.rerun()

    if st.session_state.processed and st.session_state.image_paths:
        st.divider()
        st.write("Select an image to analyze:")
        num_cols = 4
        image_paths = st.session_state.image_paths
        rows = (len(image_paths) + num_cols - 1) // num_cols
        
        for row in range(rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                img_idx = row * num_cols + col_idx
                if img_idx < len(image_paths):
                    img_path = image_paths[img_idx]
                    with cols[col_idx]:
                        try:
                            img = Image.open(img_path)
                            if not is_blank_image(img):  # Only display non-blank images
                                st.image(img, use_container_width=True, output_format="PNG")
                                if st.button(f"Select {img_idx+1}", key=f"btn_{img_idx}"):
                                    st.session_state.selected_img = img_path
                                    st.session_state.image_chat_history = []  # Clear chat when new image selected
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Error loading image: {str(e)}")
    else:
        st.write("No images found in the document.")

with tab3:
    st.subheader("General Chat")

    if 'first_load_done' not in st.session_state:
        st.session_state.first_load_done = True
        st.session_state.chat_history = []

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns(2)
    with col1:
        use_groq = st.toggle("Use Groq (faster)", value=True)
    with col2:
        enable_search = st.toggle("Enable web search", value=True)

    general_chat_container = st.container()
    render_chat(general_chat_container, st.session_state.chat_history)

    # Input section
    user_input = st.text_input(
        "Ask any question:", 
        key="general_input", 
        label_visibility="collapsed",
        placeholder="Type your message here..."
    )

    if st.button("Send", key="general_send") and user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        if user_input.lower() == 'clear chat':
            st.session_state.chat_history = []
            st.rerun()

        with st.spinner("Thinking..."):
            conversation_context = "\n".join(
                f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
                for msg in st.session_state.chat_history[-10:]
            )
            if use_groq:
                initial_answer = ask_groq(
                    f"Conversation history:\n{conversation_context}\n\n"
                    f"New question: {user_input}\n\n"
                    "Please answer the new question considering the conversation history."
                )
            else:
                initial_answer = ask_gemini(
                    f"Conversation history:\n{conversation_context}\n\n"
                    f"New question: {user_input}\n\n"
                    "Please answer the new question considering the conversation history."
                )
            needs_search = (
                enable_search and 
                ("I don't know" in initial_answer or 
                 "not sure" in initial_answer or 
                 "as of my knowledge" in initial_answer or
                 "current information" in initial_answer or
                 any(word in user_input.lower() for word in ["current", "recent", "today", "now", "202", "update"]))
            )

            if needs_search:
                with st.spinner("Searching for current information..."):
                    search_results = search_tavily(user_input)
                    if search_results:
                        relevant_links = "\n".join(
                            f"{i+1}. {result['title']} - {result['url']}" 
                            for i, result in enumerate(search_results.get('results', [])[:3])
                        )

                        search_context = f"""Web search results:
{search_results.get('answer', '')}

Relevant links:
{relevant_links}
"""
                        # Generate final answer with search context
                        if use_groq:
                            final_answer = ask_groq(
                                f"Conversation history:\n{conversation_context}\n\n"
                                f"Question: {user_input}\n\n"
                                f"Here's some additional information that might help answer better:\n"
                                f"{search_context}\n\n"
                                "Please provide an improved answer using this context and conversation history."
                            )
                        else:
                            final_answer = ask_gemini(
                                f"Conversation history:\n{conversation_context}\n\n"
                                f"Question: {user_input}\n\n"
                                f"Here's some additional information that might help answer better:\n"
                                f"{search_context}\n\n"
                                "Please provide an improved answer using this context and conversation history."
                            )

                        answer = (f"{initial_answer}\n\n"
                                  f"I found some updated information:\n{final_answer}")
                    else:
                        answer = f"{initial_answer}\n\nWeb search failed to find additional information."
            else:
                answer = initial_answer
            st.session_state.chat_history.append(AIMessage(content=answer))
            st.session_state.scroll = True
            st.rerun()

st.session_state.cleanup = cleanup
