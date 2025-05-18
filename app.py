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
from dotenv import load\_dotenv
import tempfile
import shutil
from langchain.schema import HumanMessage, AIMessage
from tavily import TavilyClient
from groq import Groq

load\_dotenv()

cohere\_api\_key = os.getenv("COHERE\_API\_KEY")
gemini\_api\_key = os.getenv("GOOGLE\_API\_KEY")
tavily\_api\_key = os.getenv("TAVILY\_API\_KEY")
groq\_api\_key = os.getenv("GROQ\_API\_KEY")

if not cohere\_api\_key or not gemini\_api\_key or not tavily\_api\_key or not groq\_api\_key:
st.error("API keys not found. Please check your .env file")
st.stop()
try:
co = cohere.Client(api\_key=cohere\_api\_key)
genai.configure(api\_key=gemini\_api\_key)
tavily = TavilyClient(api\_key=tavily\_api\_key)
groq = Groq(api\_key=groq\_api\_key)
except Exception as e:
st.error(f"Failed to initialize API clients: {str(e)}")
st.stop()

MAX\_PIXELS=1568 \* 1568
SUPPORTED\_TYPES=\["pdf", "docx", "pptx"]
GEMINI\_MODEL="gemini-1.5-flash"
IMAGE\_QUALITY=95

\#temporaril store files in streamlit server
OUTPUT\_DIR=Path(tempfile.mkdtemp())
IMAGES\_DIR=OUTPUT\_DIR /"images"
TEXT\_FILE=OUTPUT\_DIR /"extracted\_text.txt"

\#remove the exsisiting fules after the clear of the session
def cleanup():
if OUTPUT\_DIR.exists():
shutil.rmtree(OUTPUT\_DIR)

BLANK\_IMAGE\_THRESHOLD=0.95
def is\_blank\_image(pil\_image, threshold=BLANK\_IMAGE\_THRESHOLD):
if pil\_image.mode!='RGB':
pil\_image = pil\_image.convert('RGB')
img\_array = np.array(pil\_image)
dark\_pixels = np.sum((img\_array\[:,:,0] < 50) &
(img\_array\[:,:,1] < 50) &
(img\_array\[:,:,2] < 50))
total\_pixels = img\_array.shape\[0] \* img\_array.shape\[1]
dark\_ratio = dark\_pixels / total\_pixels

```
white_pixels = np.sum((img_array[:,:,0] > 200) & 
                     (img_array[:,:,1] > 200) & 
                     (img_array[:,:,2] > 200))
white_ratio = white_pixels / total_pixels

return dark_ratio > threshold or white_ratio > threshold
```

def save\_image(image\_pil, image\_count):
IMAGES\_DIR.mkdir(parents=True, exist\_ok=True)
img\_path = IMAGES\_DIR / f"image\_{image\_count}.png"
if image\_pil.mode in ('RGBA', 'LA'):
image\_pil = image\_pil.convert('RGB')
image\_pil.save(img\_path, quality=IMAGE\_QUALITY, optimize=True)
return str(img\_path)

def resize\_image(pil\_image,max\_pixels=MAX\_PIXELS):
org\_width, org\_height = pil\_image.size
if org\_width \* org\_height > max\_pixels:
scale\_factor = (max\_pixels / (org\_width \* org\_height)) \*\* 0.5
new\_width = int(org\_width \* scale\_factor)
new\_height = int(org\_height \* scale\_factor)
\# Use high-quality resampling
pil\_image = pil\_image.resize((new\_width, new\_height), Image.Resampling.LANCZOS)
return pil\_image

def base64\_from\_image(img\_path):
try:
pil\_image = Image.open(img\_path)
img\_format = pil\_image.format or "PNG"
if pil\_image.mode in ('RGBA', 'LA'):
pil\_image = pil\_image.convert('RGB')
pil\_image = resize\_image(pil\_image)
with io.BytesIO() as buffer:
pil\_image.save(buffer, format=img\_format, quality=IMAGE\_QUALITY, optimize=True)
encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
return f"data\:image/{img\_format.lower()};base64,{encoded}"
except Exception as e:
st.error(f"Error processing image: {str(e)}")
return None

\#extracting text adn images from pdf
def extract\_pdf(file):
text = ""
image\_paths = \[]
image\_count = 1
try:
with fitz.open(stream=file.read(), filetype="pdf") as pdf:
for page in pdf:
text += page.get\_text()
for img in page.get\_images(full=True):
xref = img\[0]
base\_image = pdf.extract\_image(xref)
img\_bytes = base\_image\["image"]
img\_pil = Image.open(io.BytesIO(img\_bytes))
if is\_blank\_image(img\_pil):
continue
if img\_pil.mode in ('RGBA', 'LA'):
img\_pil = img\_pil.convert('RGB')
img\_path = save\_image(img\_pil, image\_count)
image\_paths.append(img\_path)
image\_count += 1
except Exception as e:
st.error(f"Error processing PDF: {str(e)}")
return text, image\_paths
\#extrating text and images form the docs
def extract\_docx(file):
text = ""
image\_paths = \[]
image\_count = 1
try:
doc = Document(file)
for para in doc.paragraphs:
text += para.text + "\n"
for rel in doc.part.\_rels:
rel\_obj = doc.part.\_rels\[rel]
if "image" in rel\_obj.target\_ref:
img\_data = rel\_obj.target\_part.blob
img\_pil = Image.open(io.BytesIO(img\_data))
if is\_blank\_image(img\_pil):
continue
if img\_pil.mode in ('RGBA', 'LA'):
img\_pil = img\_pil.convert('RGB')
img\_path = save\_image(img\_pil, image\_count)
image\_paths.append(img\_path)
image\_count += 1
except Exception as e:
st.error(f"Error processing DOCX: {str(e)}")
return text, image\_paths
\#extractning text and images from ppt
def extract\_pptx(file):
text = ""
image\_paths = \[]
image\_count = 1
try:
prs = Presentation(file)
for slide in prs.slides:
for shape in slide.shapes:
if hasattr(shape, "text"):
text += shape.text + "\n"
if shape.shape\_type ==13:
img\_stream = shape.image.blob
img\_pil = Image.open(io.BytesIO(img\_stream))
if is\_blank\_image(img\_pil):
continue
if img\_pil.mode in ('RGBA', 'LA'):
img\_pil = img\_pil.convert('RGB')

```
                img_path = save_image(img_pil, image_count)
                image_paths.append(img_path)
                image_count += 1
except Exception as e:
    st.error(f"Error processing PPTX: {str(e)}")
return text, image_paths
```

\#gemini model
def ask\_gemini(question, context=None, img\_path=None):
"""Query Gemini with optional context and/or image"""
try:
model = genai.GenerativeModel(GEMINI\_MODEL)
if img\_path and context:
prompt = f"""You are an expert assistant. Analyze the following question using both the image and the provided context if relevant.

* Use the context and image to answer the question with precision.

* If the image or context is not relevant to the question, provide a general answer.

* Keep the response clear, concise, and informative.
  Context: {context}
  Question: {question}"""
  img = Image.open(img\_path)
  response = model.generate\_content(\[prompt, img])
  elif img\_path:
  prompt = f"""You are a knowledgeable assistant. Carefully analyze the provided image to answer the question below.

* Use the image to answer the question only if it's relevant.

* If the image is not related to the question, provide a general and accurate response based on your knowledge.

Question: {question}"""
img = Image.open(img\_path)
response = model.generate\_content(\[prompt, img])
elif context:
prompt = f"""You are an intelligent assistant. Use the following context to answer the question if it's relevant.

* If the context helps, incorporate it into your response.
* If the question is general or unrelated to the context, answer it independently.
  Context: {context}
  Question: {question}"""
  response = model.generate\_content(prompt)
  else:
  response = model.generate\_content(question)
  return response.text
  except Exception as e:
  return f"Error querying Gemini: {str(e)}"
  \#tavily:- for web search results
  def search\_tavily(query,search\_depth='advanced',max\_results=5):
  """Search the web using Tavily with enhanced parameters"""
  try:
  response = tavily.search(query=query, include\_answer=True, include\_raw\_content=True,include\_sources=True,max\_results=max\_results,search\_depth=search\_depth)
  return response
  except Exception as e:
  st.error(f"Error searching with Tavily: {str(e)}")
  return None
  \#groq for general chat
  def ask\_groq(question, context=None):
  """Query Groq with optional context"""
  try:
  messages = \[]
  if context:
  messages.append({
  "role": "system",
  "content": f"Use this context if relevant: {context}"
  })
  messages.append({
  "role": "user",
  "content": question
  })

  ```
    response = groq.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.8,
    )
    return response.choices[0].message.content
  ```

  except Exception as e:
  return f"Error querying Groq: {str(e)}"

# Streamlit UI

st.set\_page\_config(page\_title="Document Q\&A", layout="wide")
st.title("📄 Document Q\&A with Image Analysis")
if 'text\_chat\_history' not in st.session\_state:
st.session\_state.text\_chat\_history = \[]
if 'image\_chat\_history' not in st.session\_state:
st.session\_state.image\_chat\_history = \[]
if 'general\_chat\_history' not in st.session\_state:
st.session\_state.general\_chat\_history = \[]
if 'scroll' not in st.session\_state:
st.session\_state.scroll = False
if 'processed' not in st.session\_state:
st.session\_state.processed = False
st.session\_state.text = ""
st.session\_state.image\_paths = \[]
st.session\_state.selected\_img = None
if 'prev\_uploaded\_file' not in st.session\_state:
st.session\_state.prev\_uploaded\_file = None

with st.expander("Upload Document", expanded=True):
uploaded\_file = st.file\_uploader("Choose a file", type=SUPPORTED\_TYPES, key="file\_uploader")

if uploaded\_file != st.session\_state.prev\_uploaded\_file:
\# Clear previous content
st.session\_state.processed = False
st.session\_state.text = ""
st.session\_state.image\_paths = \[]
st.session\_state.selected\_img = None
st.session\_state.text\_chat\_history = \[]
st.session\_state.image\_chat\_history = \[]

```
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
```

def render\_chat(container, chat\_history):
with container:
for message in chat\_history:
if isinstance(message, HumanMessage):
st.markdown(
f"<div style='text-align: right; color: white; background-color: #0a84ff; padding: 8px; border-radius: 10px; margin: 5px 0; max-width: 80%; float: right; clear: both;'>{message.content}</div>",
unsafe\_allow\_html=True
)
elif isinstance(message, AIMessage):
st.markdown(
f"<div style='text-align: left; color: black; background-color: #d1d1d1; padding: 8px; border-radius: 10px; margin: 5px 0; max-width: 80%; float: left; clear: both;'>{message.content}</div>",
unsafe\_allow\_html=True
)

# Tabs

tab1, tab2, tab3 = st.tabs(\["📝 Text Analysis", "🖼️ Image Analysis", "💬 General Chat"])

with tab1:
st.subheader("Text Analysis")
if st.session\_state.processed:
with st.expander("View Extracted Text"):
st.text\_area("Extracted Text", st.session\_state.text, height=200, label\_visibility="collapsed")
text\_chat\_container = st.container()
user\_text\_input = st.text\_input("Ask about the text content:", key="text\_input", label\_visibility="collapsed")
text\_send\_button = st.button("Send", key="text\_send")
if text\_send\_button and user\_text\_input:
st.session\_state.text\_chat\_history.append(HumanMessage(content=user\_text\_input))
if user\_text\_input.lower() == 'close the chat':
st.stop()
with st.spinner("Analyzing text..."):
answer = ask\_gemini(user\_text\_input, context=st.session\_state.text)
st.session\_state.text\_chat\_history.append(AIMessage(content=answer))
st.session\_state.scroll = True
st.rerun()
render\_chat(text\_chat\_container, st.session\_state.text\_chat\_history)

with tab2:
st.subheader("Image Analysis")

```
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
```

with tab3:
st.subheader("General Chat")

```
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
```

{search\_results.get('answer', '')}

Relevant links:
{relevant\_links}
"""
\# Generate final answer with search context
if use\_groq:
final\_answer = ask\_groq(
f"Conversation history:\n{conversation\_context}\n\n"
f"Question: {user\_input}\n\n"
f"Here's some additional information that might help answer better:\n"
f"{search\_context}\n\n"
"Please provide an improved answer using this context and conversation history."
)
else:
final\_answer = ask\_gemini(
f"Conversation history:\n{conversation\_context}\n\n"
f"Question: {user\_input}\n\n"
f"Here's some additional information that might help answer better:\n"
f"{search\_context}\n\n"
"Please provide an improved answer using this context and conversation history."
)

```
                    answer = (f"{initial_answer}\n\n"
                              f"I found some updated information:\n{final_answer}")
                else:
                    answer = f"{initial_answer}\n\n  Web search failed to find additional information."
        else:
            answer = initial_answer
        st.session_state.chat_history.append(AIMessage(content=answer))
        st.session_state.scroll = True
        st.rerun()
```

st.session\_state.cleanup = cleanup
