"""
Claude-style chat frontend for the RAG Local Business Assistant
(https://github.com/sufyanAshraf/RAG_)

Run:
    1. Start the FastAPI backend first (from the RAG_ repo):
         .\\RAGENV\\Scripts\\python.exe -m uvicorn app.main:app --reload
    2. Then run this file:
         streamlit run streamlit_app.py

The backend exposes: POST / with body {"query": "<text>"} -> {"response": "<text>"}
"""

import requests
import streamlit as st

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
BACKEND_URL = "http://127.0.0.1:8000/"

CITIES = [
    "All cities",
    "Helsinki",
    "Espoo",
    "Vantaa",
    "Tampere",
    "Turku",
    "Oulu",
    "Jyväskylä",
    "Lahti",
    "Kuopio",
    "Pori",
]

st.set_page_config(
    page_title="Royal Hotel",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]

if "city" not in st.session_state:
    st.session_state.city = CITIES[0]

# ----------------------------------------------------------------------
# CSS — hide default Streamlit chrome, build a Claude-like dark theme
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- hide Streamlit's own chrome (sidebar toggle, hamburger, footer) ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {display: none;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #262624;
        color: #ECECE7;
    } 

    .block-container {
        padding-top: 4.5rem;
        padding-bottom: 6rem;
        max-width: 780px;
    }

    /* ---- the ONLY st.columns row in this app doubles as the fixed top bar ----
       (brand text | city select | avatar), all rendered in one flex row so
       they sit on the same line, then pinned to the top like Claude's header. */
    div[data-testid="stHorizontalBlock"] {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 56px;
        background-color: #262624;
        border-bottom: 1px solid #3a3a37;
        z-index: 999;
        align-items: center;
        padding: 0 18px;
    }

    .brand {
        font-size: 15px;
        font-weight: 600;
        color: #ECECE7;
        letter-spacing: 0.2px;
        line-height: 56px; /* vertically center against the 56px bar */
    }

    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: #b5533c;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 700;
        margin-left: auto;  
    }

    /* ---- style the city selectbox so it sits in the top bar like a pill ---- */
    div[data-testid="stSelectbox"] {
        width: 160px;
    }
    div[data-testid="stSelectbox"] > div > div {
        background-color: #2f2f2c !important;
        border: 1px solid #4a4a46 !important;
        border-radius: 16px !important;
        color: #ECECE7 !important;
        min-height: 30px !important;
        font-size: 13px !important;
    }
    div[data-testid="stSelectbox"] label {
        display: none; /* label_visibility="collapsed" already hides it, belt & suspenders */
    }

    /* ---- chat bubbles ---- */
    [data-testid="stChatMessage"] {
        background: transparent;
    }
    .stChatMessage {
        padding: 6px 0;
    }

    /* ---- chat input pinned like Claude's ---- */
    [data-testid="stChatInput"] { 
        border: 1px solid #4a4a46;
        border-radius: 20px;
    }

    /* Chat input bottom area */
    .stBottom > div {
        background-color: #262624 !important;
    } 
    
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Top bar — brand, city select, and avatar all in ONE row so they line up
# on the same line (styled by the CSS above into a fixed header bar).
# The city dropdown replaces the "apps grid" icon from the reference screenshot.
# Streamlit has no native two-way-bound <datalist>, so a styled selectbox
# stands in for it — same "pick from a list of cities" UX.
# ----------------------------------------------------------------------
brand_col, city_col, avatar_col = st.columns([4, 0.1, 0.6])

with brand_col:
    st.markdown('<div class="brand">✦Royal Hotel</div>', unsafe_allow_html=True)

with city_col:
    st.session_state.city = st.selectbox(
        "City",
        CITIES,
        index=CITIES.index(st.session_state.city),
        label_visibility="collapsed",
    )

with avatar_col:
    st.markdown('<div class="avatar">S</div>', unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Chat history
# ----------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        "<div style='text-align:center; margin-top:14vh; color:#8f8f8a; font-size:26px;'>"
        "What can I help you find?</div>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def call_backend(query: str) -> str:
    """POST the query to the FastAPI RAG backend and return its answer text."""
    try:
        resp = requests.post(BACKEND_URL, json={"query": query}, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "(no response field in backend reply)")
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Couldn't reach the backend at "
            f"`{BACKEND_URL}`. Make sure the FastAPI app is running "
            "(`uvicorn app.main:app --reload`)."
        )
    except Exception as exc:  # noqa: BLE001 — surface any backend error to the demo UI
        return f"⚠️ Backend error: {exc}"


# ----------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------
user_input = st.chat_input("Ask about hotels, spas, or restaurants in Finland…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Fold the selected city into the natural-language query so the
    # backend's LLM query parser can pick it up as a filter.
    if st.session_state.city and st.session_state.city != "All cities":
        query_for_backend = f"In {st.session_state.city}: {user_input}"
    else:
        query_for_backend = user_input

    with st.spinner("Thinking…"):
        answer = call_backend(query_for_backend)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()