"""
ui/styles.py — Custom CSS injection for the RAG QA Streamlit application.

Exposes a single `inject()` function that injects the premium dark theme
styles into the Streamlit app via st.markdown.
"""

import streamlit as st


def inject() -> None:
    """Inject the full custom CSS block into the Streamlit app."""
    st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* Global */
  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* Background */
  .stApp {
      background: linear-gradient(135deg, #0d0f1a 0%, #111827 50%, #0d1b2a 100%);
      color: #e2e8f0;
  }

  /* Main header */
  .main-header {
      background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
      border: 1px solid rgba(99,102,241,0.3);
      border-radius: 16px;
      padding: 24px 32px;
      margin-bottom: 24px;
      backdrop-filter: blur(10px);
      text-align: center;
  }
  .main-header h1 {
      background: linear-gradient(90deg, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-size: 2.5rem;
      font-weight: 700;
      margin: 0;
  }
  .main-header p {
      color: #94a3b8;
      margin: 8px 0 0 0;
      font-size: 1.05rem;
  }

  /* Status badges */
  .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 500;
      margin-right: 12px;
  }
  .status-ok {
      background: rgba(16,185,129,0.15);
      border: 1px solid rgba(16,185,129,0.4);
      color: #6ee7b7;
  }
  .status-warn {
      background: rgba(245,158,11,0.15);
      border: 1px solid rgba(245,158,11,0.4);
      color: #fcd34d;
  }
  .status-error {
      background: rgba(239,68,68,0.15);
      border: 1px solid rgba(239,68,68,0.4);
      color: #fca5a5;
  }

  /* Chat messages — clean, borderless bubbles */
  .stChatMessage {
      border-radius: 0 !important;
      margin-bottom: 4px !important;
      padding: 6px 0 !important;
      background: transparent !important;
  }
  [data-testid="stChatMessageContent"] {
      background: transparent !important;
      border: none !important;
      border-radius: 0 !important;
      padding: 4px 8px !important;
      color: #e2e8f0 !important;
      font-size: 0.97rem !important;
      line-height: 1.65 !important;
  }

  /* Avatar icon — orange-red circular badge */
  [data-testid="stChatMessageAvatar"] {
      width: 36px !important;
      height: 36px !important;
      min-width: 36px !important;
      border-radius: 50% !important;
      background: linear-gradient(135deg, #f97316 0%, #dc2626 100%) !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      box-shadow: 0 2px 8px rgba(220, 38, 38, 0.45) !important;
      overflow: hidden !important;
      font-size: 1rem !important;
  }
  [data-testid="stChatMessageAvatar"] img,
  [data-testid="stChatMessageAvatar"] svg {
      width: 22px !important;
      height: 22px !important;
      filter: brightness(0) invert(1) !important;
  }

  /* Hover highlight on message row */
  .stChatMessage:hover {
      background: rgba(255,255,255,0.015) !important;
      border-radius: 10px !important;
  }

  /* Source card */
  .source-card {
      background: rgba(99,102,241,0.08);
      border: 1px solid rgba(99,102,241,0.25);
      border-radius: 10px;
      padding: 14px 18px;
      margin-top: 10px;
      transition: border-color 0.2s;
  }
  .source-card:hover {
      border-color: rgba(99,102,241,0.5);
  }
  .source-card .source-title {
      font-size: 0.85rem;
      font-weight: 600;
      color: #a5b4fc;
      margin-bottom: 6px;
  }
  .source-card .source-meta {
      font-size: 0.8rem;
      color: #64748b;
  }
  .source-card .source-snippet {
      font-size: 0.85rem;
      color: #94a3b8;
      margin-top: 8px;
      line-height: 1.6;
      border-left: 2px solid rgba(99,102,241,0.4);
      padding-left: 12px;
  }
  .relevance-pill {
      display: inline-block;
      background: rgba(139,92,246,0.2);
      border: 1px solid rgba(139,92,246,0.4);
      color: #c4b5fd;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 0.72rem;
      font-weight: 600;
  }
  .rerank-pill {
      display: inline-block;
      background: rgba(16,185,129,0.15);
      border: 1px solid rgba(16,185,129,0.4);
      color: #6ee7b7;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 0.72rem;
      font-weight: 600;
  }

  /* Stat tile */
  .stat-tile {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 24px;
      text-align: center;
      transition: border-color 0.2s, transform 0.2s;
  }
  .stat-tile:hover {
      border-color: rgba(99,102,241,0.4);
      transform: translateY(-2px);
  }
  .stat-number {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(90deg, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
  }
  .stat-label {
      font-size: 0.9rem;
      color: #64748b;
      margin-top: 8px;
      font-weight: 500;
  }

  /* Divider */
  hr {
      border-color: rgba(255,255,255,0.08) !important;
      margin: 2rem 0 !important;
  }

  /* Buttons */
  .stButton > button {
      background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
      color: white !important;
      border: none !important;
      border-radius: 10px !important;
      font-weight: 600 !important;
      padding: 0.5rem 1rem !important;
      transition: opacity 0.2s, transform 0.2s, box-shadow 0.2s !important;
  }
  .stButton > button:hover {
      opacity: 0.9 !important;
      transform: translateY(-2px) !important;
      box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
  }

  /* Primary ingest button (make it bigger in main view) */
  div[data-testid="stVerticalBlock"] > div > div > div > div.stButton > button {
      padding: 0.75rem 1.5rem !important;
      font-size: 1.1rem !important;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
      background: rgba(255,255,255,0.03);
      border-radius: 12px;
      padding: 6px;
      gap: 8px;
  }
  .stTabs [data-baseweb="tab"] {
      border-radius: 8px !important;
      color: #94a3b8 !important;
      font-size: 1.05rem !important;
      font-weight: 600 !important;
      padding: 12px 24px !important;
  }
  .stTabs [aria-selected="true"] {
      background: rgba(99,102,241,0.2) !important;
      color: #a5b4fc !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
      background: rgba(255,255,255,0.04) !important;
      border-radius: 8px !important;
      color: #94a3b8 !important;
      font-weight: 500 !important;
  }

  /* Input fields */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
      background: rgba(255,255,255,0.05) !important;
      border: 1px solid rgba(255,255,255,0.1) !important;
      color: #e2e8f0 !important;
      border-radius: 8px !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
      border-color: rgba(99,102,241,0.5) !important;
      box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
  }

  /* File uploader dropzone */
  [data-testid="stFileUploader"] {
      background: rgba(255,255,255,0.02) !important;
      border: 2px dashed rgba(99,102,241,0.5) !important;
      border-radius: 16px !important;
      padding: 24px !important;
      transition: background 0.2s, border-color 0.2s !important;
  }
  [data-testid="stFileUploader"]:hover {
      background: rgba(99,102,241,0.04) !important;
      border-color: rgba(139,92,246,0.8) !important;
  }

  /* Chat input container */
  .stChatInputContainer {
      background: rgba(20,22,34,0.92) !important;
      border: 1px solid rgba(249,115,22,0.35) !important;
      border-radius: 14px !important;
      backdrop-filter: blur(14px) !important;
      box-shadow: 0 0 0 1px rgba(249,115,22,0.08), 0 4px 24px rgba(0,0,0,0.4) !important;
  }
  .stChatInputContainer:focus-within {
      border-color: rgba(249,115,22,0.65) !important;
      box-shadow: 0 0 0 2px rgba(249,115,22,0.12), 0 4px 24px rgba(0,0,0,0.4) !important;
  }
  .stChatInputContainer textarea {
      color: #e2e8f0 !important;
      background: transparent !important;
  }

  /* Hide streamlit branding and sidebar toggle */
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stSidebarCollapsedControl"] { display: none; }

  /* Confidence badge — hallucination detector */
  .confidence-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 12px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 500;
      margin-top: 8px;
      margin-bottom: 4px;
  }
  .confidence-high {
      background: rgba(16,185,129,0.12);
      border: 1px solid rgba(16,185,129,0.4);
      color: #6ee7b7;
  }
  .confidence-medium {
      background: rgba(245,158,11,0.12);
      border: 1px solid rgba(245,158,11,0.4);
      color: #fcd34d;
  }
  .confidence-low {
      background: rgba(239,68,68,0.12);
      border: 1px solid rgba(239,68,68,0.4);
      color: #fca5a5;
  }
</style>
""", unsafe_allow_html=True)
