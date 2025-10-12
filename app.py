import streamlit as st
import torch
import torch.nn as nn
import torch.serialization
import math
import re
import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Mira - Empathetic AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Enhanced Custom CSS with Dark Theme & Neon Accents ==========
st.markdown("""
<style>
    /* Remove default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Remove default padding and spacing */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }
    
    /* Remove spacing between elements */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div {
        gap: 0rem !important;
    }
    
    div[data-testid="stVerticalBlock"] {
        gap: 0rem !important;
    }
    
    /* Hide all empty elements */
    .element-container:has(> .stMarkdown:empty),
    .element-container:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* CRITICAL: Show and style sidebar collapse button when collapsed */
    button[kind="header"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: linear-gradient(135deg, #00f5ff, #7b2ff7) !important;
        border-radius: 0 12px 12px 0 !important;
        padding: 0.75rem !important;
        margin-top: 1rem !important;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.4) !important;
        transition: all 0.3s ease !important;
        border: none !important;
        color: white !important;
    }
    
    button[kind="header"]:hover {
        box-shadow: 0 6px 25px rgba(0, 245, 255, 0.7) !important;
        transform: translateX(3px) !important;
        background: linear-gradient(135deg, #00d4ff, #6a1fd7) !important;
    }
    
    button[kind="header"] svg {
        color: #ffffff !important;
        width: 1.5rem !important;
        height: 1.5rem !important;
    }
    
    /* Sidebar close button when open */
    section[data-testid="stSidebar"] button[kind="header"] {
        color: #00f5ff !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    
    section[data-testid="stSidebar"] button[kind="header"]:hover {
        color: #ffffff !important;
        background: rgba(0, 245, 255, 0.1) !important;
    }
    
    /* Main header with neon glow */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00f5ff, #7b2ff7, #f72585);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        margin-top: 0;
        padding-top: 0;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        letter-spacing: 2px;
        line-height: 1.2;
    }
    
    .subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1rem;
        margin-bottom: 0;
        margin-top: 0;
        padding: 0;
    }
    
    /* Chat container - WhatsApp style */
    .chat-container {
        background: #0d1117;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0 1rem 0;
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #30363d;
        box-shadow: 0 8px 32px rgba(0, 245, 255, 0.1);
    }
    
    /* Custom scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #161b22;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00f5ff, #7b2ff7);
        border-radius: 10px;
    }
    
    /* Message bubbles */
    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        margin: 0.8rem 0;
        max-width: 75%;
        animation: fadeIn 0.3s ease-in;
        word-wrap: break-word;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #7b2ff7, #f72585);
        margin-left: auto;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(123, 47, 247, 0.3);
        border-bottom-right-radius: 4px;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #1e2a3a, #2d3748);
        margin-right: auto;
        color: #e2e8f0;
        border: 1px solid #30363d;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.1);
        border-bottom-left-radius: 4px;
    }
    
    .message-label {
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        opacity: 0.8;
    }
    
    .user-label {
        color: #ffd6f3;
        text-align: right;
    }
    
    .bot-label {
        color: #00f5ff;
    }
    
    /* Input area styling */
    .stTextInput > div > div > input {
        background: #161b22 !important;
        border: 2px solid #30363d !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00f5ff !important;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.3) !important;
    }
    
    .stTextInput > label {
        color: #8892b0 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Button styling with neon glow */
    .stButton > button {
        background: linear-gradient(135deg, #00f5ff, #7b2ff7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0, 245, 255, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(0, 245, 255, 0.6) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #00f5ff !important;
        font-weight: 700 !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: #161b22 !important;
        border: 2px solid #30363d !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    
    .stSelectbox label {
        color: #8892b0 !important;
        font-weight: 600 !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #8892b0 !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        color: #e2e8f0 !important;
    }
    
    /* Slider */
    .stSlider > label {
        color: #8892b0 !important;
        font-weight: 600 !important;
    }
    
    /* Info box in sidebar */
    .sidebar-info {
        background: linear-gradient(135deg, #1e2a3a, #2d3748);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #30363d;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.1);
    }
    
    .sidebar-info p {
        color: #8892b0 !important;
        margin: 0.3rem 0 !important;
        font-size: 0.9rem;
    }
    
    .sidebar-info strong {
        color: #00f5ff !important;
    }
    
    /* Divider */
    hr {
        border-color: #30363d !important;
        opacity: 0.3;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #00f5ff transparent transparent transparent !important;
    }
    
    /* Empty state message */
    .empty-chat {
        text-align: center;
        padding: 3rem;
        color: #8892b0;
    }
    
    .empty-chat-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    /* Input container */
    .input-section {
        background: #0d1117;
        padding: 1.5rem;
        border-radius: 20px;
        margin-top: 0.5rem;
        border: 1px solid #30363d;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Hide default streamlit elements */
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
</style>
""", unsafe_allow_html=True)

# ========== Model Architecture Components ==========
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(max_len, 1, emb_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        
        self.fc_q = nn.Linear(emb_dim, emb_dim)
        self.fc_k = nn.Linear(emb_dim, emb_dim)
        self.fc_v = nn.Linear(emb_dim, emb_dim)
        self.fc_o = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.emb_dim)
        x = self.fc_o(x)
        
        return x, attention

class PositionwiseFeedforward(nn.Module):
    def __init__(self, emb_dim, ff_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(emb_dim, ff_dim)
        self.fc_2 = nn.Linear(ff_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc_1(x))
        x = self.dropout(x)
        x = self.fc_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(emb_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(_src))
        _src = self.feed_forward(src)
        src = self.norm2(src + self.dropout(_src))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.encoder_attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(emb_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attn(trg, enc_src, enc_src, src_mask)
        trg = self.norm2(trg + self.dropout(_trg))
        _trg = self.feed_forward(trg)
        trg = self.norm3(trg + self.dropout(_trg))
        return trg, attention

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, num_layers, num_heads, ff_dim, dropout, device):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = PositionalEncoding(emb_dim, dropout)
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)
    
    def forward(self, src, src_mask):
        tok_embedded = self.tok_embedding(src) * self.scale
        pos_embedded = self.pos_embedding(tok_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        src = self.dropout(pos_embedded)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, num_layers, num_heads, ff_dim, dropout, device):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = PositionalEncoding(emb_dim, dropout)
        self.layers = nn.ModuleList([DecoderLayer(emb_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        tok_embedded = self.tok_embedding(trg) * self.scale
        pos_embedded = self.pos_embedding(tok_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        trg = self.dropout(pos_embedded)
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

# ========== Helper Functions ==========
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text

def simple_tokenizer(s):
    return s.split()

@st.cache_resource
def download_model_files():
    """Download model and vocab files from Kaggle dataset"""
    try:
        if "KAGGLE_USERNAME" not in st.secrets or "KAGGLE_KEY" not in st.secrets:
            raise ValueError("Kaggle secrets not found. Check Streamlit Cloud secrets.")
        
        username = st.secrets["KAGGLE_USERNAME"]
        key = st.secrets["KAGGLE_KEY"]
        
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
        
        kaggle_json = {"username": username, "key": key}
        with open(kaggle_json_path, "w") as f:
            json.dump(kaggle_json, f)
        
        os.chmod(kaggle_json_path, 0o600)
        
        api = KaggleApi()
        api.authenticate()
        
        dataset_name = "nadeemahmad003/chatbot-data"
        if not os.path.exists("model_files"):
            os.makedirs("model_files")
        
        api.dataset_download_files(dataset_name, path="model_files", unzip=True)
        
        downloaded_files = os.listdir("model_files")
        
        if "vocab.pth" not in downloaded_files:
            raise FileNotFoundError("vocab.pth not found after download.")
        
        return True
    except Exception as e:
        st.error(f"Error downloading model files: {str(e)}")
        return False

@st.cache_resource
def load_model_and_vocab():
    """Load the trained model and vocabulary"""
    try:
        if not os.path.exists("model_files/vocab.pth"):
            with st.spinner("Downloading model files from Kaggle..."):
                if not download_model_files():
                    raise RuntimeError("Download failed.")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        vocab = torch.load('model_files/vocab.pth', map_location=device, weights_only=False)
        
        if isinstance(vocab, str):
            raise ValueError(f"Expected vocab object, but loaded a string: {vocab}")
        
        if not hasattr(vocab, '__getitem__'):
            raise ValueError(f"Loaded vocab object is not usable: {type(vocab)}")
        
        VOCAB_SIZE = len(vocab)
        EMBED_DIM = 512
        NUM_HEADS = 4
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3
        DROPOUT = 0.1
        FF_DIM = 4 * EMBED_DIM
        PAD_IDX = 1
        
        encoder = Encoder(VOCAB_SIZE, EMBED_DIM, NUM_ENCODER_LAYERS, NUM_HEADS, FF_DIM, DROPOUT, device)
        decoder = Decoder(VOCAB_SIZE, EMBED_DIM, NUM_DECODER_LAYERS, NUM_HEADS, FF_DIM, DROPOUT, device)
        model = Seq2SeqTransformer(encoder, decoder, PAD_IDX, PAD_IDX, device).to(device)
        
        model.load_state_dict(torch.load('model_files/best-model-v4-stable.pt', map_location=device, weights_only=True))
        model.eval()
        
        return model, vocab, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def greedy_decode(model, vocab, src_sentence, device, max_len=50):
    """Greedy decoding for inference"""
    model.eval()
    
    BOS_IDX = vocab['<bos>']
    EOS_IDX = vocab['<eos>']
    
    tokens = simple_tokenizer(src_sentence)
    src_indexes = [BOS_IDX] + [vocab[token] for token in tokens] + [EOS_IDX]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    trg_indexes = [BOS_IDX]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token_idx = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token_idx)
        
        if pred_token_idx == EOS_IDX:
            break
    
    trg_tokens = vocab.lookup_tokens(trg_indexes)
    response = " ".join(trg_tokens)
    response = response.replace("<bos>", "").replace("<eos>", "").strip()
    
    return response

def beam_search_decode(model, vocab, src_sentence, device, beam_width=3, max_len=50):
    """Beam search decoding for inference"""
    import torch.nn.functional as F
    
    model.eval()
    
    BOS_IDX = vocab['<bos>']
    EOS_IDX = vocab['<eos>']
    
    tokens = simple_tokenizer(src_sentence)
    src_indexes = [BOS_IDX] + [vocab[token] for token in tokens] + [EOS_IDX]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    beams = [([BOS_IDX], 0.0)]
    completed_beams = []
    
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == EOS_IDX:
                completed_beams.append((seq, score))
                continue
            
            trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            
            with torch.no_grad():
                output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            
            log_probs = F.log_softmax(output[:, -1, :], dim=-1)
            top_log_probs, top_idxs = torch.topk(log_probs, beam_width)
            
            for i in range(beam_width):
                new_seq = seq + [top_idxs[0][i].item()]
                new_score = score + top_log_probs[0][i].item()
                new_beams.append((new_seq, new_score))
        
        if not new_beams:
            break
        
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    
    completed_beams.extend(beams)
    best_beam = sorted(completed_beams, key=lambda x: x[1] / len(x[0]), reverse=True)[0]
    best_seq = best_beam[0]
    
    trg_tokens = vocab.lookup_tokens(best_seq)
    response = " ".join(trg_tokens)
    response = response.replace("<bos>", "").replace("<eos>", "").strip()
    
    return response

# ========== Main Application ==========
def main():
    # Header - direct markdown without extra containers
    st.markdown('<div class="main-header">✨ MIRA</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your Empathetic AI Companion</div>', unsafe_allow_html=True)
    
    # Load model
    model, vocab, device = load_model_and_vocab()
    
    if model is None:
        st.error("Failed to load model. Check Streamlit Cloud logs for errors.")
        st.info("Verify Kaggle credentials and dataset accessibility.")
        return
    
    # Sidebar Settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        
        emotions = ["afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive", 
                   "ashamed", "caring", "confident", "content", "devastated", "disappointed",
                   "disgusted", "embarrassed", "excited", "faithful", "furious", "grateful",
                   "guilty", "hopeful", "impressed", "jealous", "joyful", "lonely", "nostalgic",
                   "prepared", "proud", "sad", "sentimental", "surprised", "terrified", "trusting"]
        
        selected_emotion = st.selectbox("Emotion Context", ["auto"] + emotions)
        
        decoding_method = st.radio("Decoding Strategy", ["Greedy Search", "Beam Search"])
        
        beam_width = 3
        if decoding_method == "Beam Search":
            beam_width = st.slider("Beam Width", 2, 5, 3)
        
        st.markdown("---")
        
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown(f"**Device:** {device}")
        st.markdown(f"**Vocab Size:** {len(vocab):,}")
        st.markdown(f"**Model:** Transformer Seq2Seq")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat Display Area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="empty-chat">
            <div class="empty-chat-icon">💬</div>
            <p>Start a conversation with Mira</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-label user-label">You</div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="message-label bot-label">Mira</div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    situation = st.text_input("📍 Situation (Optional)", placeholder="Describe the context or situation...", key="situation_input")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("💭 Your Message", placeholder="Type your message here...", key="user_input", label_visibility="collapsed")
    
    with col2:
        send_button = st.button("Send", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle message sending
    if send_button and user_input:
        normalized_input = normalize_text(user_input)
        
        if selected_emotion == "auto":
            emotion = "neutral"
        else:
            emotion = selected_emotion
        
        if situation.strip():
            normalized_situation = normalize_text(situation)
        else:
            normalized_situation = "general conversation"
        
        input_text = f"Emotion: {emotion} | Situation: {normalized_situation} | Customer: {normalized_input} Agent:"
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Mira is thinking..."):
            if decoding_method == "Greedy Search":
                response = greedy_decode(model, vocab, input_text, device)
            else:
                response = beam_search_decode(model, vocab, input_text, device, beam_width=beam_width)
        
        st.session_state.messages.append({"role": "bot", "content": response})
        
        st.rerun()

if __name__ == "__main__":
    main()
