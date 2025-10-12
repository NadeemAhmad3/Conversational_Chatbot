import streamlit as st
import torch
import torch.nn as nn
import torch.serialization
import math
import re
import os
import json
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Empathetic Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS ==========
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0 !important;
    }
    
    .block-container {
        padding: 1rem 2rem !important;
        max-width: 100% !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Welcome Screen */
    .welcome-container {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        max-width: 600px;
        margin: 2rem auto;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    /* Chat Container */
    .chat-container {
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        height: calc(100vh - 4rem);
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    /* Chat Header */
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        color: white;
        border-radius: 20px 20px 0 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .chat-header-title {
        font-size: 1.5rem;
        font-weight: 600;
        flex-grow: 1;
    }
    
    .status-indicator {
        width: 10px;
        height: 10px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Messages Area */
    .messages-container {
        flex-grow: 1;
        overflow-y: auto;
        padding: 2rem;
        background: #f9fafb;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    
    /* Individual Messages */
    .message {
        display: flex;
        gap: 0.75rem;
        animation: slideIn 0.3s ease-out;
        max-width: 75%;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .message.user {
        align-self: flex-end;
        flex-direction: row-reverse;
    }
    
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    
    .user .message-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .bot .message-avatar {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .message-content {
        background: white;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        word-wrap: break-word;
    }
    
    .user .message-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .bot .message-content {
        background: white;
        color: #1f2937;
        border-bottom-left-radius: 4px;
    }
    
    .message-time {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        gap: 0.75rem;
        padding: 1rem;
        align-self: flex-start;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
        padding: 1rem;
        background: white;
        border-radius: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #9ca3af;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    /* Input Area */
    .input-container {
        padding: 1.5rem;
        background: white;
        border-top: 1px solid #e5e7eb;
        border-radius: 0 0 20px 20px;
    }
    
    /* Situation Badge */
    .situation-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #f3f4f6;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-size: 0.875rem;
        color: #4b5563;
        margin-bottom: 1rem;
    }
    
    .situation-badge-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Emotion Pills */
    .emotion-pill {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
        background: #f3f4f6;
        color: #4b5563;
    }
    
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background: white;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stat-label {
        font-size: 0.875rem;
        opacity: 0.9;
    }
    
    /* Scrollbar */
    .messages-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .messages-container::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    .messages-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
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
            raise ValueError(f"Expected a torchtext.vocab.vocab.Vocab object, but loaded a string: {vocab}")
        
        if not hasattr(vocab, '__getitem__'):
            raise ValueError(f"Loaded vocab object is not usable as a vocabulary: {type(vocab)}")
        
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
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False
    if "situation" not in st.session_state:
        st.session_state.situation = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Load model
    model, vocab, device = load_model_and_vocab()
    
    if model is None:
        st.error("Failed to load model. Check Streamlit Cloud logs for detailed errors.")
        st.info("""
If issues persist:
- Verify your Kaggle API key is valid and the dataset 'nadeemahmad003/chatbot-data' contains 'vocab.pth' and 'best-model-v4-stable.pt'.
- Ensure the dataset is public or accessible to your account.
- Try regenerating your Kaggle API token.
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Emotion selection
        emotions = ["afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive", 
                   "ashamed", "caring", "confident", "content", "devastated", "disappointed",
                   "disgusted", "embarrassed", "excited", "faithful", "furious", "grateful",
                   "guilty", "hopeful", "impressed", "jealous", "joyful", "lonely", "nostalgic",
                   "prepared", "proud", "sad", "sentimental", "surprised", "terrified", "trusting"]
        
        selected_emotion = st.selectbox("🎭 Emotion Context", ["auto"] + emotions, key="emotion_select")
        
        # Decoding strategy
        st.markdown("---")
        decoding_method = st.radio("🔍 Decoding Strategy", ["Greedy Search", "Beam Search"])
        
        if decoding_method == "Beam Search":
            beam_width = st.slider("Beam Width", 2, 5, 3)
        
        st.markdown("---")
        
        # Statistics
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-number">{len(st.session_state.messages) // 2}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">💬 Messages Exchanged</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f"**🖥️ Device:** {device}")
        st.markdown(f"**📚 Vocab Size:** {len(vocab):,}")
        
        st.markdown("---")
        
        # Clear conversation
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_started = False
            st.session_state.situation = ""
            st.rerun()
        
        # View History
        if st.button("📜 View History", use_container_width=True, type="secondary"):
            if st.session_state.chat_history:
                st.markdown("### 📜 Previous Conversations")
                for idx, hist in enumerate(reversed(st.session_state.chat_history[-5:])):
                    with st.expander(f"Session {len(st.session_state.chat_history) - idx}"):
                        st.markdown(f"**🕒 Time:** {hist['timestamp']}")
                        st.markdown(f"**💬 Messages:** {hist['message_count']}")
                        if hist['situation']:
                            st.markdown(f"**📝 Situation:** {hist['situation']}")
            else:
                st.info("No chat history yet!")
    
    # Welcome Screen
    if not st.session_state.chat_started:
        st.markdown("""
        <div class="welcome-container">
            <div style="font-size: 4rem; margin-bottom: 1rem;">💬</div>
            <div class="welcome-title">Empathetic AI Assistant</div>
            <div class="welcome-subtitle">I'm here to listen and help. Let's have a meaningful conversation.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Situation Input (Optional)
        col1, col2 = st.columns([3, 1])
        with col1:
            situation_input = st.text_input(
                "📝 Describe your situation (Optional)",
                placeholder="e.g., Having a rough day at work, celebrating a success...",
                key="situation_input"
            )
        
        # Quick situation templates
        st.markdown("**💡 Quick Templates:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("😊 General Chat", use_container_width=True):
                st.session_state.situation = "general conversation"
                st.session_state.chat_started = True
                st.rerun()
        with col2:
            if st.button("💼 Work Related", use_container_width=True):
                st.session_state.situation = "work related discussion"
                st.session_state.chat_started = True
                st.rerun()
        with col3:
            if st.button("🎉 Celebration", use_container_width=True):
                st.session_state.situation = "celebrating good news"
                st.session_state.chat_started = True
                st.rerun()
        with col4:
            if st.button("😔 Need Support", use_container_width=True):
                st.session_state.situation = "seeking emotional support"
                st.session_state.chat_started = True
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Start Chat Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Conversation", use_container_width=True, type="primary"):
                if situation_input:
                    st.session_state.situation = situation_input
                else:
                    st.session_state.situation = "general conversation"
                st.session_state.chat_started = True
                
                # Add welcome message
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "Hello! I'm your empathetic AI assistant. I'm here to listen and help. How can I support you today?",
                    "time": datetime.now().strftime("%I:%M %p")
                })
                st.rerun()
    
    # Chat Interface
    else:
        # Chat Container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Chat Header
        st.markdown(f"""
        <div class="chat-header">
            <div style="font-size: 2rem;">🤖</div>
            <div>
                <div class="chat-header-title">Empathetic Assistant</div>
                <div style="font-size: 0.875rem; opacity: 0.9;">
                    <span class="status-indicator"></span> Online • Ready to help
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Messages Container
        st.markdown('<div class="messages-container">', unsafe_allow_html=True)
        
        # Display situation badge if set
        if st.session_state.situation and st.session_state.situation != "general conversation":
            situation_class = "situation-badge situation-badge-active"
            st.markdown(f"""
            <div class="{situation_class}">
                📝 Context: {st.session_state.situation.title()}
            </div>
            """, unsafe_allow_html=True)
        
        # Display messages
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            time = message.get("time", "")
            
            if role == "user":
                st.markdown(f"""
                <div class="message user">
                    <div class="message-avatar">👤</div>
                    <div>
                        <div class="message-content">{content}</div>
                        <div class="message-time">{time}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message bot">
                    <div class="message-avatar">🤖</div>
                    <div>
                        <div class="message-content">{content}</div>
                        <div class="message-time">{time}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input Container
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Update situation option
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.situation and st.session_state.situation != "general conversation":
                st.markdown(f"<small>📝 Current context: <b>{st.session_state.situation}</b></small>", unsafe_allow_html=True)
        with col2:
            if st.button("✏️ Change", use_container_width=True, key="change_situation"):
                st.session_state.chat_started = False
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Message Input Area (outside chat container for better positioning)
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your message",
                placeholder="Type your message here...",
                key="message_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("📤 Send", use_container_width=True, type="primary")
        
        # Process message
        if send_button and user_input:
            # Add user message
            current_time = datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "time": current_time
            })
            
            # Normalize input
            normalized_input = normalize_text(user_input)
            
            # Determine emotion
            if selected_emotion == "auto":
                emotion = "neutral"
            else:
                emotion = selected_emotion
            
            # Prepare situation
            normalized_situation = normalize_text(st.session_state.situation)
            
            # Create input text
            input_text = f"Emotion: {emotion} | Situation: {normalized_situation} | Customer: {normalized_input} Agent:"
            
            # Generate response
            with st.spinner(""):
                if decoding_method == "Greedy Search":
                    response = greedy_decode(model, vocab, input_text, device)
                else:
                    response = beam_search_decode(model, vocab, input_text, device, beam_width=beam_width)
            
            # Add bot response
            st.session_state.messages.append({
                "role": "bot",
                "content": response,
                "time": datetime.now().strftime("%I:%M %p")
            })
            
            # Update history
            if len(st.session_state.messages) % 10 == 0:
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
                    "message_count": len(st.session_state.messages),
                    "situation": st.session_state.situation
                })
            
            st.rerun()

if __name__ == "__main__":
    main()
