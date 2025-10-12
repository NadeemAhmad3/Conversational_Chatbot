import streamlit as st
import torch
import torch.nn as nn
import torch.serialization
import math
import re
import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import getpass  # Not needed, but for completeness

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Empathetic Chatbot",
    page_icon="💬",
    layout="wide"
)

# ========== Custom CSS for Better UI ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f0f0f0;
        text-align: left;
    }
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== Model Architecture Components ==========
# (Keep all your existing classes unchanged: PositionalEncoding, MultiHeadAttention, PositionwiseFeedforward, EncoderLayer, DecoderLayer, Encoder, Decoder, Seq2SeqTransformer)
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
# (Keep unchanged: normalize_text, simple_tokenizer)

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
    """Download model and vocab files from Kaggle dataset with explicit auth setup"""
    try:
        # Step 1: Load secrets and set environment variables (environment method)
        if "KAGGLE_USERNAME" not in st.secrets or "KAGGLE_KEY" not in st.secrets:
            raise ValueError("Kaggle secrets not found. Check Streamlit Cloud secrets.")
        
        username = st.secrets["KAGGLE_USERNAME"]
        key = st.secrets["KAGGLE_KEY"]
        
        # Debug: Confirm secrets loaded (visible in logs)
        print(f"Loaded username: {username}")  # Use print for logs; st.write may not show in background
        
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        
        # Step 2: Create kaggle.json file in the expected location (file method as fallback)
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
        
        kaggle_json = {"username": username, "key": key}
        with open(kaggle_json_path, "w") as f:
            json.dump(kaggle_json, f)
        
        # Set permissions (secure the file)
        os.chmod(kaggle_json_path, 0o600)
        
        print(f"Created kaggle.json at {kaggle_json_path}")  # Debug log
        
        # Step 3: Authenticate and download
        api = KaggleApi()
        api.authenticate()  # Now it should find kaggle.json
        
        dataset_name = "nadeemahmad003/chatbot-data"
        if not os.path.exists("model_files"):
            os.makedirs("model_files")
        
        api.dataset_download_files(dataset_name, path="model_files", unzip=True)
        
        # Step 4: Debug - List downloaded files
        downloaded_files = os.listdir("model_files")
        print(f"Downloaded files: {downloaded_files}")  # Check if vocab.pth etc. are there
        
        if "vocab.pth" not in downloaded_files:
            raise FileNotFoundError("vocab.pth not found after download. Check dataset contents.")
        
        return True
    except Exception as e:
        print(f"Download error details: {str(e)}")  # Detailed log
        st.error(f"Error downloading model files: {str(e)}")
        return False

@st.cache_resource
def load_model_and_vocab():
    """Load the trained model and vocabulary"""
    try:
        # Download if needed
        if not os.path.exists("model_files/vocab.pth"):
            with st.spinner("Downloading model files from Kaggle..."):
                if not download_model_files():
                    raise RuntimeError("Download failed.")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary with weights_only=False since it's a Vocab object
        vocab = torch.load('model_files/vocab.pth', map_location=device, weights_only=False)
        
        # Debug: Log the type of vocab
        print(f"Type of loaded vocab: {type(vocab)}")  # Will appear in Streamlit Cloud logs
        
        # Check if vocab is a string (unexpected)
        if isinstance(vocab, str):
            raise ValueError(f"Expected a torchtext.vocab.vocab.Vocab object, but loaded a string: {vocab}")
        
        # Verify vocab is usable
        if not hasattr(vocab, '__getitem__'):
            raise ValueError(f"Loaded vocab object is not usable as a vocabulary: {type(vocab)}")
        
        # Model hyperparameters (must match training)
        VOCAB_SIZE = len(vocab)
        EMBED_DIM = 512
        NUM_HEADS = 4
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3
        DROPOUT = 0.1
        FF_DIM = 4 * EMBED_DIM
        PAD_IDX = 1
        
        # Initialize model
        encoder = Encoder(VOCAB_SIZE, EMBED_DIM, NUM_ENCODER_LAYERS, NUM_HEADS, FF_DIM, DROPOUT, device)
        decoder = Decoder(VOCAB_SIZE, EMBED_DIM, NUM_DECODER_LAYERS, NUM_HEADS, FF_DIM, DROPOUT, device)
        model = Seq2SeqTransformer(encoder, decoder, PAD_IDX, PAD_IDX, device).to(device)
        
        # Load trained weights with weights_only=True (since it's a state_dict)
        model.load_state_dict(torch.load('model_files/best-model-v4-stable.pt', map_location=device, weights_only=True))
        model.eval()
        
        return model, vocab, device
    except Exception as e:
        print(f"Error loading model details: {str(e)}")  # Detailed log
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# (Keep your existing decoding functions unchanged: greedy_decode, beam_search_decode)

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
    
    # Clean response
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
    
    # Clean response
    response = response.replace("<bos>", "").replace("<eos>", "").strip()
    
    return response

# ========== Main Application ==========
# (Keep the main() function unchanged, but update the error message to remove redundant instructions since secrets are set)

def main():
    # Header
    st.markdown('<div class="main-header">💬 Empathetic Conversational Chatbot</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, vocab, device = load_model_and_vocab()
    
    if model is None:
        st.error("Failed to load model. Check Streamlit Cloud logs for detailed errors (e.g., authentication or dataset issues).")
        st.info("""
If issues persist:
- Verify your Kaggle API key is valid and the dataset 'nadeemahmad003/chatbot-data' contains 'vocab.pth' and 'best-model-v4-stable.pt'.
- Ensure the dataset is public or accessible to your account.
- Try regenerating your Kaggle API token.
        """)
        return
    
    # Sidebar (unchanged)
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Emotion selection
        emotions = ["afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive", 
                   "ashamed", "caring", "confident", "content", "devastated", "disappointed",
                   "disgusted", "embarrassed", "excited", "faithful", "furious", "grateful",
                   "guilty", "hopeful", "impressed", "jealous", "joyful", "lonely", "nostalgic",
                   "prepared", "proud", "sad", "sentimental", "surprised", "terrified", "trusting"]
        
        selected_emotion = st.selectbox("Select Emotion (Optional)", ["auto"] + emotions)
        
        # Decoding strategy
        decoding_method = st.radio("Decoding Strategy", ["Greedy Search", "Beam Search"])
        
        if decoding_method == "Beam Search":
            beam_width = st.slider("Beam Width", 2, 5, 3)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("**About this Chatbot:**")
        st.markdown("This is a Transformer-based empathetic chatbot trained from scratch.")
        st.markdown(f"- Device: {device}")
        st.markdown(f"- Vocab Size: {len(vocab)}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">👤 You: {message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">🤖 Agent: {message["content"]}</div>', 
                       unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        situation = st.text_input("Situation (Optional):", placeholder="Describe the situation...")
    
    with col2:
        st.write("")  # Spacing
    
    user_input = st.text_input("Your Message:", placeholder="Type your message here...", key="user_input")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        send_button = st.button("Send 📤", use_container_width=True)
    
    if send_button and user_input:
        # Normalize input
        normalized_input = normalize_text(user_input)
        
        # Determine emotion
        if selected_emotion == "auto":
            emotion = "neutral"
        else:
            emotion = selected_emotion
        
        # Prepare situation
        if situation.strip():
            normalized_situation = normalize_text(situation)
        else:
            normalized_situation = "general conversation"
        
        # Create input text
        input_text = f"Emotion: {emotion} | Situation: {normalized_situation} | Customer: {normalized_input} Agent:"
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate response
        with st.spinner("Generating response..."):
            if decoding_method == "Greedy Search":
                response = greedy_decode(model, vocab, input_text, device)
            else:
                response = beam_search_decode(model, vocab, input_text, device, beam_width=beam_width)
        
        # Add bot response to history
        st.session_state.messages.append({"role": "bot", "content": response})
        
        # Rerun to update chat
        st.rerun()

if __name__ == "__main__":
    main()
