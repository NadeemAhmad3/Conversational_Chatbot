# app.py - Empathetic Conversational Chatbot

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import requests
from io import BytesIO

# --- MODEL HYPERPARAMETERS (Must match trained model) ---
EMBED_DIM = 512
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1
FF_DIM = 4 * EMBED_DIM
DEVICE = torch.device('cpu')

# --- SPECIAL TOKEN INDICES ---
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# --- MODEL ARCHITECTURE CLASSES ---

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(max_len, 1, emb_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float):
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
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(DEVICE)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.emb_dim)
        x = self.fc_o(x)
        return x, attention

class PositionwiseFeedforward(nn.Module):
    def __init__(self, emb_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.fc_1 = nn.Linear(emb_dim, ff_dim)
        self.fc_2 = nn.Linear(ff_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc_2(self.dropout(self.relu(self.fc_1(x))))

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
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, num_heads, ff_dim, dropout) 
                                     for _ in range(num_layers)])
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
        self.layers = nn.ModuleList([DecoderLayer(emb_dim, num_heads, ff_dim, dropout) 
                                     for _ in range(num_layers)])
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        tok_embedded = self.tok_embedding(trg) * self.scale
        pos_embedded = self.pos_embedding(tok_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        trg = self.dropout(pos_embedded)
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        return self.fc_out(trg), attention

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones((trg.shape[1], trg.shape[1]), device=self.device)).bool()
        return trg_pad_mask & trg_sub_mask
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        return self.decoder(trg, enc_src, trg_mask, src_mask)

# --- LOAD MODEL AND VOCABULARY FROM HUGGINGFACE ---

@st.cache_resource
def load_model_and_vocab():
    """Loads model and vocabulary from HuggingFace."""
    
    with st.spinner("Loading model and vocabulary from HuggingFace..."):
        # HuggingFace direct download URLs
        vocab_url = "https://huggingface.co/Nadeemoo3/Chatbot/resolve/main/vocab.pth"
        model_url = "https://huggingface.co/Nadeemoo3/Chatbot/resolve/main/best-model-v4-stable.pt"
        
        try:
            # Download vocabulary
            vocab_response = requests.get(vocab_url)
            vocab_response.raise_for_status()
            
            # Load vocab with weights_only=False (vocab contains custom torchtext class)
            # This is safe because we trust the source (your own HuggingFace repo)
            vocab = torch.load(
                BytesIO(vocab_response.content), 
                map_location=DEVICE,
                weights_only=False
            )
            
            # Get vocab size
            VOCAB_SIZE = len(vocab)
            
            # Initialize model architecture
            encoder = Encoder(VOCAB_SIZE, EMBED_DIM, NUM_ENCODER_LAYERS, NUM_HEADS, FF_DIM, DROPOUT, DEVICE)
            decoder = Decoder(VOCAB_SIZE, EMBED_DIM, NUM_DECODER_LAYERS, NUM_HEADS, FF_DIM, DROPOUT, DEVICE)
            model = Seq2SeqTransformer(encoder, decoder, PAD_IDX, PAD_IDX, DEVICE).to(DEVICE)
            
            # Download and load model weights
            model_response = requests.get(model_url)
            model_response.raise_for_status()
            
            # Load model state dict with weights_only=True (safer for model weights)
            state_dict = torch.load(
                BytesIO(model_response.content), 
                map_location=DEVICE,
                weights_only=True
            )
            model.load_state_dict(state_dict)
            model.eval()
            
            return model, vocab
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error("Please ensure the model files are accessible on HuggingFace.")
            st.stop()

model, vocab = load_model_and_vocab()
BOS_TOKEN = vocab.lookup_token(BOS_IDX)
EOS_TOKEN = vocab.lookup_token(EOS_IDX)

# Extract emotion list from vocabulary
EMOTION_LIST = sorted([emo.replace('<emotion_', '').replace('>', '') 
                       for emo in vocab.get_itos() if emo.startswith('<emotion_')])

# --- INFERENCE FUNCTIONS ---

def simple_tokenizer(s):
    return s.split()

def greedy_decode(model, src_sentence, max_len=50):
    model.eval()
    tokens = simple_tokenizer(src_sentence)
    src_indexes = [BOS_IDX] + [vocab[token] for token in tokens] + [EOS_IDX]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(DEVICE)
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    trg_indexes = [BOS_IDX]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(DEVICE)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token_idx = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token_idx)
        
        if pred_token_idx == EOS_IDX:
            break
    
    trg_tokens = vocab.lookup_tokens(trg_indexes)
    return " ".join(trg_tokens)

def beam_search_decode(model, src_sentence, beam_width=3, max_len=50):
    model.eval()
    tokens = simple_tokenizer(src_sentence)
    src_indexes = [BOS_IDX] + [vocab[token] for token in tokens] + [EOS_IDX]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(DEVICE)
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
            
            trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(DEVICE)
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
    trg_tokens = vocab.lookup_tokens(best_beam[0])
    return " ".join(trg_tokens)

# --- STREAMLIT UI ---

st.set_page_config(page_title="Empathetic Chatbot", page_icon="💬", layout="wide")

st.title("💬 Empathetic Conversational Chatbot")
st.caption("A Transformer-based chatbot built from scratch using Multi-Head Attention")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

selected_emotion = st.sidebar.selectbox(
    "🎭 Select Emotion:",
    EMOTION_LIST,
    index=0
)

situation_input = st.sidebar.text_area(
    "📝 Describe the Situation:",
    placeholder="Example: I remember going to the fireworks with my best friend...",
    height=100
)

decoding_strategy = st.sidebar.radio(
    "🔍 Decoding Strategy:",
    ("Greedy Search", "Beam Search (k=3)")
)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
    st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Validate situation input
    if not situation_input or situation_input.strip() == "":
        st.warning("⚠️ Please provide a situation in the sidebar before sending a message.")
        st.stop()
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            # Construct model input (as per requirements)
            emotion_token = f"<emotion_{selected_emotion}>"
            input_text = f"Emotion: {selected_emotion} | Situation: {situation_input.strip()} | Customer: {prompt} Agent:"
            
            # Decode
            if decoding_strategy == "Greedy Search":
                response = greedy_decode(model, input_text)
            else:
                response = beam_search_decode(model, input_text, beam_width=3)
            
            # Clean response
            clean_response = response.replace(BOS_TOKEN, "").strip()
            if EOS_TOKEN in clean_response:
                clean_response = clean_response.split(EOS_TOKEN)[0].strip()
            
            # Typing effect
            full_response = ""
            for chunk in clean_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Built with ❤️ using PyTorch & Streamlit**")
st.sidebar.markdown("Model: Custom Transformer (512d, 4 heads, 3 layers)")
