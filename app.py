# app.py - Final Corrected Streamlit Chatbot Application

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import re
import os
import requests

# --- 1. DEFINE THE ENTIRE MODEL ARCHITECTURE ---
# (This section is unchanged, it defines the model classes)

# --- Model Hyperparameters (Must match the trained model) ---
VOCAB_SIZE = 16812 # Placeholder, will be updated
EMBED_DIM = 512
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1
FF_DIM = 4 * EMBED_DIM
DEVICE = torch.device('cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# --- Paste all your model classes here ---
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
        self.emb_dim, self.num_heads, self.head_dim = emb_dim, num_heads, emb_dim // num_heads
        self.fc_q, self.fc_k, self.fc_v, self.fc_o = nn.Linear(emb_dim, emb_dim), nn.Linear(emb_dim, emb_dim), nn.Linear(emb_dim, emb_dim), nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(DEVICE)
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q, K, V = self.fc_q(query), self.fc_k(key), self.fc_v(value)
        Q, K, V = [x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) for x in (Q, K, V)]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None: energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.emb_dim)
        return self.fc_o(x), attention

class PositionwiseFeedforward(nn.Module):
    def __init__(self, emb_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.fc_1, self.fc_2 = nn.Linear(emb_dim, ff_dim), nn.Linear(ff_dim, emb_dim)
        self.dropout, self.relu = nn.Dropout(dropout), nn.ReLU()
    def forward(self, x): return self.fc_2(self.dropout(self.relu(self.fc_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(emb_dim, ff_dim, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(emb_dim), nn.LayerNorm(emb_dim)
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
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(emb_dim), nn.LayerNorm(emb_dim), nn.LayerNorm(emb_dim)
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
        for layer in self.layers: src = layer(src, src_mask)
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
        for layer in self.layers: trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        return self.fc_out(trg), attention

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.device = device
    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones((trg.shape[1], trg.shape[1]), device=self.device)).bool()
        return trg_pad_mask & trg_sub_mask
    def forward(self, src, trg):
        src_mask, trg_mask = self.make_src_mask(src), self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        return self.decoder(trg, enc_src, trg_mask, src_mask)

# --- 2. FILE HANDLING AND MODEL LOADING ---

### CRITICAL CHANGE ###
# Using a static, hardcoded list of the 32 emotions from the dataset.
# This is robust and prevents non-emotion text from appearing in the list.
EMOTION_LIST = [
    'afraid', 'angry', 'annoyed', 'anticipating', 'anxious', 'apprehensive',
    'ashamed', 'caring', 'confident', 'content', 'devastated', 'disappointed',
    'disgusted', 'embarrassed', 'excited', 'faithful', 'furious', 'grateful',
    'guilty', 'hopeful', 'impressed', 'jealous', 'joyful', 'lonely', 'nostalgic',
    'prepared', 'proud', 'sad', 'sentimental', 'surprised', 'terrified', 'trusting'
]
@st.cache_resource
def load_model_and_vocab():
    """
    Downloads model files from a public URL if they don't exist locally,
    then loads the model and vocabulary.
    """
    MODEL_URL = "https://huggingface.co/Nadeemoo3/Chatbot/resolve/main/best-model-v4-stable.pt"
    VOCAB_URL = "https://huggingface.co/Nadeemoo3/Chatbot/resolve/main/vocab.pth"

    MODEL_PATH = "best-model-v4-stable.pt"
    VOCAB_PATH = "vocab.pth"

    def download_file(url, filename):
        if not os.path.exists(filename):
            st.info(f"Downloading required file: {filename}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                
                with open(filename, 'wb') as f:
                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = int((downloaded_size / total_size) * 100) if total_size > 0 else 0
                        progress_bar.progress(progress)
                progress_bar.empty()
                
                # --- START OF TEMPORARY DEBUG CODE ---
                with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                    content_start = f.read(100)
                    st.warning(f"DEBUG: Start of downloaded file '{filename}':")
                    st.code(content_start)
                # --- END OF TEMPORARY DEBUG CODE ---

            except requests.exceptions.RequestException as e:
                st.error(f"Error downloading {filename}: {e}")
                return False
        return True

    if not download_file(MODEL_URL, MODEL_PATH) or not download_file(VOCAB_URL, VOCAB_PATH):
        st.error("Could not download necessary model files. The app cannot continue.")
        st.stop()

    try:
        vocab = torch.load(VOCAB_PATH, weights_only=False)
        global VOCAB_SIZE
        VOCAB_SIZE = len(vocab)
        
        encoder = Encoder(VOCAB_SIZE, EMBED_DIM, NUM_ENCODER_LAYERS, NUM_HEADS, FF_DIM, DROPOUT, DEVICE)
        decoder = Decoder(VOCAB_SIZE, EMBED_DIM, NUM_DECODER_LAYERS, NUM_HEADS, FF_DIM, DROPOUT, DEVICE)
        model = Seq2SeqTransformer(encoder, decoder, PAD_IDX, PAD_IDX, DEVICE).to(DEVICE)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
        model.eval()
        
        return model, vocab
    except Exception as e:
        st.error(f"Failed to load the model or vocabulary. Error: {e}")
        return None, None


model, vocab = load_model_and_vocab()
if model and vocab:
    BOS_TOKEN, EOS_TOKEN = vocab.lookup_tokens([BOS_IDX, EOS_IDX])
else:
    st.stop()

# --- 3. INFERENCE AND TEXT PROCESSING FUNCTIONS ---

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text

def generate_response(model, src_sentence, decoding_strategy="Greedy Search", beam_width=3, max_len=50):
    model.eval()
    normalized_sentence = normalize_text(src_sentence)
    tokens = [token for token in normalized_sentence.split() if token in vocab]
    
    src_indexes = [BOS_IDX] + [vocab[token] for token in tokens] + [EOS_IDX]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(DEVICE)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    if decoding_strategy == "Greedy Search":
        trg_indexes = [BOS_IDX]
        for _ in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(DEVICE)
            trg_mask = model.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            pred_token_idx = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token_idx)
            if pred_token_idx == EOS_IDX:
                break
        final_indices = trg_indexes
    else: # Beam Search
        beams = [([BOS_IDX], 0.0)]
        completed_beams = []
        for _ in range(max_len):
            new_beams = []
            all_done = True
            for seq, score in beams:
                if seq[-1] == EOS_IDX:
                    completed_beams.append((seq, score))
                    continue
                all_done = False
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
            if all_done:
                break
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        completed_beams.extend(beams)
        best_beam = sorted(completed_beams, key=lambda x: x[1] / len(x[0] if len(x[0]) > 0 else 1), reverse=True)[0]
        final_indices = best_beam[0]

    trg_tokens = vocab.lookup_tokens(final_indices)
    return " ".join(trg_tokens)

# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="Empathetic Chatbot", layout="wide")

st.title("💬 Empathetic Conversational Chatbot")
st.caption("A Transformer-based chatbot built from scratch to provide empathetic replies.")

st.sidebar.header("Chat Configuration")
selected_emotion = st.sidebar.selectbox("1. Choose the emotion you are feeling:", EMOTION_LIST, index=22) # Default to 'joyful'
decoding_strategy = st.sidebar.radio("2. Select a Decoding Strategy:", ("Greedy Search", "Beam Search (k=3)"))
situation_text = st.sidebar.text_area(
    "3. Briefly describe the situation (optional but recommended):",
    "I just got a promotion at work!",
    height=100
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please tell me what's on your mind."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Share your thoughts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            
            ### CRITICAL CHANGE ###
            # The input format MUST EXACTLY MATCH the format used during training.
            # The model expects a special token like <emotion_joyful>, not the text "Emotion: joyful".
            emotion_token = f"<emotion_{selected_emotion}>"
            input_text = f"{emotion_token} | Situation: {situation_text} | Customer: {prompt} Agent:"
            
            response = generate_response(model, input_text, decoding_strategy)
            
            # Clean the output for better display
            clean_response = response.replace(BOS_TOKEN, "").strip()
            if EOS_TOKEN in clean_response:
                clean_response = clean_response.split(EOS_TOKEN)[0].strip()

        # Simulate typing effect
        full_response = ""
        for chunk in clean_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
