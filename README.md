# Empathetic Conversational Chatbot with Transformer Architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Transformer-based encoder-decoder chatbot built from scratch that generates empathetic responses given emotional context, situation, and user utterances. This project implements multi-head attention mechanisms without using any pretrained model weights.

## 🎯 Project Overview

This chatbot is designed to generate contextually appropriate and empathetic responses by understanding:
- **Emotion**: The emotional state of the conversation (e.g., sentimental, afraid, excited)
- **Situation**: Background context provided by the user
- **Customer Utterance**: The actual message from the user

### Example Interactions

**Example 1:**
- **Input**: `Emotion: sentimental | Situation: I remember going to the fireworks with my best friend... | Customer: This was a best friend. I miss her. Agent:`
- **Output**: `Where has she gone?`

**Example 2:**
- **Input**: `Emotion: afraid | Situation: I used to scare for darkness | Customer: it feels like hitting to blank wall when I see the darkness Agent:`
- **Output**: `Oh ya? I don't really see how`

## 🚀 Live Demo

🔗 **Deployed Application**: [Your Streamlit/Gradio Link Here]

Try out the chatbot with different emotions and situations to see empathetic responses generated in real-time!

## 📊 Dataset

This project uses the **Empathetic Dialogues** dataset from Facebook AI Research:
- **Source**: [Kaggle - Empathetic Dialogues](https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai)
- **Size**: ~25k conversations
- **Split**: 80% Train / 10% Validation / 10% Test

## 🏗️ Architecture

### Transformer Components

Our implementation includes all core Transformer components built from scratch:

1. **Multi-Head Attention**
   - Self-attention mechanism in encoder
   - Masked self-attention in decoder
   - Cross-attention between encoder and decoder

2. **Positional Encoding**
   - Sinusoidal position embeddings
   - Enables model to understand token positions

3. **Feed-Forward Networks**
   - Position-wise fully connected layers
   - Two linear transformations with ReLU activation

4. **Layer Normalization & Residual Connections**
   - Stabilizes training
   - Enables gradient flow through deep networks

### Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 512 |
| Number of Attention Heads | 8 |
| Encoder Layers | 6 |
| Decoder Layers | 6 |
| Feed-Forward Dimension | 2048 |
| Dropout Rate | 0.1 |
| Max Sequence Length | 128 |
| Vocabulary Size | 10000 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Optimizer | Adam (β₁=0.9, β₂=0.98) |
| Learning Rate | 5e-4 |
| Learning Rate Scheduler | Warmup + Decay |
| Training Strategy | Teacher Forcing |
| Loss Function | Cross Entropy |
| Number of Epochs | 50 |

## 📁 Project Structure

```
empathetic-chatbot/
│
├── data/
│   ├── raw/                      # Original dataset files
│   ├── processed/                # Preprocessed data
│   └── vocab.pkl                 # Vocabulary object
│
├── models/
│   ├── transformer.py            # Main Transformer architecture
│   ├── encoder.py                # Encoder implementation
│   ├── decoder.py                # Decoder implementation
│   ├── attention.py              # Multi-head attention
│   ├── positional_encoding.py   # Positional encoding
│   └── utils.py                  # Helper functions
│
├── training/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── metrics.py                # BLEU, ROUGE, chrF, Perplexity
│   └── dataset.py                # Custom PyTorch Dataset
│
├── inference/
│   ├── predict.py                # Inference logic
│   ├── beam_search.py            # Beam search decoder
│   └── greedy_search.py          # Greedy decoder
│
├── app/
│   ├── streamlit_app.py          # Streamlit UI
│   └── gradio_app.py             # Gradio UI (alternative)
│
├── notebooks/
│   ├── data_exploration.ipynb    # EDA
│   ├── model_analysis.ipynb      # Model analysis
│   └── attention_visualization.ipynb
│
├── checkpoints/                  # Saved model weights
├── logs/                         # Training logs
├── evaluation/
│   └── results.json              # Evaluation metrics
│
├── requirements.txt              # Python dependencies
├── config.yaml                   # Configuration file
├── README.md                     # This file
└── setup.py                      # Package setup
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/empathetic-chatbot.git
cd empathetic-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
```bash
# Download from Kaggle
kaggle datasets download -d atharvjairath/empathetic-dialogues-facebook-ai
unzip empathetic-dialogues-facebook-ai.zip -d data/raw/
```

## 📝 Usage

### 1. Data Preprocessing

```bash
python training/preprocess.py --input data/raw/ --output data/processed/
```

This will:
- Normalize text (lowercase, clean whitespace, punctuation)
- Build vocabulary from training split
- Add special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<sep>`
- Split dataset: 80% train, 10% validation, 10% test

### 2. Training

```bash
python training/train.py --config config.yaml
```

**Training options:**
```bash
python training/train.py \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --num_epochs 50 \
  --embed_dim 512 \
  --num_heads 8 \
  --num_encoder_layers 6 \
  --num_decoder_layers 6 \
  --dropout 0.1 \
  --save_dir checkpoints/
```

**Monitor training:**
```bash
tensorboard --logdir logs/
```

### 3. Evaluation

```bash
python training/evaluate.py --checkpoint checkpoints/best_model.pt --test_data data/processed/test.pkl
```

### 4. Inference

**Command-line inference:**
```bash
python inference/predict.py \
  --checkpoint checkpoints/best_model.pt \
  --emotion "sentimental" \
  --situation "I remember my childhood days" \
  --utterance "I miss those carefree moments" \
  --decode_strategy "beam" \
  --beam_width 5
```

**Interactive mode:**
```bash
python inference/predict.py --checkpoint checkpoints/best_model.pt --interactive
```

### 5. Web Application

**Streamlit:**
```bash
streamlit run app/streamlit_app.py
```

**Gradio:**
```bash
python app/gradio_app.py
```

## 📈 Evaluation Metrics

### Automatic Metrics (Test Set)

| Metric | Score |
|--------|-------|
| **BLEU-1** | 0.xx |
| **BLEU-4** | 0.xx |
| **ROUGE-L** | 0.xx |
| **chrF** | 0.xx |
| **Perplexity** | xx.xx |

### Human Evaluation

Evaluated on 100 random test samples:

| Criterion | Average Score (1-5) |
|-----------|---------------------|
| **Fluency** | x.xx |
| **Relevance** | x.xx |
| **Adequacy** | x.xx |
| **Empathy** | x.xx |

### Qualitative Examples

**Example 1:**
- **Input**: `Emotion: proud | Situation: My daughter graduated college | Customer: She worked so hard for this Agent:`
- **Ground Truth**: `You must be so proud of her achievement!`
- **Model Output**: `That's wonderful! What is she planning to do next?`

**Example 2:**
- **Input**: `Emotion: anxious | Situation: Waiting for exam results | Customer: I can't stop thinking about it Agent:`
- **Ground Truth**: `I understand. The waiting is always the hardest part.`
- **Model Output**: `Try to relax. I'm sure you did your best!`

## 🎨 Features

### Core Features
- ✅ Transformer encoder-decoder from scratch
- ✅ Multi-head attention mechanism
- ✅ Positional encoding
- ✅ Teacher forcing during training
- ✅ Label smoothing for better generalization
- ✅ Multiple decoding strategies (Greedy & Beam Search)

### UI Features
- 🎯 Real-time response generation
- 📊 Attention visualization heatmap
- 💬 Conversation history tracking
- 🎭 Emotion selection dropdown
- ⚙️ Adjustable decoding parameters
- 📝 Copy/export conversation

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
model:
  embed_dim: 512
  num_heads: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
  ff_dim: 2048
  dropout: 0.1
  max_seq_length: 128

training:
  batch_size: 64
  learning_rate: 5e-4
  num_epochs: 50
  warmup_steps: 4000
  gradient_clip: 1.0
  label_smoothing: 0.1

data:
  vocab_size: 10000
  min_freq: 2
  max_length: 128
```

## 🐛 Debugging & Sanity Checks

### Overfitting on Small Subset
To verify model implementation:
```bash
python training/train.py --debug --num_samples 10 --num_epochs 100
```
Model should achieve near-zero loss on tiny dataset.

### Causal Mask Verification
Ensure decoder cannot attend to future tokens:
```bash
python tests/test_causal_mask.py
```

### Gradient Flow Check
```bash
python tests/test_gradients.py
```

## 📊 Training Tips

1. **Start Small**: Test on 1000 samples first
2. **Learning Rate**: Use warmup (4000 steps recommended)
3. **Regularization**: Dropout 0.1-0.3, Label smoothing 0.1
4. **Checkpointing**: Save every epoch, keep best 3 models
5. **Early Stopping**: Stop if validation BLEU doesn't improve for 5 epochs
6. **Gradient Clipping**: Clip at 1.0 to prevent exploding gradients

## 🚨 Common Issues

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or sequence length
```bash
python training/train.py --batch_size 32 --max_seq_length 64
```

### Issue: Poor BLEU Scores
**Solutions**:
- Train longer (50+ epochs)
- Increase model capacity (more layers/heads)
- Tune learning rate
- Check data preprocessing

### Issue: Model generates repetitive responses
**Solutions**:
- Use beam search with diversity penalty
- Increase temperature during sampling
- Add repetition penalty

## 📚 Technical Details

### Input Format
```
Emotion: {emotion} | Situation: {situation} | Customer: {utterance} Agent:
```

### Special Tokens
- `<pad>`: Padding token (id: 0)
- `<unk>`: Unknown token (id: 1)
- `<bos>`: Beginning of sequence (id: 2)
- `<eos>`: End of sequence (id: 3)
- `<sep>`: Separator token (id: 4)

### Attention Masks
- **Encoder**: Padding mask only
- **Decoder**: Causal mask + padding mask
- **Cross-attention**: Encoder padding mask

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- **[Your Name]** - [GitHub](https://github.com/yourusername) - [LinkedIn](https://linkedin.com/in/yourprofile)
- **[Partner Name]** - [GitHub](https://github.com/partnername) - [LinkedIn](https://linkedin.com/in/partnerprofile)

## 🙏 Acknowledgments

- **Dataset**: [Empathetic Dialogues by Facebook AI](https://github.com/facebookresearch/EmpatheticDialogues)
- **Paper**: "Towards Empathetic Open-domain Conversation Models" by Rashkin et al.
- **Inspiration**: "Attention Is All You Need" by Vaswani et al.
- **Course**: [Your Course Name] - [University Name]

## 📖 Blog Post

Read our detailed blog post about this project: [Medium Link]

Topics covered:
- Dataset exploration and preprocessing
- Transformer architecture deep dive
- Training challenges and solutions
- Evaluation methodology
- Future improvements

## 📞 Contact

For questions or feedback:
- **Email**: your.email@example.com
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/empathetic-chatbot/issues)

## 🔮 Future Work

- [ ] Add emotion classification component
- [ ] Implement persona-based responses
- [ ] Multi-turn conversation handling
- [ ] Fine-tune on domain-specific data
- [ ] Add multilingual support
- [ ] Implement reinforcement learning from human feedback
- [ ] Knowledge-grounded responses

## 📊 Training Logs

| Epoch | Train Loss | Val Loss | BLEU-4 | ROUGE-L | Perplexity |
|-------|-----------|----------|--------|---------|------------|
| 1     | x.xxx     | x.xxx    | 0.xxx  | 0.xxx   | xx.xx      |
| 5     | x.xxx     | x.xxx    | 0.xxx  | 0.xxx   | xx.xx      |
| 10    | x.xxx     | x.xxx    | 0.xxx  | 0.xxx   | xx.xx      |
| ...   | ...       | ...      | ...    | ...     | ...        |
| 50    | x.xxx     | x.xxx    | 0.xxx  | 0.xxx   | xx.xx      |

---

⭐ **Star this repository if you find it helpful!**

📝 **Citation**:
```bibtex
@misc{empathetic-chatbot-2025,
  author = {Your Name},
  title = {Empathetic Conversational Chatbot with Transformer Architecture},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/empathetic-chatbot}
}
```
