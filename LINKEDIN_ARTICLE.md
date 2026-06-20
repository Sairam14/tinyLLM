# Building a Production Transformer from Scratch: From Attention to Real Training

## How I Built a Complete Language Model in ~800 Lines of Python (And What I Learned About German Tokenization)

---

When you read about transformer models—GPT, LLaMA, Claude—the architecture sounds intimidating: multi-head attention, residual connections, feed-forward networks, Pre-LayerNorm. But I've realized something: **the fundamentals are simpler than the terminology suggests.**

Over the past month, I've built a complete transformer LLM from scratch and trained it on real German literature (Kafka, Goethe, Schiller). I want to walk you through the journey—from understanding how attention works, to assembling a full transformer block, to watching the model actually learn on real text.

And I discovered something unexpected: the choice of tokenization algorithm matters far less than I thought, but the insights from that comparison unlock something important about why large language models are structured the way they are.

---

## Part 2: Multi-Head Attention—The Heart of It All

If you want to understand modern AI, you need to understand attention. Not the simplified "query-key-value" explanation, but the *actual mechanism* that makes it work.

Here's the core idea:

**Attention lets each token learn which other tokens are relevant to it.**

When I'm processing the sentence "The bank executive was arrested," the word "bank" needs to figure out: am I talking about a river bank or a financial institution? I can only answer that by attending to the word "executive." That's attention.

Mathematically:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

This looks like magic notation, but here's what it actually does:

1. **Q (Queries):** "What am I looking for?" (one per token)
2. **K (Keys):** "What can I help with?" (one per token)
3. **Softmax(QK^T):** Compute similarity scores—which keys match which queries?
4. **V (Values):** "Here's my actual information"

The `/ √d_k` is a scaling factor that prevents the dot products from exploding into numerical instability. That's it. The entire attention mechanism is linear algebra.

I implemented this in under 40 lines of Python:

```python
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)
```

**But here's the key insight:** doing this *once* isn't enough. A single attention head can only learn one pattern. What if a token needs to attend to *different* tokens for *different* reasons?

That's why we use **multiple heads.**

Each head has its own Q, K, V projections. One head might learn "attend to adjectives" while another learns "attend to verbs." The model discovers these patterns automatically. By stacking 4, 8, or 16 heads in parallel, we get a rich understanding of token relationships.

This is why Transformers are so powerful—they're not computing one fixed pattern, they're learning dozens of patterns simultaneously, then combining them.

---

## Part 3: The Transformer Block—Assembling the Machine

Okay, so we have multi-head attention. But a model needs more than just attention.

If you stack only attention layers, your model becomes very wide but shallow in understanding. You need to add **non-linearity**—something to let the model learn complex patterns.

Enter the **Feed-Forward Network (FFN):**

```python
self.ffn = nn.Sequential(
    nn.Linear(d_model, d_model * 4),  # Expand
    nn.ReLU(),                         # Non-linearity
    nn.Linear(d_model * 4, d_model),  # Contract
)
```

This is elegantly simple: expand your representation to 4× the width, apply ReLU (which kills negative values), then contract back down. The expansion gives the model capacity to learn complex interactions; the contraction forces it to compress that information back into the original space.

But here's the catch: if you apply attention, then FFN, then attention again, the gradients don't flow properly through deep networks. Early layers don't get strong learning signals. The model either doesn't train or trains very slowly.

The solution: **Residual Connections (Skip Connections).**

```python
x = x + attention(norm(x))  # Attention with residual
x = x + ffn(norm(x))       # FFN with residual
```

By adding the input directly to the output, you create a shortcut for gradients to flow backward. Suddenly, deep networks become trainable. This is why modern LLMs can have 40+ layers.

And notice: **we normalize BEFORE each sublayer, not after.** This is called Pre-LayerNorm, and it's the modern standard because it's more stable.

Here's the complete transformer block (simplified):

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Attention sublayer with Pre-LN and residual
        x = x + self.attention(self.norm1(x))
        
        # FFN sublayer with Pre-LN and residual
        x = x + self.ffn(self.norm2(x))
        
        return x
```

Stack these blocks on top of each other, add token embeddings at the bottom and a vocabulary projection at the top, and you have the architecture of GPT.

That's the entire thing. Not some mysterious black box—just attention, feedforward, norms, and residuals, repeated.

---

## Part 4: Training on Real Text—Where Theory Meets Reality

Here's where it gets interesting. I took the transformer I built and trained it on 2 million characters of real German literature from Project Gutenberg: Kafka's *Der Proceß*, Goethe's *Faust*, Schiller's *Die Räuber*.

### The Setup

**Tokenization:** First, I tokenized the corpus using BPE (Byte-Pair Encoding).

BPE is simple: start with characters, then repeatedly merge the most frequent adjacent pair. After ~200 merges, you have a vocabulary of 256 tokens. This is not a coincidence—256 is the number of bytes in a byte pair, so it's a natural stopping point.

```
"Donaudampfschifffahrtsgesellschaftskapitän"
→ ['D', 'o', 'na', 'u', 'da', 'm', 'p', 'f', 'sch', 'i', 'ff', ...]
→ 31 tokens (BPE)
```

**The Model:** 
- d_model: 64 (embedding dimension)
- n_heads: 4
- n_blocks: 2 (2 transformer layers)
- vocab_size: 256

This is tiny compared to production models (GPT-3 is 12,288 dimensions, 96 layers), but it's enough to see the dynamics.

**Training:**
- Next-token prediction (causal language modeling)
- 500 sequences × 128 tokens = 64,000 training examples
- 3 epochs
- Batch size: 16
- Optimizer: Adam (lr=1e-3)
- Loss: Cross-entropy

### The Results

Here's what the loss curve looked like:

**Epoch 1:** Loss drops from ~5.5 → 3.2 (rapid improvement)
**Epoch 2:** 3.2 → 2.8 (still learning)
**Epoch 3:** 2.8 → 2.6 (learning plateaus)

**Final Perplexity: 13.5**

What does this mean? The model is 13.5× more uncertain than if it had a perfect prior. That's actually impressive for a 100K-parameter model trained on 64K tokens. For comparison, a random baseline would have perplexity = vocab_size = 256.

### The Unexpected Discovery: Tokenization Comparison

This is where I learned something important. I benchmarked three tokenization approaches on the same corpus:

| Algorithm | Tokens | Fertility | Time |
|-----------|--------|-----------|------|
| **BPE** | 889,643 | 3.667 | 158.2s |
| **Unigram (SentencePiece)** | 938,456 | 3.867 | 0.9s ⚡ |
| **Morphology-aware + BPE** | 938,456 | 3.867 | 151.9s |

**The insight:** I expected Unigram (a probabilistic algorithm) to beat BPE (greedy merging). And morphology-aware pre-segmentation should surely help, given German's compound words, right?

Wrong.

BPE won. It produced 5% fewer tokens than the alternatives, despite being "just" a greedy algorithm. Why?

**Because greedy frequency matching is a better heuristic than I thought for natural language.** The algorithm doesn't need to be fancy if it's optimizing for the right objective: minimizing token count.

Morphology-aware pre-segmentation didn't help because BPE already discovers linguistic boundaries through frequency. When you pre-segment at morpheme boundaries, you're just doing BPE's job for it—and you're removing flexibility that BPE uses to find even better splits.

### What This Means for Building LLMs

Here's the key insight: **tokenization quality directly affects model convergence.**

Better tokenization → fewer tokens per word → smaller embedding matrix → faster training → better generalization.

This is why:
- English models can use 50K vocabulary size and be efficient
- German models need ~100K vocabulary to match English efficiency
- Subword tokenization beats character-level and word-level approaches

The 5% token efficiency difference between BPE and alternatives might not sound like much, but at billion-token training runs, that's 50 million fewer tokens—**tens of millions of dollars in compute saved.**

---

## Building This Yourself

If you want to follow along, I've open-sourced the complete educational code:

**Part 1:** [bpe_tokenizer.py](https://github.com/Sairam14/tinyLLM) — Tokenization from scratch
**Part 2:** [multi_head_attention.py](https://github.com/Sairam14/tinyLLM) — Attention mechanism
**Part 3:** [transformer_block.py](https://github.com/Sairam14/tinyLLM) — Complete block assembly
**Part 4:** [train_transformer.py](https://github.com/Sairam14/tinyLLM) — Training on real German text

Each file is under 300 lines and includes working demos. You can run any of them in seconds.

```bash
# See it all work
python demo.py                    # Tokenization
python multi_head_attention.py    # Attention
python transformer_block.py       # Blocks
python train_transformer.py       # Training
```

---

## Key Takeaways

1. **Transformers are simpler than they look.** Multi-head attention + FFN + residuals + norms. That's it. It all connects with ~50 lines of core code.

2. **Attention is learning relevance.** Each head learns which tokens matter for understanding other tokens. Multiple heads learn different relevance patterns.

3. **Residuals enable depth.** Without skip connections, gradients vanish in deep networks. With them, you can stack 40+ layers and still train efficiently.

4. **Tokenization is underrated.** The choice of algorithm matters less than the objective. BPE wins because it minimizes token count—which is the right objective for language modeling.

5. **Simple baselines are hard to beat.** Fancy methods (morphology-aware segmentation) underperformed greedy frequency matching. Occam's Razor applies to algorithms too.

---

## What's Next?

I'm currently working on:
- Extending this to multi-GPU training with gradient checkpointing
- Comparing token efficiency across language pairs (English, German, Chinese)
- Implementing KV-cache for efficient inference
- Building an OpenAI-compatible API around the model

If you're interested in LLM internals, production ML, or building models from first principles, I'd love to hear what you'd want to explore next.

**Questions? Ideas? Let me know in the comments.**

---

*This article is part of a larger educational series on building language models. Code, data, and detailed walkthroughs are available at [github.com/Sairam14/tinyLLM](https://github.com/Sairam14/tinyLLM).*

*Update: This article received great feedback. I've added a Part 5 covering sampling strategies (temperature, top-k, nucleus) and a Production section showing how to scale this to multi-GPU training with DDP.*
