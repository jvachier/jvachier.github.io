---
layout: default
title: RAG as a Hilbert Space Problem
date: 2026-02-15
excerpt: Understanding Retrieval-Augmented Generation through the lens of Hilbert spaces, explaining why L2 normalization is essential for semantic similarity and how geometric search structures enable efficient document retrieval.
---

# RAG as a Hilbert Space Problem

## Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language model generation. Instead of relying solely on knowledge encoded in model parameters during training, RAG systems dynamically retrieve relevant documents from an external knowledge base and condition the language model's response on this retrieved context. This approach is widely used for question answering over enterprise documents, technical support systems, scientific literature search, and any application requiring factually grounded responses from large, evolving knowledge bases.

Large language models hallucinate, they generate fluent text that is factually wrong. RAG addresses this by grounding responses in retrieved documents, giving the model factual content to draw from rather than relying on parametric memory alone. When a user asks about a specific technical specification, the system retrieves the relevant documentation before generating a response, ensuring accuracy.

A RAG pipeline involves several steps:

1. Loading
2. Cleaning
3. Chunking 
4. Embedding 
5. Indexing 
6. Retrieval 
7. Generation 

This post focuses on steps 4 and 5, embedding and indexing, where Hilbert space geometry becomes essential. I explain why L2 normalization is mathematically necessary, how transformers build semantic representations, and how geometric search structures enable efficient retrieval at scale. At the end, I share a personal connection to this mathematical framework from my time at the Max Planck Institute.

**[→ View Full Implementation on GitHub](https://github.com/jvachier/scientific-literature-rag)**

**[→ View Transformer Implementation on GitHub](https://github.com/jvachier/Sentiment_Analysis)**

**[→ View Transformer Implementation on Kaggle](https://www.kaggle.com/code/jvachier/transformer-nmt-en-fr)**

## The Vocabulary Mismatch Problem: Why High-Dimensional Hilbert Space Matters

Keyword-based retrieval is not mathematically tractable. Consider a query $Q$ = "motor overheat" and a document $D$ = "engine temperature high alarm." These describe the same failure mode, yet:

$$Q \cap D = \emptyset$$

The terms share no overlap, so a keyword engine cannot retrieve $D$. The same holds in scientific literature: $Q$ = "motility-induced phase separation" and $D$ = "activity-driven clustering of self-propelled colloids". This is the vocabulary mismatch problem.

Keywords are discrete symbols with no notion of distance. To make semantic comparison tractable, text must be mapped into a space where distance carries meaning, a Hilbert space $\mathbb{R}^d$ with inner product $\langle \cdot, \cdot \rangle$, where similar documents point in similar directions and the angle between them quantifies similarity via cosine similarity:

$$\text{cos\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{||\mathbf{u}||_2 \; ||\mathbf{v}||_2} = \cos(\theta)$$

ranging from $-1$ (opposing) to $+1$ (identical direction). A value near $0$ means the vectors share no semantic content: "Fokker-Planck equation" and "chocolate recipe" would fall here.

I built a scientific literature RAG system around this framework. In my system, I use SPECTER2 embeddings with $d = 768$; production systems commonly use models like AWS Titan v2 with $d = 1024$ for general-domain applications.

## Embedding: From Text to Hypersphere

### Step 1: Tokenization

Starting with `"Fokker-Planck equation for active swimmers"`, the tokenizer produces sub-word units mapped to integer IDs:

```
["Fokker", "-", "Planck", "equation", "for", "active", "swim", "##mers"]   Shape: (8,)
```

"Swimmers" becomes "swim" + "##mers", rare terms are decomposed into recognizable sub-units.

### Step 2: Transformer Encoding

The most effective way to build semantically rich embeddings is through the transformer architecture, and specifically its attention mechanism, which allows each token to attend to all others and build context-aware representations. I provide a detailed walkthrough in my from-scratch transformer implementation for neural machine translation on [GitHub](https://github.com/jvachier/Sentiment_Analysis) and [Kaggle](https://www.kaggle.com/code/jvachier/transformer-nmt-en-fr); here, the key points.

Each token ID is first looked up in a learned embedding table, then self-attention contextualizes each vector. "Active" attends to "swim"/"##mers," forming the compound concept; "equation" attends to "Fokker"/"Planck," recognizing the named equation. After multiple attention layers, each token $i$ has a contextualized representation $\mathbf{h}_i \in \mathbb{R}^{768}$:

```
[
    [0.31, 0.62, ..., 0.40] -> Fokker 
    ...
    [0.22, 0.18, ..., 0.52] -> ##mers
]
Shape: (768, 8)
```

Each row is a 768-dimensional vector encoding the token's meaning in the context of this specific text.

### Step 3: Mean Pooling

To obtain a single embedding vector $\mathbf{e}$ for the text, we average across all token representations: 

$$\mathbf{e} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i$$

```
[0.29, 0.42, ..., 0.34]    Shape: (768,)
```

Mean pooling aggregates the contextual information from all tokens into one vector. This pooled vector $\mathbf{e} \in \mathbb{R}^{768}$ represents the entire text's meaning, ready for normalization and comparison.

Alternative pooling strategies exist: sentence-transformers supports weighted pooling (where token contributions are learned) and CLS token embeddings (using only the special classification token's representation). SPECTER2 (allenai/specter2_base), which I use in my scientific RAG system, is built on sentence-transformers but uses mean pooling as described above, finding this approach effective for scientific text.

### Step 4: L2 Normalization

We normalize $\mathbf{e}$ to unit length:

$$\hat{\mathbf{e}} = \frac{\mathbf{e}}{||\mathbf{e}||_2}$$

```
[0.017, 0.025, ..., 0.020]    Shape: (768,)
```

L2 normalization projects the embedding onto the unit hypersphere $S^{d-1}$. This is essential for two reasons. First, it ensures that similarity is measured by the angle between directions, capturing semantic meaning, rather than by vector magnitude, which reflects properties like document length that are irrelevant to meaning. Second, it reduces cosine similarity to a simple inner product: $\langle \hat{\mathbf{u}}, \hat{\mathbf{v}} \rangle = \cos(\theta)$, eliminating the norm divisions at every comparison and making retrieval significantly faster.

## Indexing: Geometric Search

Once embeddings are directions on $S^{d-1}$, retrieval is finding the closest directions in angle.

My scientific RAG uses ChromaDB with HNSW graphs, approximate $O(\log N)$ search with >95% recall, suitable for research-scale corpora ($\sim 10^4$ documents). For larger scale, FAISS provides `IndexFlatIP` (exhaustive, exact) and `IndexIVFFlat`, which partitions embedding space via Voronoi tessellation, clustering into cells via k-means and searching only the nearest cells at query time.

This is the same Voronoi partitioning we used in our research on [self-generated oxygen gradients controlling collective aggregation of photosynthetic microbes](https://royalsocietypublishing.org/rsif/article/18/185/20210553/90068), identical geometry applied to semantic space.

## Practical Lessons

Querying "Fokker-Planck equation" retrieves papers on "probability density evolution in stochastic systems." Active matter papers retrieve relevant stochastic thermodynamics work when the mathematics overlaps.

L2 normalization must be applied, without it, similarity rankings are meaningless. Abstracts are the right embedding unit (titles carry too little, full text adds noise). Domain-specific models like SPECTER2 matter more than index sophistication. Batch encoding gives ~100x speedup.

## A Note on History

Hilbert spaces are the foundational language of quantum mechanics: "The state space of a quantum system is a Hilbert space" (Appel, 2007). This mathematical framework has over a century of use across the physical sciences.

During my PhD at the Max Planck Institute in Göttingen, I attended lectures in the same room where David Hilbert taught these foundations. There is something fitting about applying his framework to finding meaning in text. The mathematics does not care whether it describes quantum states or semantic similarity. That universality is the point.

---

## References

Appel, W. (2007). *Mathematics for Physics and Physicists*. Princeton University Press.
