/**
 * Simple Embedding Service
 * Uses TF-IDF (Term Frequency-Inverse Document Frequency) for embeddings
 * NO pre-trained models - pure algorithmic approach
 */
export class SimpleEmbeddingService {
  private vocabulary: Map<string, number> = new Map();
  private idf: Map<string, number> = new Map();
  private documentCount = 0;
  private readonly embeddingDim = 384; // Fixed dimension

  /**
   * Simple tokenizer - splits text into words
   */
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0);
  }

  /**
   * Update vocabulary and IDF scores with new text
   */
  updateVocabulary(text: string): void {
    const tokens = this.tokenize(text);
    const uniqueTokens = new Set(tokens);

    // Add to vocabulary
    for (const token of uniqueTokens) {
      if (!this.vocabulary.has(token)) {
        this.vocabulary.set(token, this.vocabulary.size);
      }

      // Update IDF (count how many documents contain this term)
      const currentIdf = this.idf.get(token) || 0;
      this.idf.set(token, currentIdf + 1);
    }

    this.documentCount++;
  }

  /**
   * Generate TF-IDF embedding for text
   * Returns a fixed-length vector of 384 dimensions
   */
  generateEmbedding(text: string): number[] {
    const tokens = this.tokenize(text);
    const embedding = new Array(this.embeddingDim).fill(0);

    if (tokens.length === 0) {
      return embedding;
    }

    // Calculate term frequency
    const termFreq = new Map<string, number>();
    for (const token of tokens) {
      termFreq.set(token, (termFreq.get(token) || 0) + 1);
    }

    // Calculate TF-IDF and place into embedding vector
    for (const [term, freq] of termFreq.entries()) {
      const tf = freq / tokens.length;

      // Calculate IDF (inverse document frequency)
      const docFreq = this.idf.get(term) || 1;
      const idf = Math.log((this.documentCount + 1) / docFreq);

      const tfidf = tf * idf;

      // Get position in vocabulary
      const vocabIndex = this.vocabulary.get(term);
      if (vocabIndex !== undefined && vocabIndex < this.embeddingDim) {
        embedding[vocabIndex] = tfidf;
      } else if (vocabIndex !== undefined) {
        // Use hash for overflow vocabulary
        const hashIndex = vocabIndex % this.embeddingDim;
        embedding[hashIndex] += tfidf;
      }
    }

    // Normalize vector (L2 normalization)
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= magnitude;
      }
    }

    return embedding;
  }

  /**
   * Get vocabulary size
   */
  getVocabularySize(): number {
    return this.vocabulary.size;
  }

  /**
   * Get document count
   */
  getDocumentCount(): number {
    return this.documentCount;
  }
}

const simpleEmbeddingService = new SimpleEmbeddingService();
export default simpleEmbeddingService;
