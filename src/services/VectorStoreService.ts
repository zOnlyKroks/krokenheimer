import {ChromaClient, Collection} from 'chromadb';
import {StoredMessage} from '../types/llm.js';
import simpleEmbeddingService from './SimpleEmbeddingService.js';

export class VectorStoreService {
  private client: ChromaClient;
  private collection: Collection | null = null;
  private collectionName = 'discord_messages';

  constructor() {
    this.client = new ChromaClient();
  }

  async initialize(): Promise<void> {
    try {
      // Get or create collection with TF-IDF dimensions
      this.collection = await this.client.getOrCreateCollection({
        name: this.collectionName,
        metadata: {
          description: 'Discord message embeddings (TF-IDF, 384 dimensions)',
          embeddingDimension: 384
        }
      });
      console.log('VectorStore initialized successfully (TF-IDF embeddings, 384 dimensions)');
    } catch (error) {
      console.error('Failed to initialize VectorStore:', error);
      throw error;
    }
  }

  async storeMessage(message: StoredMessage): Promise<void> {
    if (!this.collection) {
      throw new Error('VectorStore not initialized');
    }

    try {
      // Update vocabulary for TF-IDF
      simpleEmbeddingService.updateVocabulary(message.content);

      // Generate embedding using simple TF-IDF (no pre-trained models)
      const embedding = simpleEmbeddingService.generateEmbedding(message.content);

      // Store in ChromaDB
      await this.collection.add({
        ids: [message.id],
        embeddings: [embedding],
        metadatas: [{
          channelId: message.channelId,
          channelName: message.channelName,
          authorId: message.authorId,
          authorName: message.authorName,
          timestamp: message.timestamp.toString(),
          hasAttachments: message.hasAttachments.toString()
        }],
        documents: [message.content]
      });
    } catch (error) {
      console.error('Failed to store message in vector store:', error);
      // Don't throw - we don't want to crash on vector store failures
    }
  }

  async findSimilarMessages(query: string, channelId?: string, limit: number = 10): Promise<StoredMessage[]> {
    if (!this.collection) {
      throw new Error('VectorStore not initialized');
    }

    try {
      // Generate embedding for query using simple TF-IDF (no pre-trained models)
      const queryEmbedding = simpleEmbeddingService.generateEmbedding(query);

      // Build where clause
      const where = channelId ? { channelId } : undefined;

      // Query similar messages
      const results = await this.collection.query({
        queryEmbeddings: [queryEmbedding],
        nResults: limit,
        where
      });

      // Convert results back to StoredMessage format
      const messages: StoredMessage[] = [];

      if (results.ids && results.ids[0] && results.metadatas && results.metadatas[0] && results.documents && results.documents[0]) {
        for (let i = 0; i < results.ids[0].length; i++) {
          const metadata = results.metadatas[0][i];
          if (metadata && results.documents[0][i]) {
              messages.push({
                  // @ts-ignore
              id: results.ids[0][i],
              channelId: metadata.channelId as string,
              channelName: metadata.channelName as string,
              authorId: metadata.authorId as string,
              authorName: metadata.authorName as string,
              content: results.documents[0][i] as string,
              timestamp: parseInt(metadata.timestamp as string),
              hasAttachments: metadata.hasAttachments === 'true'
            });
          }
        }
      }

      return messages;
    } catch (error) {
      console.error('Failed to find similar messages:', error);
      return [];
    }
  }

  async getCollectionCount(): Promise<number> {
    if (!this.collection) {
      return 0;
    }

    try {
        return await this.collection.count();
    } catch (error) {
      console.error('Failed to get collection count:', error);
      return 0;
    }
  }

  async clear(): Promise<void> {
    if (!this.collection) {
      return;
    }

    try {
      await this.client.deleteCollection({ name: this.collectionName });
      await this.initialize();
    } catch (error) {
      console.error('Failed to clear collection:', error);
    }
  }
}

const vectorStoreService = new VectorStoreService();
export default vectorStoreService;