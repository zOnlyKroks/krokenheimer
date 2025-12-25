import { Ollama } from 'ollama';
import { LLMConfig, StoredMessage } from '../types/llm.js';

export class OllamaService {
  private ollama: Ollama;
  private readonly config: LLMConfig;

  constructor() {
    this.ollama = new Ollama();
    this.config = {
      model: process.env.OLLAMA_MODEL || 'llama3.2:3b',
      temperature: parseFloat(process.env.OLLAMA_TEMPERATURE || '0.9'),
      maxTokens: parseInt(process.env.OLLAMA_MAX_TOKENS || '200'),
      contextWindow: parseInt(process.env.OLLAMA_CONTEXT_WINDOW || '2048')
    };
  }

  async checkConnection(): Promise<boolean> {
    try {
      await this.ollama.list();
      return true;
    } catch (error) {
      console.error('Ollama connection failed:', error);
      return false;
    }
  }

  async ensureModelExists(): Promise<boolean> {
    try {
      const models = await this.ollama.list();
      console.log(`🔍 Checking for model: ${this.config.model}`);
      console.log(`📋 Available models:`, models.models.map(m => m.name));

      // @ts-ignore
      const hasModel = models.models.some(m => {
        const modelName = m.name;
        const targetName = this.config.model;
        // Match either exact name or base name (e.g., llama3.2 matches llama3.2:3b)
        // @ts-ignore
          return modelName === targetName || modelName.startsWith(targetName.split(':')[0]);
      });

      if (!hasModel) {
        console.log(`❌ Model ${this.config.model} not found. Please run: ollama pull ${this.config.model}`);
        return false;
      }

      console.log(`✅ Model ${this.config.model} is available`);
      return true;
    } catch (error) {
      console.error('❌ Failed to check model:', error);
      return false;
    }
  }

  async generateMessage(context: StoredMessage[], channelName: string): Promise<string> {
    try {
      // Build context from recent messages
      const contextText = this.buildContextPrompt(context, channelName);

      const response = await this.ollama.generate({
        model: this.config.model,
        prompt: contextText,
        options: {
          temperature: this.config.temperature,
          num_predict: this.config.maxTokens,
        }
      });

      // Clean up the response
      let generatedText = response.response.trim();

      // Remove any markdown formatting or code blocks
      generatedText = generatedText.replace(/```[\s\S]*?```/g, '');
      generatedText = generatedText.replace(/`[^`]+`/g, '');

      // Remove common AI prefixes
      generatedText = generatedText.replace(/^(Here's a message|Here is a message|Message:|Response:)\s*/i, '');

      // Take only the first message if multiple lines
      const lines = generatedText.split('\n').filter(l => l.trim().length > 0);
      generatedText = lines[0] || generatedText;

      // Limit length to reasonable Discord message size
      if (generatedText.length > 500) {
        generatedText = generatedText.substring(0, 500).trim();
        // Try to end at a sentence
        const lastPeriod = generatedText.lastIndexOf('.');
        const lastQuestion = generatedText.lastIndexOf('?');
        const lastExclamation = generatedText.lastIndexOf('!');
        const lastSentence = Math.max(lastPeriod, lastQuestion, lastExclamation);
        if (lastSentence > 100) {
          generatedText = generatedText.substring(0, lastSentence + 1);
        }
      }

      return generatedText;
    } catch (error) {
      console.error('Failed to generate message:', error);
      throw error;
    }
  }

  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await this.ollama.embeddings({
        model: this.config.model,
        prompt: text
      });

      return response.embedding;
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      throw error;
    }
  }

  private buildContextPrompt(messages: StoredMessage[], channelName: string): string {
    // Format recent messages
    const messageHistory = messages
      .slice(-20) // Last 20 messages
      .map(m => `${m.authorName}: ${m.content}`)
      .join('\n');

    return `You are participating in a Discord server chat in the #${channelName} channel. Generate a single, natural message that fits the conversation style and context. The message should be casual, authentic, and match the tone of recent messages.

Recent conversation:
${messageHistory}

Generate a single, short message (1-2 sentences) that would naturally continue this conversation. Do not include any labels, prefixes, or explanations. Just write the message itself.

Message:`;
  }

  getConfig(): LLMConfig {
    return { ...this.config };
  }

  setModel(model: string): void {
    this.config.model = model;
  }
}

const ollamaService = new OllamaService();
export default ollamaService;