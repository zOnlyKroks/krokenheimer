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

  async generateMessage(context: StoredMessage[], channelName: string, channelId?: string): Promise<string> {
    try {
      // Get relevant historical messages using vector search ACROSS ALL CHANNELS
      let historicalContext: StoredMessage[] = [];
      if (context.length > 0) {
        const vectorStoreService = (await import('./VectorStoreService.js')).default;

        // Use recent messages to create a search query
        const recentTopics = context.slice(-5).map(m => m.content).join(' ');
        // Search ALL channels (no channelId filter) - bot knows everything
        historicalContext = await vectorStoreService.findSimilarMessages(recentTopics, undefined, 20);

        // Remove duplicates that are already in recent context
        const recentIds = new Set(context.map(m => m.id));
        historicalContext = historicalContext.filter(m => !recentIds.has(m.id));
      }

      // Build context from historical + recent messages
      const contextText = this.buildContextPrompt(context, channelName, historicalContext);

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

  async generateMentionResponse(context: StoredMessage[], mentionContent: string, mentionAuthor: string): Promise<string> {
    try {
      // Get relevant historical messages using vector search ACROSS ALL CHANNELS
      let historicalContext: StoredMessage[] = [];
      if (context.length > 0) {
        const vectorStoreService = (await import('./VectorStoreService.js')).default;

        // Search based on the mention content
        const cleanedContent = mentionContent.replace(/<@!?\d+>/g, '').trim();
        if (cleanedContent) {
          historicalContext = await vectorStoreService.findSimilarMessages(cleanedContent, undefined, 20);

          // Remove duplicates
          const recentIds = new Set(context.map(m => m.id));
          historicalContext = historicalContext.filter(m => !recentIds.has(m.id));
        }
      }

      // Build context for mention response
      const contextText = this.buildMentionPrompt(context, mentionContent, mentionAuthor, historicalContext);

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
      generatedText = generatedText.replace(/^(Here's a response|Here is a response|Response:|Reply:)\s*/i, '');

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
      console.error('Failed to generate mention response:', error);
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

  private buildContextPrompt(messages: StoredMessage[], channelName: string, historicalContext?: StoredMessage[]): string {
    // Format recent messages (increased from 20 to 50)
    const recentHistory = messages
      .slice(-50)
      .map(m => `${m.authorName}: ${m.content}`)
      .join('\n');

    // Format historical context if available (includes ALL channels)
    let historicalSection = '';
    if (historicalContext && historicalContext.length > 0) {
      const historicalMessages = historicalContext
        .map(m => `[#${m.channelName}] ${m.authorName}: ${m.content}`)
        .join('\n');

      historicalSection = `\n\nRelevant past conversations from across the entire server:\n${historicalMessages}\n`;
    }

    return `You are a member of this Discord server who knows EVERYTHING that has been said in ALL channels. You have perfect memory of every conversation. You're now writing in #${channelName}. Generate a single, natural message that fits the current conversation. Be casual, authentic, and match the tone and patterns you've learned from the entire server.
${historicalSection}
Current conversation in #${channelName}:
${recentHistory}

Generate a single, short message (1-2 sentences) that would naturally continue this conversation based on everything you know from the entire server. Do not include any labels, prefixes, or explanations. Just write the message itself.

Message:`;
  }

  private buildMentionPrompt(messages: StoredMessage[], mentionContent: string, mentionAuthor: string, historicalContext?: StoredMessage[]): string {
    // Format recent messages for context
    const messageHistory = messages
      .slice(-30) // Last 30 messages
      .map(m => `${m.authorName}: ${m.content}`)
      .join('\n');

    // Format historical context if available (includes ALL channels)
    let historicalSection = '';
    if (historicalContext && historicalContext.length > 0) {
      const historicalMessages = historicalContext
        .map(m => `[#${m.channelName}] ${m.authorName}: ${m.content}`)
        .join('\n');

      historicalSection = `\n\nRelevant past conversations from across the entire server:\n${historicalMessages}\n`;
    }

    // Remove the bot mention from the content to get the actual question/message
    const cleanedContent = mentionContent.replace(/<@!?\d+>/g, '').trim();

    return `You are a member of this Discord server who knows EVERYTHING that has been said in ALL channels. You have perfect memory of every conversation. ${mentionAuthor} just mentioned you with the following message:

"${cleanedContent}"
${historicalSection}
Recent conversation context:
${messageHistory}

Respond directly to ${mentionAuthor}'s message in a natural, conversational way. Be helpful, friendly, and concise. Match the tone of the server. Use your knowledge from all channels if relevant. If they're asking a question, answer based on everything you know. If they're just saying hi or making a comment, respond appropriately.

Do not include any labels, prefixes, or explanations. Just write your response directly.

Response:`;
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