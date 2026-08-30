import { Ollama } from 'ollama';
import type { StoredMessage } from './MessageDatabase.js';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';

export interface LLMConfig {
  baseModel: string;
  customModelName: string;
  ollamaHost?: string;
  maxContextMessages?: number;
}

export class LLMService {
  private ollama: Ollama;
  private config: LLMConfig;
  private currentModel: string;

  constructor(config: LLMConfig) {
    this.config = {
      maxContextMessages: 20,
      ...config,
    };

    this.ollama = new Ollama({
      host: config.ollamaHost || 'http://localhost:11434',
    });

    this.currentModel = this.config.customModelName;
  }

  public async generateResponse(
    currentMessage: string,
    recentMessages: StoredMessage[],
    botUsername: string
  ): Promise<string> {
    try {
      // Build context from recent messages
      const contextMessages = recentMessages
        .reverse() // Oldest to newest
        .slice(-this.config.maxContextMessages!)
        .map(msg => `${msg.username}: ${msg.content}`)
        .join('\n');

      const systemPrompt = `You are ${botUsername}, a member of this Discord server. Respond EXACTLY like the other users in this chat - same tone, same style, same energy. DO NOT act like an AI assistant or be formal. Just chat naturally like everyone else here.`;

      const prompt = `${systemPrompt}\n\nRecent conversation:\n${contextMessages}\n${currentMessage}\n\n${botUsername}:`;

      const response = await this.ollama.generate({
        model: this.currentModel,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 150,
        }
      });

      // Clean up response and limit length
      let cleanedResponse = response.response.trim();

      // Remove any potential username prefix if the model added it
      if (cleanedResponse.startsWith(`${botUsername}:`)) {
        cleanedResponse = cleanedResponse.substring(botUsername.length + 1).trim();
      }

      // Limit to 2000 chars (Discord limit)
      if (cleanedResponse.length > 2000) {
        cleanedResponse = cleanedResponse.substring(0, 1997) + '...';
      }

      return cleanedResponse;
    } catch (error) {
      console.error('Error generating LLM response:', error);
      throw error;
    }
  }

  public async fineTuneModel(messages: StoredMessage[], guildId: string): Promise<void> {
    try {
      const dataDir = './data/training';
      if (!existsSync(dataDir)) {
        mkdirSync(dataDir, { recursive: true });
      }

      console.log(`Processing ${messages.length} messages for training...`);

      // Filter and prepare quality training data
      const qualityMessages = this.filterQualityMessages(messages);
      console.log(`Filtered to ${qualityMessages.length} quality messages`);

      // Create conversation examples for training
      const conversationExamples = this.createConversationExamples(qualityMessages);
      console.log(`Created ${conversationExamples.length} conversation examples`);

      // Build comprehensive training prompt
      const trainingPrompt = this.buildTrainingPrompt(conversationExamples, qualityMessages);

      // Create the fine-tuned model
      console.log(`Creating fine-tuned model ${this.currentModel}...`);

      await this.ollama.create({
        model: this.currentModel,
        from: this.config.baseModel,
        system: trainingPrompt,
      });

      console.log(`Model ${this.currentModel} created successfully`);
    } catch (error) {
      console.error('Error fine-tuning model:', error);
      throw error;
    }
  }

  private filterQualityMessages(messages: StoredMessage[]): StoredMessage[] {
    return messages.filter(msg => {
      // Filter out low-quality messages
      if (!msg.content || msg.content.length < 3) return false;

      // Filter out pure command spam
      if (msg.content.startsWith('!') && msg.content.length < 20) return false;

      // Filter out pure URLs
      if (msg.content.startsWith('http') && !msg.content.includes(' ')) return false;

      // Filter out pure emoji spam (more than 80% emojis)
      const emojiRegex = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu;
      const emojiCount = (msg.content.match(emojiRegex) || []).length;
      if (emojiCount > msg.content.length * 0.8) return false;

      return true;
    });
  }

  private createConversationExamples(messages: StoredMessage[]): string[] {
    const examples: string[] = [];
    const conversationWindow = 5; // Look at groups of 5 messages

    for (let i = 0; i < messages.length - conversationWindow; i++) {
      const window = messages.slice(i, i + conversationWindow);

      // Create a context -> response pair
      const context = window.slice(0, -1).map(m => `${m.username}: ${m.content}`).join('\n');
      const response = window[window.length - 1];

      // Only include if response is substantive
      if (response.content.length > 10) {
        examples.push(`Context:\n${context}\n\nResponse:\n${response.username}: ${response.content}`);
      }
    }

    // Sample up to 100 best examples to keep training focused
    return this.sampleBestExamples(examples, 100);
  }

  private sampleBestExamples(examples: string[], maxCount: number): string[] {
    if (examples.length <= maxCount) return examples;

    // Sample evenly across the entire conversation history
    const step = Math.floor(examples.length / maxCount);
    const sampled: string[] = [];

    for (let i = 0; i < examples.length && sampled.length < maxCount; i += step) {
      sampled.push(examples[i]);
    }

    return sampled;
  }

  private buildTrainingPrompt(examples: string[], allMessages: StoredMessage[]): string {
    // Analyze conversation style
    const usernames = new Set(allMessages.map(m => m.username));
    const avgLength = allMessages.reduce((sum, m) => sum + m.content.length, 0) / allMessages.length;

    // Extract common topics/themes (simple keyword extraction)
    const commonWords = this.extractCommonThemes(allMessages);

    const examplesText = examples.slice(0, 50).join('\n\n---\n\n');

    return `You are a chat participant. Below are real conversations from this Discord server. Learn the speaking style, topics, and tone from these examples and respond EXACTLY like the other users would.

${examplesText}

Common topics: ${commonWords.slice(0, 10).join(', ')}
Typical message length: ${Math.round(avgLength)} chars
Style: Match the casual, natural tone you see above. NO formal AI responses.`;
  }

  private extractCommonThemes(messages: StoredMessage[]): string[] {
    // Simple word frequency analysis
    const wordCounts = new Map<string, number>();
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'it', 'this', 'that', 'i', 'you', 'he', 'she', 'we', 'they']);

    messages.forEach(msg => {
      const words = msg.content.toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(w => w.length > 3 && !stopWords.has(w) && !w.startsWith('http'));

      words.forEach(word => {
        wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
      });
    });

    // Sort by frequency and return top words
    return Array.from(wordCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([word]) => word);
  }

  private formatMessagesForTraining(messages: StoredMessage[]): string {
    // Format messages as natural conversation
    return messages
      .map(msg => `${msg.username}: ${msg.content}`)
      .join('\n');
  }

  public async checkModelAvailable(modelName?: string): Promise<boolean> {
    try {
      const checkModel = modelName || this.currentModel;
      const models = await this.ollama.list();
      return models.models.some(m => m.name === checkModel || m.name.startsWith(checkModel));
    } catch (error) {
      console.error('Error checking Ollama models:', error);
      return false;
    }
  }

  public async pullBaseModel(): Promise<void> {
    console.log(`Pulling base model ${this.config.baseModel}...`);
    await this.ollama.pull({ model: this.config.baseModel, stream: false });
    console.log(`Base model ${this.config.baseModel} pulled successfully`);
  }

  public async initializeModel(): Promise<void> {
    const customExists = await this.checkModelAvailable(this.currentModel);

    if (customExists) {
      console.log(`Using existing custom model: ${this.currentModel}`);
      return;
    }

    const baseExists = await this.checkModelAvailable(this.config.baseModel);

    if (!baseExists) {
      console.log(`Base model ${this.config.baseModel} not found, pulling...`);
      await this.pullBaseModel();
    }

    // Use base model until first fine-tune
    this.currentModel = this.config.baseModel;
    console.log(`Using base model: ${this.currentModel}`);
  }
}