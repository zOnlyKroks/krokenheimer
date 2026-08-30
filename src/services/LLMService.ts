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
      const contextMessages = recentMessages
        .reverse()
        .slice(-this.config.maxContextMessages!)
        .map(msg => `${msg.username}: ${msg.content}`)
        .join('\n');

      const systemPrompt = `${botUsername}: Discord member. Match server tone and style. No formal AI responses.`;
      const prompt = `${systemPrompt}\n\n${contextMessages}\n${currentMessage}\n\n${botUsername}:`;

      const response = await this.ollama.generate({
        model: this.currentModel,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
        }
      });

      let cleanedResponse = response.response.trim();

      if (cleanedResponse.startsWith(`${botUsername}:`)) {
        cleanedResponse = cleanedResponse.substring(botUsername.length + 1).trim();
      }

      if (cleanedResponse.length > 2000) {
        cleanedResponse = cleanedResponse.substring(0, 1997) + '...';
      }

      return cleanedResponse;
    } catch (error) {
      console.error('Error generating LLM response:', error);
      throw error;
    }
  }

  // Real training is now handled by Python script (scripts/train_model.py)

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