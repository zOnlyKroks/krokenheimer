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

      const prompt = `${contextMessages}\n${currentMessage}`;

      const response = await this.ollama.generate({
        model: this.currentModel,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.8,
          top_p: 0.9,
        }
      });

      return response.response.trim();
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

      // Format messages as training data
      const trainingData = this.formatMessagesForTraining(messages);
      const trainingFilePath = join(dataDir, `training_${guildId}_${Date.now()}.txt`);
      writeFileSync(trainingFilePath, trainingData);

      // Create Modelfile for fine-tuning
      const modelfilePath = join(dataDir, `Modelfile_${guildId}`);
      const modelfileContent = `FROM ${this.config.baseModel}

# Fine-tuned on Discord server messages
TEMPLATE """{{ .Prompt }}"""

# Training data
ADAPTER ${trainingFilePath}

# Parameters for conversation
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"
`;

      writeFileSync(modelfilePath, modelfileContent);

      // Create the fine-tuned model
      console.log(`Creating fine-tuned model ${this.currentModel}...`);

      // Note: This creates a new model based on the base model
      // In practice, true fine-tuning requires external tools
      // This approach creates a custom model with adjusted parameters
      const modelfile = `FROM ${this.config.baseModel}

SYSTEM """You are a Discord bot. You've learned from the following conversation history in this server. Respond naturally and consistently with the tone and style you've observed:

${trainingData.slice(0, 50000)}"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
`;

      const tempModelfilePath = join(dataDir, `Modelfile_temp`);
      writeFileSync(tempModelfilePath, modelfile);

      // Create model using Ollama
      await this.ollama.create({
        model: this.currentModel,
        path: tempModelfilePath,
        stream: false,
      });

      console.log(`Model ${this.currentModel} created successfully`);
    } catch (error) {
      console.error('Error fine-tuning model:', error);
      throw error;
    }
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