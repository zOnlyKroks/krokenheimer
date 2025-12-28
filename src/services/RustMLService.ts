import { StoredMessage } from '../types/llm.js';
import { promises as fs } from 'fs';
import path from 'path';

/**
 * Rust ML Service
 * High-performance machine learning service using Rust + Candle
 * Falls back to basic responses if Rust module is not available
 */
export class RustMLService {
  private rustModule: any = null;
  private isInitialized = false;
  private modelPath = './data/models/krokenheimer-rust';
  private fallbackResponses = [
    "👍",
    "I see!",
    "Interesting!",
    "Got it!",
    "Nice!",
    "Cool!",
    "Yep!",
    "Alright!",
    "Understood!"
  ];

  async initialize(): Promise<boolean> {
    try {
      // Try to load the Rust module
      const rustModulePath = path.resolve('./rust-ml/index.node');

      // Check if compiled Rust module exists
      try {
        await fs.access(rustModulePath);
        this.rustModule = require(rustModulePath);

        // Try to load the model
        const modelLoaded = this.rustModule.loadModel(this.modelPath);
        if (modelLoaded) {
          this.isInitialized = true;
          console.log('[RustML] Successfully initialized with Rust module');
          return true;
        }
      } catch (error) {
        console.log('[RustML] Rust module not available, using fallback mode');
      }

      // Fallback initialization
      this.isInitialized = true;
      return false; // Indicates fallback mode
    } catch (error) {
      console.error('[RustML] Failed to initialize:', error);
      return false;
    }
  }

  async checkModelExists(): Promise<boolean> {
    if (this.rustModule) {
      // Check using Rust module
      try {
        const configPath = path.join(this.modelPath, 'config.json');
        const tokenizerPath = path.join(this.modelPath, 'tokenizer.json');
        const weightsPath = path.join(this.modelPath, 'model.safetensors');

        await fs.access(configPath);
        await fs.access(tokenizerPath);
        await fs.access(weightsPath);
        return true;
      } catch {
        return false;
      }
    }

    // Fallback check - look for any model files
    try {
      await fs.access(`${this.modelPath}/config.json`);
      return true;
    } catch {
      return false;
    }
  }

  async generateMessage(context: StoredMessage[], channelName: string, channelId?: string): Promise<string> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    if (this.rustModule) {
      return this.generateWithRust(context, channelName, channelId);
    } else {
      return this.generateFallback(context, channelName);
    }
  }

  async generateMentionResponse(context: StoredMessage[], messageContent: string, authorName: string): Promise<string> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    if (this.rustModule) {
      return this.generateMentionWithRust(context, messageContent, authorName);
    } else {
      return this.generateMentionFallback(messageContent, authorName);
    }
  }

  private async generateWithRust(context: StoredMessage[], channelName: string, channelId?: string): Promise<string> {
    try {
      // Build context for Rust module
      const prompt = this.buildContextPrompt(context, channelName);

      // Generate using Rust module
      const result = this.rustModule.generateText(prompt, 100, 0.9);

      if (result && result.length > 0) {
        return result;
      }
    } catch (error) {
      console.error('[RustML] Generation failed:', error);
    }

    return this.getRandomFallback();
  }

  private async generateMentionWithRust(context: StoredMessage[], messageContent: string, authorName: string): Promise<string> {
    try {
      // Use Rust module for mention responses
      const result = this.rustModule.generateMentionResponse(context, messageContent, authorName);

      if (result && result.length > 0) {
        return result;
      }
    } catch (error) {
      console.error('[RustML] Mention generation failed:', error);
    }

    return this.getRandomFallback();
  }

  private generateFallback(context: StoredMessage[], channelName: string): Promise<string> {
    // Simple fallback logic
    const recentMessage = context[context.length - 1];

    if (recentMessage) {
      const content = recentMessage.content.toLowerCase();

      // Simple keyword-based responses
      if (content.includes('hello') || content.includes('hi')) {
        return Promise.resolve('Hello!');
      }
      if (content.includes('how are you')) {
        return Promise.resolve('Doing well, thanks!');
      }
      if (content.includes('thanks') || content.includes('thank you')) {
        return Promise.resolve('You\'re welcome!');
      }
      if (content.includes('good') || content.includes('great') || content.includes('awesome')) {
        return Promise.resolve('Glad to hear that!');
      }
      if (content.includes('help')) {
        return Promise.resolve('How can I assist you?');
      }
    }

    return Promise.resolve(this.getRandomFallback());
  }

  private generateMentionFallback(messageContent: string, authorName: string): Promise<string> {
    const responses = [
      `Hi ${authorName}!`,
      `What's up, ${authorName}?`,
      `Hey there!`,
      `Hello!`,
      `Yes?`,
      `How can I help?`
    ];

    const randomResponse = responses[Math.floor(Math.random() * responses.length)];
    return Promise.resolve(randomResponse);
  }

  private buildContextPrompt(context: StoredMessage[], channelName: string): string {
    let prompt = '';

    // Limit context to avoid token overflow
    const limitedContext = context.slice(-8);

    limitedContext.forEach(msg => {
      const content = msg.content.length > 150 ? msg.content.substring(0, 150) + '...' : msg.content;
      prompt += `${msg.authorName}: ${content}\n`;
    });

    prompt += 'Krokenheimer: ';
    return prompt;
  }

  private getRandomFallback(): string {
    return this.fallbackResponses[Math.floor(Math.random() * this.fallbackResponses.length)];
  }

  // Training methods
  async shouldStartTraining(messageCount: number, lastTrainCount: number, threshold: number = 1000): Promise<{shouldTrain: boolean, reason: string}> {
    if (this.rustModule) {
      try {
        const result = this.rustModule.shouldStartTraining(messageCount, lastTrainCount, threshold);
        return {
          shouldTrain: result.shouldTrain,
          reason: result.reason
        };
      } catch (error) {
        console.error('[RustML] Training check failed:', error);
      }
    }

    // Fallback logic
    const newMessages = messageCount - lastTrainCount;
    const shouldTrain = newMessages >= threshold;

    return {
      shouldTrain,
      reason: shouldTrain
        ? `Ready to train with ${newMessages} new messages`
        : `Only ${newMessages} new messages (need ${threshold})`
    };
  }

  async trainModel(trainingDataPath: string, epochs: number = 3): Promise<boolean> {
    if (this.rustModule) {
      try {
        const outputPath = `./data/models/krokenheimer-rust-${Date.now()}`;
        const result = this.rustModule.trainModel(trainingDataPath, outputPath, epochs);

        if (result) {
          // Update model path to new trained model
          this.modelPath = outputPath;
          // Reload the model
          await this.initialize();
          console.log(`[RustML] Training completed successfully. New model: ${outputPath}`);
          return true;
        }
      } catch (error) {
        console.error('[RustML] Training failed:', error);
      }
    } else {
      console.log('[RustML] Training requested but Rust module not available');
    }

    return false;
  }

  async getTrainingStatus(): Promise<{training_in_progress: boolean, current_epoch?: number, total_epochs?: number}> {
    if (this.rustModule) {
      try {
        return this.rustModule.checkTrainingStatus();
      } catch (error) {
        console.error('[RustML] Training status check failed:', error);
      }
    }

    return { training_in_progress: false };
  }

  getConfig() {
    if (this.rustModule) {
      try {
        return this.rustModule.getConfig();
      } catch (error) {
        console.error('[RustML] Config retrieval failed:', error);
      }
    }

    return {
      model: 'krokenheimer-rust (fallback mode)',
      temperature: 0.9,
      maxTokens: 100,
      contextWindow: 512,
      backend: 'fallback'
    };
  }

  getModelInfo() {
    if (this.rustModule) {
      try {
        return this.rustModule.getModelInfo();
      } catch (error) {
        console.error('[RustML] Model info retrieval failed:', error);
      }
    }

    return {
      model: 'krokenheimer-rust',
      version: '0.1.0',
      status: 'fallback_mode'
    };
  }
}

const rustMLService = new RustMLService();
export default rustMLService;