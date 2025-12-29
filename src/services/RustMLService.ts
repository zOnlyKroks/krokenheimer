import { StoredMessage } from '../types/llm.js';
import { promises as fs } from 'fs';
import path from 'path';
import { createRequire } from 'module';

/**
 * Rust ML Service
 * High-performance machine learning service using Rust + Candle
 * Falls back to basic responses if Rust module is not available
 */
export class RustMLService {
  private rustModule: any = null;
  private isInitialized = false;
  private modelPath = './data/models/krokenheimer';
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
      const rustModulePath = path.resolve('./rust-ml');

      // Check if compiled Rust module exists
      try {
        console.log('[RustML] 🔍 Attempting to access module directory:', rustModulePath);
        await fs.access(rustModulePath);
        console.log('[RustML] ✅ Module directory exists, attempting to require...');

        // Check what's in the directory
        const dirContents = await fs.readdir(rustModulePath);
        console.log('[RustML] 📂 Directory contents:', dirContents);

        // Use createRequire for native modules in ES context
        const require = createRequire(import.meta.url);
        this.rustModule = require(rustModulePath);
        console.log('[RustML] ✅ Module loaded successfully, attempting model load...');

        // Try to load the model
        const modelLoaded = this.rustModule.loadModel(this.modelPath);
        if (modelLoaded) {
          this.isInitialized = true;
          console.log('[RustML] Successfully initialized with Rust module');
          return true;
        } else {
          console.log('[RustML] ⚠️ Module loaded but model failed to load from:', this.modelPath);
        }
      } catch (error) {
        console.log('[RustML] ❌ Rust module not available, using fallback mode');
        console.log('[RustML] 🔍 Module path attempted:', rustModulePath);
        console.log('[RustML] 📋 Error details:', error instanceof Error ? error.message : String(error));
        console.log('[RustML] 🗂️ Error stack:', error instanceof Error ? error.stack : 'No stack available');
        console.log('[RustML] 💡 This means training will not work! Build the module with: cd rust-ml && npm run build');
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

  async generateMessage(context: StoredMessage[], channelName: string, channelId?: string): Promise<string | undefined> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    if (this.rustModule) {
      return this.generateWithRust(context, channelName, channelId);
    }
  }

  async generateMentionResponse(context: StoredMessage[], messageContent: string, authorName: string): Promise<string | undefined> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    if (this.rustModule) {
      const result = await this.generateMentionWithRust(context, messageContent, authorName);
      if (result) {
        return result;
      }
    }

    return undefined;
  }

  private async generateWithRust(context: StoredMessage[], channelName: string, channelId?: string): Promise<string | undefined> {
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

    return undefined;
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

  private async generateMentionWithRust(context: StoredMessage[], messageContent: string, authorName: string): Promise<string | undefined> {
    try {
      // Use Rust module for mention responses
      const result = this.rustModule.generateMentionResponse(context, messageContent, authorName);

      if (result && result.length > 0) {
        return result;
      }
    } catch (error) {
      console.error('[RustML] Mention generation failed:', error);
    }

    return undefined;
  }

  async startTraining(): Promise<{success: boolean, error?: string}> {
    try {
      // Prepare training data from stored messages
      const trainingDataPath = await this.prepareTrainingData();
      const success = await this.trainModel(trainingDataPath, 10);

      if (success) {
        return { success: true };
      } else {
        return { success: false, error: 'Training failed - check logs for details' };
      }
    } catch (error) {
      console.error('[RustML] Start training failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  private async prepareTrainingData(): Promise<string> {
    // Create training data file from stored messages
    const dataPath = './data/training_data.jsonl';

    try {
      // Import the message storage service to get actual stored messages
      const { default: messageStorageService } = await import('./MessageStorageService.js');

      // Get messages from the database
      const messages = await messageStorageService.getAllMessages();

      if (!messages || messages.length === 0) {
        console.warn('[RustML] No stored messages found in database, cannot train');
        throw new Error('No training data available - collect some messages first');
      }

      // Format messages as conversation data for training
      const trainingData: any[] = [];

      // Group messages into conversations and format them
      for (let i = 0; i < messages.length - 1; i++) {
        const currentMsg = messages[i];
        const nextMsg = messages[i + 1];

        if(!currentMsg || !nextMsg) {
            throw new Error("[RustML] Current or next message empty")
        }

        // Skip if same author or if it's bot responding to itself
        if (currentMsg.authorName === nextMsg.authorName) continue;
        if (currentMsg.authorName === 'Krokenheimer') continue;

        // Create training conversation
          trainingData.push({
          messages: [
            { role: 'user', content: currentMsg.content },
            { role: 'assistant', content: nextMsg.authorName === 'Krokenheimer' ? nextMsg.content : '👍' }
          ]
        });
      }

      // Write training data as JSONL (one JSON object per line)
      const jsonlData = trainingData.map(item => JSON.stringify(item)).join('\n');
      await fs.writeFile(dataPath, jsonlData, 'utf8');

      console.log(`[RustML] Prepared ${trainingData.length} training conversations from ${messages.length} stored messages`);
      return dataPath;
    } catch (error) {
      console.error('[RustML] Failed to prepare training data:', error);
      throw error;
    }
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
      console.log('[RustML] ❌ Training requested but Rust module not available');
      console.log('[RustML] 🔍 Expected module location: ./rust-ml');
      console.log('[RustML] 💡 To fix: 1) cd rust-ml && npm run build, 2) verify index.node exists, 3) restart bot');
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