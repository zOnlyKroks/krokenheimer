import { promises as fs } from 'fs';
import { spawn } from 'child_process';
import messageStorageService from './MessageStorageService.js';
import ollamaService from './OllamaService.js';

export class FineTuningService {
  private isTraining = false;
  private messagesSinceLastTrain = 0;
  private lastTrainMessageCount = 0;
  private trainingThreshold = 50; // Retrain every x new messages
  private modelBaseName = 'krokenheimer';
  private modelVersion = 0;

  constructor() {
    this.loadState();
  }

  private async loadState(): Promise<void> {
    try {
      const data = await fs.readFile('./data/training_state.json', 'utf-8');
      const state = JSON.parse(data);
      this.lastTrainMessageCount = state.lastTrainMessageCount || 0;
      this.modelVersion = state.modelVersion || 0;

      // Calculate messages since last train
      const currentCount = messageStorageService.getTotalMessageCount();
      this.messagesSinceLastTrain = currentCount - this.lastTrainMessageCount;

      console.log(`📊 Training state loaded: ${this.messagesSinceLastTrain} new messages since last train`);
    } catch (error) {
      console.log('📊 No training state found, starting fresh');
    }
  }

  private async saveState(): Promise<void> {
    const state = {
      lastTrainMessageCount: this.lastTrainMessageCount,
      modelVersion: this.modelVersion,
      lastTrainDate: new Date().toISOString()
    };

    await fs.mkdir('./data', { recursive: true });
    await fs.writeFile('./data/training_state.json', JSON.stringify(state, null, 2));
  }

  incrementMessageCount(): void {
    this.messagesSinceLastTrain++;
  }

  shouldTrain(): boolean {
    return !this.isTraining && this.messagesSinceLastTrain >= this.trainingThreshold;
  }

  getTrainingStatus(): { isTraining: boolean; messagesSinceLastTrain: number; threshold: number } {
    return {
      isTraining: this.isTraining,
      messagesSinceLastTrain: this.messagesSinceLastTrain,
      threshold: this.trainingThreshold
    };
  }

  private async checkPythonEnvironment(): Promise<boolean> {
    console.log('🔍 Checking Python environment...');

    // Check if venv exists or python3 is available
    const pythonCmd = await fs.access('./venv/bin/python3')
      .then(() => './venv/bin/python3')
      .catch(() => 'python3');

    return new Promise((resolve) => {
      // Check if packages are installed (don't import unsloth as it needs GPU/CPU detection)
      const checkProcess = spawn(pythonCmd, [
        '-c',
        'import sys; import torch, transformers, trl, datasets; sys.exit(0)'
      ]);

      checkProcess.on('close', (code) => {
        if (code === 0) {
          console.log('✅ Python environment ready (torch, transformers, trl, datasets installed)');
          resolve(true);
        } else {
          console.error('❌ Training dependencies not installed');
          console.error('   Missing: torch, transformers, trl, or datasets');
          console.error('   Run: ./scripts/setup_training.sh');
          resolve(false);
        }
      });

      checkProcess.on('error', () => {
        console.error('❌ Python not found');
        console.error('   Run: ./scripts/setup_training.sh');
        resolve(false);
      });
    });
  }

  async exportTrainingData(): Promise<string> {
    console.log('📝 Exporting training data...');

    // Get all messages from the database
    const allChannels = messageStorageService.getActiveChannels();
    const trainingData: Array<{ messages: Array<{ role: string; content: string }> }> = [];

    for (const channel of allChannels) {
      const messages = messageStorageService.getMessagesByChannel(channel.channelId, 10000);

      // Group messages into conversation windows
      for (let i = 0; i < messages.length - 5; i += 3) {
        const window = messages.slice(i, i + 10);

        // Skip if window is too small
        if (window.length < 5) continue;

        // Format as conversation: context -> response
        const contextMessages = window.slice(0, -1).map(m =>
          `${m.authorName}: ${m.content}`
        ).join('\n');

        const response = window[window.length - 1];

        if(!response) {
            return "Bröken";
        }

        trainingData.push({
          messages: [
            {
              role: 'system',
              content: `You are a member of this Discord server who knows everything that has been said in all channels. You have perfect memory of every conversation. Match the tone and style of the server.`
            },
            {
              role: 'user',
              content: `Continue this conversation naturally:\n\n${contextMessages}`
            },
            {
              role: 'assistant',
              content: response.content
            }
          ]
        });
      }
    }

    console.log(`📊 Exported ${trainingData.length} training examples`);

    // Save as JSONL format (required by Ollama/most fine-tuning tools)
    const outputPath = './data/training_data.jsonl';
    await fs.mkdir('./data', { recursive: true });

    const jsonlContent = trainingData.map(entry => JSON.stringify(entry)).join('\n');
    await fs.writeFile(outputPath, jsonlContent);

    console.log(`✅ Training data saved to ${outputPath}`);
    return outputPath;
  }

  async createModelfile(baseModel: string): Promise<string> {
    const modelVersion = this.modelVersion + 1;
    const modelName = `${this.modelBaseName}-v${modelVersion}`;

    // Get config from OllamaService (pulled from .env)
    const config = ollamaService.getConfig();

    const modelfileContent = `FROM ${baseModel}

# Model parameters (from .env)
PARAMETER temperature ${config.temperature}
PARAMETER num_ctx ${config.contextWindow}

# System prompt
SYSTEM You are a member of this Discord server who knows everything that has been said in all channels. You have perfect memory of every conversation. Match the tone and style of the server. Be casual, authentic, and natural.
`;

    const modelfilePath = './data/Modelfile';
    await fs.writeFile(modelfilePath, modelfileContent);

    return modelfilePath;
  }

  async startTraining(): Promise<void> {
    if (this.isTraining) {
      console.log('⚠️  Training already in progress');
      return;
    }

    this.isTraining = true;
    console.log('🚀 Starting fine-tuning process...');

    try {
      // Check if Python environment is ready
      const isReady = await this.checkPythonEnvironment();
      if (!isReady) {
        console.error('❌ Python environment not ready!');
        console.error('   Run: ./scripts/setup_training.sh');
        this.isTraining = false;
        return;
      }

      // Export training data
      const trainingDataPath = await this.exportTrainingData();

      // Get current base model
      const baseModel = ollamaService.getConfig().model;

      // Check if we have enough data
      const totalMessages = messageStorageService.getTotalMessageCount();
      if (totalMessages < 500) {
        console.log('⚠️  Not enough messages for training (need at least 500)');
        this.isTraining = false;
        return;
      }

      console.log(`📚 Training with ${totalMessages} total messages`);
      console.log('⏳ This will take 3-7 days on CPU. Bot will continue working normally.');
      console.log('💡 Training runs in the background.');

      // Start TRUE fine-tuning with Unsloth (LoRA training)
      console.log('🔥 Starting TRUE LoRA fine-tuning with Unsloth...');
      await this.trainWithUnsloth(baseModel, trainingDataPath);

      // Update state
      this.lastTrainMessageCount = totalMessages;
      this.messagesSinceLastTrain = 0;
      this.modelVersion++;
      await this.saveState();

      console.log('✅ Model created successfully');
      console.log(`📦 New model: ${this.modelBaseName}-v${this.modelVersion}`);

      // Automatically switch to the new model
      await this.switchToNewModel();

    } catch (error) {
      console.error('❌ Training failed:', error);
    } finally {
      this.isTraining = false;
    }
  }

  private async trainWithUnsloth(baseModel: string, trainingDataPath: string): Promise<void> {
    return new Promise(async (resolve, reject) => {
      const modelName = `${this.modelBaseName}-v${this.modelVersion + 1}`;

      console.log(`🔥 Training ${modelName} with Unsloth LoRA...`);
      console.log(`   This performs TRUE fine-tuning on model weights`);

      // Convert Ollama model name to HuggingFace format
      // e.g., llama3.2:3b -> unsloth/Llama-3.2-3B-Instruct
      const hfModel = this.ollamaToHuggingFace(baseModel);

      console.log(`   Base model: ${hfModel}`);
      console.log(`   Training data: ${trainingDataPath}`);

      // Run Python training script (use venv if available)
      const pythonCmd = await fs.access('./venv/bin/python3')
        .then(() => './venv/bin/python3')
        .catch(() => 'python3');

      console.log(`   Python: ${pythonCmd}`);

      const pythonProcess = spawn(pythonCmd, [
        './scripts/train_unsloth.py',
        hfModel,
        trainingDataPath,
        modelName
      ]);

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        const text = data.toString();
        stdout += text;
        console.log(`   ${text.trim()}`);
      });

      pythonProcess.stderr.on('data', (data) => {
        const text = data.toString();
        stderr += text;
        // Python prints progress to stderr too
        console.log(`   ${text.trim()}`);
      });

      pythonProcess.on('close', async (code) => {
        if (code === 0) {
          console.log(`✅ Training complete!`);

          // Import GGUF into Ollama
          await this.importGGUFToOllama(modelName);

          resolve();
        } else {
          console.error('❌ Training script failed!');
          console.error('--- STDOUT ---');
          console.error(stdout);
          console.error('--- STDERR ---');
          console.error(stderr);
          reject(new Error(`Training failed with code ${code}. Check logs above for details.`));
        }
      });

      pythonProcess.on('error', (error) => {
        console.error('❌ Failed to spawn Python process:', error);
        reject(new Error(`Failed to start training: ${error.message}`));
      });
    });
  }

  private ollamaToHuggingFace(ollamaModel: string): string {
    // Map common Ollama models to standard HuggingFace models
    const modelMap: Record<string, string> = {
      'llama3.2:3b': 'meta-llama/Llama-3.2-3B-Instruct',
      'llama3.2:1b': 'meta-llama/Llama-3.2-1B-Instruct',
      'llama3.1:8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
      'llama3:8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
      'mistral:7b': 'mistralai/Mistral-7B-Instruct-v0.3',
    };

    // @ts-ignore
    return modelMap[ollamaModel] || modelMap['llama3.2:3b'];
  }

  private async importGGUFToOllama(modelName: string): Promise<void> {
    console.log(`📦 Importing trained model into Ollama...`);

    const ggufPath = `./data/models/${modelName}/*.gguf`;

    // Create Modelfile that references the GGUF
    const config = ollamaService.getConfig();

    const modelfileContent = `FROM ${ggufPath}

# Model parameters (from .env)
PARAMETER temperature ${config.temperature}
PARAMETER num_ctx ${config.contextWindow}

# System prompt
SYSTEM You are a member of this Discord server who knows everything that has been said in all channels. You have perfect memory of every conversation. Match the tone and style of the server. Be casual, authentic, and natural.
`;

    await fs.writeFile('./data/Modelfile-trained', modelfileContent);

    return new Promise((resolve, reject) => {
      const importProcess = spawn('ollama', ['create', modelName, '-f', './data/Modelfile-trained']);

      importProcess.stdout.on('data', (data) => {
        console.log(`   ${data.toString().trim()}`);
      });

      importProcess.stderr.on('data', (data) => {
        console.error(`   ${data.toString().trim()}`);
      });

      importProcess.on('close', (code) => {
        if (code === 0) {
          console.log(`✅ Model ${modelName} imported to Ollama!`);
          resolve();
        } else {
          reject(new Error(`Ollama import failed with code ${code}`));
        }
      });
    });
  }

  async switchToNewModel(): Promise<void> {
    const newModelName = `${this.modelBaseName}-v${this.modelVersion}`;

    console.log(`🔄 Switching to new model: ${newModelName}`);

    // Check if model exists
    const { Ollama } = await import('ollama');
    const ollama = new Ollama();

    try {
      const models = await ollama.list();
      const modelExists = models.models.some(m => m.name.includes(newModelName));

      if (modelExists) {
        ollamaService.setModel(newModelName);
        console.log(`✅ Now using model: ${newModelName}`);
      } else {
        console.log(`⚠️  Model ${newModelName} not found, keeping current model`);
      }
    } catch (error) {
      console.error('❌ Failed to switch model:', error);
    }
  }

  getCurrentModelName(): string {
    if (this.modelVersion > 0) {
      return `${this.modelBaseName}-v${this.modelVersion}`;
    }
    return ollamaService.getConfig().model;
  }
}

const fineTuningService = new FineTuningService();
export default fineTuningService;
