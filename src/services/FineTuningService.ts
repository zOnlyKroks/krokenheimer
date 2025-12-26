import { promises as fs } from 'fs';
import { spawn } from 'child_process';
import messageStorageService from './MessageStorageService.js';
import ollamaService from './OllamaService.js';

export class FineTuningService {
  private isTraining = false;
  private messagesSinceLastTrain = 0;
  private lastTrainMessageCount = 0;
  private trainingThreshold = 999999; // Disabled - train manually with !llmtrain now
  private modelBaseName = 'krokenheimer';
  private modelVersion = 0;

  // Training progress tracking
  private trainingProgress = {
    currentStep: 0,
    totalSteps: 0,
    currentEpoch: 0,
    totalEpochs: 1,  // Updated to match train_text_lora.py
    currentLoss: 0,
    startTime: 0,
    phase: 'idle' as 'idle' | 'preparing' | 'training' | 'saving' | 'importing'
  };

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

  getTrainingStatus(): {
    isTraining: boolean;
    messagesSinceLastTrain: number;
    threshold: number;
    progress: {
      currentStep: number;
      totalSteps: number;
      currentEpoch: number;
      totalEpochs: number;
      currentLoss: number;
      startTime: number;
      phase: string;
      percentComplete: number;
      estimatedTimeRemaining: string;
      elapsedTime: string;
    };
  } {
    // Calculate progress percentage
    const percentComplete = this.trainingProgress.totalSteps > 0
      ? Math.round((this.trainingProgress.currentStep / this.trainingProgress.totalSteps) * 100)
      : 0;

    // Calculate elapsed time
    const elapsedMs = this.trainingProgress.startTime > 0
      ? Date.now() - this.trainingProgress.startTime
      : 0;
    const elapsedTime = this.formatDuration(elapsedMs);

    // Estimate time remaining
    let estimatedTimeRemaining = 'Calculating...';
    if (this.trainingProgress.currentStep > 0 && this.trainingProgress.totalSteps > 0) {
      const avgTimePerStep = elapsedMs / this.trainingProgress.currentStep;
      const stepsRemaining = this.trainingProgress.totalSteps - this.trainingProgress.currentStep;
      const remainingMs = avgTimePerStep * stepsRemaining;
      estimatedTimeRemaining = this.formatDuration(remainingMs);
    }

    return {
      isTraining: this.isTraining,
      messagesSinceLastTrain: this.messagesSinceLastTrain,
      threshold: this.trainingThreshold,
      progress: {
        currentStep: this.trainingProgress.currentStep,
        totalSteps: this.trainingProgress.totalSteps,
        currentEpoch: this.trainingProgress.currentEpoch,
        totalEpochs: this.trainingProgress.totalEpochs,
        currentLoss: this.trainingProgress.currentLoss,
        startTime: this.trainingProgress.startTime,
        phase: this.trainingProgress.phase,
        percentComplete,
        estimatedTimeRemaining,
        elapsedTime
      }
    };
  }

  private formatDuration(ms: number): string {
    if (ms === 0) return 'N/A';

    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) {
      const remainingHours = hours % 24;
      return `${days}d ${remainingHours}h`;
    } else if (hours > 0) {
      const remainingMinutes = minutes % 60;
      return `${hours}h ${remainingMinutes}m`;
    } else if (minutes > 0) {
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds}s`;
    } else {
      return `${seconds}s`;
    }
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

            console.log(`🔥 Training ${modelName} with CPU LoRA...`);

            // Initialize training progress
            this.trainingProgress = {
                currentStep: 0,
                totalSteps: 0,
                currentEpoch: 0,
                totalEpochs: 1,  // Updated to match train_text_lora.py
                currentLoss: 0,
                startTime: Date.now(),
                phase: 'preparing'
            };

            // Convert Ollama model to HuggingFace format
            const hfModel = this.ollamaToHuggingFace(baseModel);

            console.log(`   Base model: ${hfModel}`);
            console.log(`   Training data: ${trainingDataPath}`);

            // Run the CORRECT training script
            const pythonCmd = await fs.access('./venv/bin/python3')
                .then(() => './venv/bin/python3')
                .catch(() => 'python3');

            console.log(`   Python: ${pythonCmd}`);

            // Use the text LoRA training script
            const pythonProcess = spawn(pythonCmd, [
                './scripts/train_text_lora.py',
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

                // Parse training progress from output
                this.parseTrainingProgress(text);
            });

            pythonProcess.stderr.on('data', (data) => {
                const text = data.toString();
                stderr += text;
                console.log(`   ${text.trim()}`);

                // Also check stderr for progress (HuggingFace Trainer logs to stderr)
                this.parseTrainingProgress(text);
            });

            pythonProcess.on('close', async (code) => {
                if (code === 0) {
                    console.log(`✅ Training complete!`);

                    // Update phase to importing
                    this.trainingProgress.phase = 'importing';

                    // Import into Ollama
                    await this.importModelToOllama(modelName);

                    // Reset progress
                    this.trainingProgress.phase = 'idle';
                    resolve();
                } else {
                    console.error('❌ Training script failed!');
                    console.error('--- STDOUT ---');
                    console.error(stdout);
                    console.error('--- STDERR ---');
                    console.error(stderr);

                    // Reset progress
                    this.trainingProgress.phase = 'idle';
                    reject(new Error(`Training failed with code ${code}. Check logs above for details.`));
                }
            });

            pythonProcess.on('error', (error) => {
                console.error('❌ Failed to spawn Python process:', error);

                // Reset progress
                this.trainingProgress.phase = 'idle';
                reject(new Error(`Failed to start training: ${error.message}`));
            });
        });
    }

    private parseTrainingProgress(text: string): void {
        // Check for phase changes
        if (text.includes('STARTING TRAINING')) {
            this.trainingProgress.phase = 'training';
        } else if (text.includes('Training samples:')) {
            // Extract total steps estimate: samples / batch_size / gradient_accumulation
            const match = text.match(/Training samples:\s*(\d+)/);
            if (match) {
                // @ts-ignore
                const samples = parseInt(match[1]);
                // batch_size=1, gradient_accumulation=4, epochs=1 (updated to match train_text_lora.py)
                this.trainingProgress.totalSteps = Math.ceil((samples / 4) * 1);
            }
        } else if (text.includes('Saving model')) {
            this.trainingProgress.phase = 'saving';
        }

        // Parse HuggingFace Trainer progress logs
        // Format: {'loss': 2.5, 'learning_rate': 0.0001, 'epoch': 0.5, 'step': 10}
        const progressMatch = text.match(/\{'loss':\s*([\d.]+).*?'epoch':\s*([\d.]+).*?'step':\s*(\d+)/);
        if (progressMatch) {
            // @ts-ignore
            this.trainingProgress.currentLoss = parseFloat(progressMatch[1]);
            // @ts-ignore
            this.trainingProgress.currentEpoch = parseFloat(progressMatch[2]);
            // @ts-ignore
            this.trainingProgress.currentStep = parseInt(progressMatch[3]);
        }

        // Also handle progress bar format: [123/456 12:34 < 45:67, 1.23 it/s]
        const progressBarMatch = text.match(/\[(\d+)\/(\d+)/);
        if (progressBarMatch) {
            // @ts-ignore
            this.trainingProgress.currentStep = parseInt(progressBarMatch[1]);
            // @ts-ignore
            this.trainingProgress.totalSteps = parseInt(progressBarMatch[2]);
        }
    }

  private ollamaToHuggingFace(ollamaModel: string): string {
    // Map to OPEN models (no authentication required)
    const modelMap: Record<string, string> = {
      'llama3.2:3b': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',  // Open alternative
      'llama3.2:1b': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
      'llama3.1:8b': 'mistralai/Mistral-7B-Instruct-v0.2',  // Open
      'llama3:8b': 'mistralai/Mistral-7B-Instruct-v0.2',
      'mistral:7b': 'mistralai/Mistral-7B-Instruct-v0.2',
    };

    // @ts-ignore
    return modelMap[ollamaModel] || 'TinyLlama/TinyLlama-1.1B-Chat-v1.0';  // Default to TinyLlama (1.1B, fully open)
  }

    private async importModelToOllama(modelName: string): Promise<void> {
        console.log(`📦 Importing trained model into Ollama...`);

        const modelDir = `./data/models/${modelName}`;
        const modelfilePath = `${modelDir}/Modelfile`;

        // Check if Modelfile exists, create if not
        try {
            await fs.access(modelfilePath);
        } catch {
            // Create simple Modelfile
            const config = ollamaService.getConfig();
            const modelfileContent = `FROM ${modelDir}

PARAMETER temperature ${config.temperature}
PARAMETER num_ctx ${config.contextWindow}

SYSTEM You are a member of this Discord server who knows everything that has been said in all channels. You have perfect memory of every conversation. Match the tone and style of the server.`;

            await fs.writeFile(modelfilePath, modelfileContent);
        }

        return new Promise((resolve, reject) => {
            const importProcess = spawn('ollama', ['create', modelName, '-f', modelfilePath]);

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
