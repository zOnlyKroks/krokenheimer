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
    totalEpochs: 10,  // From-scratch training needs more epochs
    currentLoss: 0,
    startTime: 0,
    phase: 'idle' as 'idle' | 'preparing' | 'training' | 'saving'
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
    console.log('📝 Exporting training data from database...');

    // Get all messages from database in a single efficient query
    const allMessages = messageStorageService.getAllMessages();
    console.log(`📦 Loaded ${allMessages.length} messages from database`);

    const trainingData: Array<{ messages: Array<{ role: string; content: string }> }> = [];

    // Group messages into conversation windows
    for (let i = 0; i < allMessages.length - 5; i += 3) {
      const window = allMessages.slice(i, i + 10);

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

    console.log(`📊 Created ${trainingData.length} training examples`);

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

      // Check if we have enough data
      const totalMessages = messageStorageService.getTotalMessageCount();
      if (totalMessages < 500) {
        console.log('⚠️  Not enough messages for training (need at least 500)');
        this.isTraining = false;
        return;
      }

      console.log(`📚 Training with ${totalMessages} total messages`);
      console.log('⏳ This will take several hours on CPU. Bot will continue working normally.');
      console.log('💡 Training runs in the background.');

      // Start from-scratch training (incremental)
      console.log('🔥 Starting from-scratch training (GPT-2 Small)...');
      await this.trainFromScratch(trainingDataPath);

      // Update state
      this.lastTrainMessageCount = totalMessages;
      this.messagesSinceLastTrain = 0;
      this.modelVersion++;
      await this.saveState();

      console.log('✅ Model training complete!');
      console.log(`📦 Model: ${this.modelBaseName} (incrementally trained)`);

    } catch (error) {
      console.error('❌ Training failed:', error);
    } finally {
      this.isTraining = false;
    }
  }

    private async trainFromScratch(trainingDataPath: string): Promise<void> {
        return new Promise(async (resolve, reject) => {
            const modelName = this.modelBaseName; // Single model, incrementally trained

            console.log(`🔥 Training from scratch (GPT-2 Small 124M params)...`);

            // Initialize training progress
            this.trainingProgress = {
                currentStep: 0,
                totalSteps: 0,
                currentEpoch: 0,
                totalEpochs: 10,  // From-scratch needs more epochs
                currentLoss: 0,
                startTime: Date.now(),
                phase: 'preparing'
            };

            console.log(`   Training data: ${trainingDataPath}`);

            // Check if we have a checkpoint to resume from
            const modelDir = `./data/models/${modelName}`;
            let resumeCheckpoint = null;

            try {
                const checkpoints = await fs.readdir(modelDir);
                const checkpointDirs = checkpoints.filter(f => f.startsWith('checkpoint-'));

                if (checkpointDirs.length > 0) {
                    // Get latest checkpoint
                    checkpointDirs.sort((a, b) => {
                        const numA = parseInt(a.replace('checkpoint-', ''));
                        const numB = parseInt(b.replace('checkpoint-', ''));
                        return numB - numA;
                    });
                    resumeCheckpoint = `${modelDir}/${checkpointDirs[0]}`;
                    console.log(`📂 Resuming from checkpoint: ${checkpointDirs[0]}`);
                }
            } catch (error) {
                console.log('📝 Starting fresh training (no checkpoint found)');
            }

            // Run the from-scratch training script
            const pythonCmd = await fs.access('./venv/bin/python3')
                .then(() => './venv/bin/python3')
                .catch(() => 'python3');

            console.log(`   Python: ${pythonCmd}`);

            const args = [
                './scripts/train_from_scratch.py',
                trainingDataPath,
                modelName,
                '--epochs', '10'
            ];

            if (resumeCheckpoint) {
                args.push('--resume', resumeCheckpoint);
            }

            const pythonProcess = spawn(pythonCmd, args);

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
                    console.log(`📦 Model saved to: ${modelDir}`);

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
        if (text.includes('STARTING TRAINING FROM SCRATCH')) {
            this.trainingProgress.phase = 'training';
        } else if (text.includes('Training samples:')) {
            // Extract total steps estimate: samples / batch_size / gradient_accumulation / epochs
            const match = text.match(/Training samples:\s*(\d+)/);
            if (match) {
                // @ts-ignore
                const samples = parseInt(match[1]);
                // batch_size=1, gradient_accumulation=4, epochs=10 (from train_from_scratch.py)
                this.trainingProgress.totalSteps = Math.ceil((samples / 4) * 10);
            }
        } else if (text.includes('Saving final model')) {
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

  getCurrentModelName(): string {
    return `${this.modelBaseName} (v${this.modelVersion} - trained from scratch)`;
  }
}

const fineTuningService = new FineTuningService();
export default fineTuningService;
