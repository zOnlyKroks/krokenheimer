import { promises as fs } from 'fs';
import messageStorageService from './MessageStorageService.js';
import trainingConfig from '../config/trainingConfig.js';
import { logger } from '../core/util/logger.js';

/**
 * FineTuningService for Remote Training Architecture
 *
 * This service handles data management and export for remote training.
 * Training is exclusively performed by remote Windows clients.
 * Local training capabilities have been removed.
 */
export class FineTuningService {
  private messagesSinceLastTrain = 0;
  private lastTrainMessageCount = 0;
  private modelBaseName = 'krokenheimer';
  private modelVersion = 0;

  // Remote training status (for API responses)
  private remoteTrainingStatus = {
    isTraining: false,
    phase: 'idle' as 'idle' | 'preparing' | 'training' | 'saving',
    currentStep: 0,
    totalSteps: 0,
    currentEpoch: 0,
    totalEpochs: 10,
    currentLoss: 0,
    startTime: 0,
    elapsedTime: 0,
    eta: 0
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
      const currentCount = await messageStorageService.getTotalMessageCount();
      this.messagesSinceLastTrain = currentCount - this.lastTrainMessageCount;

      logger.info(`📊 Training state loaded: ${this.messagesSinceLastTrain} new messages since last train`);
    } catch (error) {
      logger.info('📊 No training state found, starting fresh');
    }
  }

  public async saveState(): Promise<void> {
    const state = {
      lastTrainMessageCount: this.lastTrainMessageCount,
      modelVersion: this.modelVersion,
      lastTrainDate: new Date().toISOString()
    };

    await fs.mkdir('./data', { recursive: true });
    await fs.writeFile('./data/training_state.json', JSON.stringify(state, null, 2));
    logger.info('💾 Training state saved');
  }

  /**
   * Increment the message count (called when new messages are received)
   */
  incrementMessageCount(): void {
    this.messagesSinceLastTrain++;
  }

  /**
   * Check if training should be recommended (for remote clients)
   */
  shouldRecommendTraining(): boolean {
    // No automatic threshold - let remote clients decide based on their configuration
    return this.messagesSinceLastTrain > 0;
  }

  /**
   * Get current training status (for API responses)
   */
  getTrainingStatus(): {
    isTraining: boolean;
    messagesSinceLastTrain: number;
    phase: string;
    currentStep: number;
    totalSteps: number;
    currentEpoch: number;
    totalEpochs: number;
    currentLoss: number;
    startTime: number;
    elapsedTime: number;
    eta: number;
  } {
    return {
      isTraining: this.remoteTrainingStatus.isTraining,
      messagesSinceLastTrain: this.messagesSinceLastTrain,
      phase: this.remoteTrainingStatus.phase,
      currentStep: this.remoteTrainingStatus.currentStep,
      totalSteps: this.remoteTrainingStatus.totalSteps,
      currentEpoch: this.remoteTrainingStatus.currentEpoch,
      totalEpochs: this.remoteTrainingStatus.totalEpochs,
      currentLoss: this.remoteTrainingStatus.currentLoss,
      startTime: this.remoteTrainingStatus.startTime,
      elapsedTime: this.remoteTrainingStatus.elapsedTime,
      eta: this.remoteTrainingStatus.eta
    };
  }

  /**
   * Update remote training status (called by API when remote client reports progress)
   */
  updateRemoteTrainingStatus(status: Partial<typeof this.remoteTrainingStatus>): void {
    this.remoteTrainingStatus = { ...this.remoteTrainingStatus, ...status };
  }

  /**
   * Mark training as started (called when remote training begins)
   */
  markTrainingStarted(): void {
    this.remoteTrainingStatus.isTraining = true;
    this.remoteTrainingStatus.startTime = Date.now();
    this.remoteTrainingStatus.phase = 'preparing';
    logger.info('🚀 Remote training started');
  }

  /**
   * Mark training as completed (called when remote training finishes)
   */
  async markTrainingCompleted(success: boolean): Promise<void> {
    this.remoteTrainingStatus.isTraining = false;
    this.remoteTrainingStatus.phase = 'idle';

    if (success) {
      // Update training state
      const currentCount = await messageStorageService.getTotalMessageCount();
      this.lastTrainMessageCount = currentCount;
      this.messagesSinceLastTrain = 0;
      this.modelVersion++;
      await this.saveState();

      logger.info(`✅ Remote training completed successfully! Model version: ${this.modelVersion}`);
    } else {
      logger.error('❌ Remote training failed');
    }

    // Reset progress
    this.remoteTrainingStatus.currentStep = 0;
    this.remoteTrainingStatus.totalSteps = 0;
    this.remoteTrainingStatus.currentEpoch = 0;
    this.remoteTrainingStatus.currentLoss = 0;
    this.remoteTrainingStatus.startTime = 0;
    this.remoteTrainingStatus.elapsedTime = 0;
    this.remoteTrainingStatus.eta = 0;
  }

  /**
   * Export training data for remote training clients
   */
  async exportTrainingData(): Promise<string> {
    logger.info('📝 Exporting training data from database...');

    // Get all messages from database
    const allMessages = await messageStorageService.getAllMessages();
    logger.info(`📦 Loaded ${allMessages.length} messages from database`);

    // Filter messages based on training config
    const filteredMessages = allMessages.filter(msg =>
      trainingConfig.shouldIncludeInTraining(msg.authorId, msg.channelId)
    );

    const excludedCount = allMessages.length - filteredMessages.length;
    if (excludedCount > 0) {
      logger.info(`🚫 Filtered out ${excludedCount} messages (bot messages + excluded channels)`);
      logger.info(`   Excluded channels: ${trainingConfig.getExcludedChannels().join(', ')}`);
    }
    logger.info(`✅ Using ${filteredMessages.length} messages for training (from ALL servers)`);

    const trainingData: Array<{ messages: Array<{ role: string; content: string }> }> = [];

    // Group messages into conversation windows
    for (let i = 0; i < filteredMessages.length - 5; i += 3) {
      const window = filteredMessages.slice(i, i + 10);

      // Skip if window is too small
      if (window.length < 5) continue;

      // Format as conversation: context -> response
      const contextMessages = window.slice(0, -1).map(m =>
        `${m.authorName}: ${m.content}`
      ).join('\n');

      const response = window[window.length - 1];

      if (!response) {
        continue;
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

    logger.info(`📊 Created ${trainingData.length} training examples`);

    // Save as JSONL format
    const outputPath = './data/training_data.jsonl';
    await fs.mkdir('./data', { recursive: true });

    const jsonlContent = trainingData.map(entry => JSON.stringify(entry)).join('\n');
    await fs.writeFile(outputPath, jsonlContent);

    logger.info(`✅ Training data saved to ${outputPath}`);
    return outputPath;
  }

  /**
   * Get training statistics for display
   */
  async getTrainingStats(): Promise<{
    totalMessages: number;
    newMessages: number;
    lastTrainDate: string | null;
    modelVersion: number;
    modelName: string;
    remoteTraining: boolean;
  }> {
    const totalMessages = await messageStorageService.getTotalMessageCount();
    const activeChannels = await messageStorageService.getActiveChannels();

    let lastTrainDate = null;
    try {
      const stateData = await fs.readFile('./data/training_state.json', 'utf-8');
      const state = JSON.parse(stateData);
      lastTrainDate = state.lastTrainDate;
    } catch (error) {
      // No previous training
    }

    return {
      totalMessages,
      newMessages: this.messagesSinceLastTrain,
      lastTrainDate,
      modelVersion: this.modelVersion,
      modelName: this.modelBaseName,
      remoteTraining: true // This service only supports remote training
    };
  }

  /**
   * Check if sufficient data is available for training
   */
  async hasInsufficientData(): Promise<{ insufficient: boolean; reason?: string }> {
    const totalMessages = await messageStorageService.getTotalMessageCount();

    if (totalMessages < 500) {
      return {
        insufficient: true,
        reason: `Only ${totalMessages} messages available (need at least 500)`
      };
    }

    const allMessages = await messageStorageService.getAllMessages();
    const filteredMessages = allMessages.filter(msg =>
      trainingConfig.shouldIncludeInTraining(msg.authorId, msg.channelId)
    );

    if (filteredMessages.length < 100) {
      return {
        insufficient: true,
        reason: `Only ${filteredMessages.length} training messages after filtering (need at least 100)`
      };
    }

    return { insufficient: false };
  }

  /**
   * Get current model name/version
   */
  getCurrentModelName(): string {
    return `${this.modelBaseName} (v${this.modelVersion} - remote trained)`;
  }

  /**
   * Get model version number
   */
  getModelVersion(): number {
    return this.modelVersion;
  }

  /**
   * Set model version (used when remote client uploads new model)
   */
  setModelVersion(version: number): void {
    this.modelVersion = version;
  }

  /**
   * Get message count since last training
   */
  getMessagesSinceLastTrain(): number {
    return this.messagesSinceLastTrain;
  }

  /**
   * Get last training message count
   */
  getLastTrainMessageCount(): number {
    return this.lastTrainMessageCount;
  }

  /**
   * Reset message count (for testing/admin purposes)
   */
  async resetMessageCount(): Promise<void> {
    const currentCount = await messageStorageService.getTotalMessageCount();
    this.lastTrainMessageCount = currentCount;
    this.messagesSinceLastTrain = 0;
    await this.saveState();
    logger.info('🔄 Message count reset');
  }
}

const fineTuningService = new FineTuningService();
export default fineTuningService;