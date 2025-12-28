/**
 * Remote Training Configuration for API-Based Architecture
 * Manages configuration for remote Windows clients that connect via REST API
 * SSH-based training has been removed in favor of client-initiated training
 */

export interface RemoteTrainingSettings {
  // API Configuration
  apiEnabled: boolean;
  apiPort: number;
  apiToken: string;
  enableAuth: boolean;

  // Training thresholds and policies
  minMessagesThreshold: number;
  trainingIntervalHours: number;
  maxConcurrentClients: number;

  // Model configuration
  defaultEpochs: number;
  modelBaseName: string;

  // Client preferences
  recommendGpu: boolean;
  preferredGpuType: 'rocm' | 'directml' | 'cpu';

  // Logging and monitoring
  logRemoteTraining: boolean;
  notifyOnCompletion: boolean;
}

export class RemoteTrainingConfig {
  private settings: RemoteTrainingSettings | null = null;
  private initialized = false;

  constructor() {
    // Initialize after environment variables are loaded
  }

  /**
   * Initialize the config after dotenv has loaded
   */
  public initialize(): void {
    if (!this.initialized) {
      console.log('🔍 Initializing remote training API config...');
      this.settings = this.loadFromEnvironment();
      this.initialized = true;
    }
  }

  private ensureSettingsLoaded(): void {
    if (this.settings === null) {
      // Return safe defaults if not initialized
      this.settings = {
        apiEnabled: true,
        apiPort: 3000,
        apiToken: 'default-token-change-me',
        enableAuth: true,
        minMessagesThreshold: 1000,
        trainingIntervalHours: 12,
        maxConcurrentClients: 1,
        defaultEpochs: 10,
        modelBaseName: 'krokenheimer',
        recommendGpu: true,
        preferredGpuType: 'rocm',
        logRemoteTraining: true,
        notifyOnCompletion: true
      };
    }
  }

  private loadFromEnvironment(): RemoteTrainingSettings {
    console.log('🔍 Loading remote training API environment variables...');
    console.log(`   REMOTE_API_ENABLED: "${process.env.REMOTE_API_ENABLED}"`);
    console.log(`   REMOTE_API_PORT: "${process.env.REMOTE_API_PORT}"`);
    console.log(`   REMOTE_API_TOKEN: ${process.env.REMOTE_API_TOKEN ? '[SET]' : '[NOT SET]'}`);
    console.log(`   REMOTE_API_AUTH: "${process.env.REMOTE_API_AUTH}"`);

    const settings: RemoteTrainingSettings = {
      // API Configuration
      apiEnabled: process.env.REMOTE_API_ENABLED !== 'false',
      apiPort: parseInt(process.env.REMOTE_API_PORT || '3000'),
      apiToken: process.env.REMOTE_API_TOKEN || 'default-token-change-me',
      enableAuth: process.env.REMOTE_API_AUTH !== 'false',

      // Training thresholds
      minMessagesThreshold: parseInt(process.env.REMOTE_TRAINING_MIN_MESSAGES || '1000'),
      trainingIntervalHours: parseInt(process.env.REMOTE_TRAINING_INTERVAL_HOURS || '12'),
      maxConcurrentClients: parseInt(process.env.REMOTE_TRAINING_MAX_CLIENTS || '1'),

      // Model configuration
      defaultEpochs: parseInt(process.env.REMOTE_TRAINING_DEFAULT_EPOCHS || '10'),
      modelBaseName: process.env.REMOTE_TRAINING_MODEL_NAME || 'krokenheimer',

      // GPU preferences (recommendations for clients)
      recommendGpu: process.env.REMOTE_TRAINING_RECOMMEND_GPU !== 'false',
      preferredGpuType: (process.env.REMOTE_TRAINING_PREFERRED_GPU as 'rocm' | 'directml' | 'cpu') || 'rocm',

      // Monitoring
      logRemoteTraining: process.env.REMOTE_TRAINING_LOG !== 'false',
      notifyOnCompletion: process.env.REMOTE_TRAINING_NOTIFY !== 'false'
    };

    console.log('🔍 Parsed API settings:');
    console.log(`   API enabled: ${settings.apiEnabled}`);
    console.log(`   API port: ${settings.apiPort}`);
    console.log(`   Auth enabled: ${settings.enableAuth}`);
    console.log(`   Min messages threshold: ${settings.minMessagesThreshold}`);
    console.log(`   Training interval: ${settings.trainingIntervalHours}h`);
    console.log(`   Preferred GPU: ${settings.preferredGpuType}`);

    return settings;
  }

  /**
   * Get remote training settings
   */
  getSettings(): RemoteTrainingSettings {
    this.ensureSettingsLoaded();
    return { ...this.settings! };
  }

  /**
   * Check if remote training API is enabled
   */
  isRemoteTrainingApiEnabled(): boolean {
    this.ensureSettingsLoaded();
    return this.settings!.apiEnabled;
  }

  /**
   * Get API configuration for the remote training service
   */
  getApiConfig(): { port: number; token: string; enableAuth: boolean } {
    this.ensureSettingsLoaded();
    return {
      port: this.settings!.apiPort,
      token: this.settings!.apiToken,
      enableAuth: this.settings!.enableAuth
    };
  }

  /**
   * Get training policy configuration
   */
  getTrainingPolicy(): {
    minMessagesThreshold: number;
    trainingIntervalHours: number;
    maxConcurrentClients: number;
    defaultEpochs: number;
  } {
    this.ensureSettingsLoaded();
    return {
      minMessagesThreshold: this.settings!.minMessagesThreshold,
      trainingIntervalHours: this.settings!.trainingIntervalHours,
      maxConcurrentClients: this.settings!.maxConcurrentClients,
      defaultEpochs: this.settings!.defaultEpochs
    };
  }

  /**
   * Get GPU recommendations for remote clients
   */
  getGpuRecommendations(): { recommendGpu: boolean; preferredType: string } {
    this.ensureSettingsLoaded();
    return {
      recommendGpu: this.settings!.recommendGpu,
      preferredType: this.settings!.preferredGpuType
    };
  }

  /**
   * Update settings (for dynamic configuration)
   */
  updateSettings(newSettings: Partial<RemoteTrainingSettings>): void {
    this.ensureSettingsLoaded();
    this.settings = { ...this.settings!, ...newSettings };
  }

  /**
   * Get status information for reporting
   */
  getStatusInfo(): {
    apiEnabled: boolean;
    apiPort: number;
    authEnabled: boolean;
    minMessagesThreshold: number;
    trainingIntervalHours: number;
    preferredGpuType: string;
    loggingEnabled: boolean;
  } {
    this.ensureSettingsLoaded();
    return {
      apiEnabled: this.settings!.apiEnabled,
      apiPort: this.settings!.apiPort,
      authEnabled: this.settings!.enableAuth,
      minMessagesThreshold: this.settings!.minMessagesThreshold,
      trainingIntervalHours: this.settings!.trainingIntervalHours,
      preferredGpuType: this.settings!.preferredGpuType,
      loggingEnabled: this.settings!.logRemoteTraining
    };
  }

  /**
   * Validate API token (for backwards compatibility with any remaining checks)
   */
  isValidApiToken(token: string): boolean {
    this.ensureSettingsLoaded();
    return token === this.settings!.apiToken;
  }

  /**
   * Check if training should be recommended based on thresholds
   */
  shouldRecommendTraining(messagesSinceLastTrain: number, hoursSinceLastTrain: number): boolean {
    this.ensureSettingsLoaded();

    const meetsMsgThreshold = messagesSinceLastTrain >= this.settings!.minMessagesThreshold;
    const meetsTimeThreshold = hoursSinceLastTrain >= this.settings!.trainingIntervalHours;

    return meetsMsgThreshold && meetsTimeThreshold;
  }

  /**
   * Get model configuration
   */
  getModelConfig(): { baseName: string; defaultEpochs: number } {
    this.ensureSettingsLoaded();
    return {
      baseName: this.settings!.modelBaseName,
      defaultEpochs: this.settings!.defaultEpochs
    };
  }

  /**
   * Legacy method compatibility - always returns false since we removed SSH
   * @deprecated Use isRemoteTrainingApiEnabled() instead
   */
  isRemoteTrainingAvailable(): boolean {
    console.warn('⚠️ isRemoteTrainingAvailable() is deprecated. Use isRemoteTrainingApiEnabled() instead.');
    return false; // SSH-based training is no longer available
  }

  /**
   * Legacy method compatibility - always returns false since we removed local training
   * @deprecated Remote training no longer falls back to local training
   */
  shouldFallbackToLocal(): boolean {
    console.warn('⚠️ shouldFallbackToLocal() is deprecated. Local training fallback has been removed.');
    return false;
  }

  /**
   * Legacy method compatibility
   * @deprecated Use getGpuRecommendations() instead
   */
  getGpuConfig(): { useGpu: boolean; type: string } {
    console.warn('⚠️ getGpuConfig() is deprecated. Use getGpuRecommendations() instead.');
    this.ensureSettingsLoaded();
    return {
      useGpu: this.settings!.recommendGpu,
      type: this.settings!.preferredGpuType
    };
  }

  /**
   * Legacy method compatibility
   * @deprecated Use getTrainingPolicy().defaultEpochs instead
   */
  getPreferredTrainingLocation(): 'remote' | 'local' {
    console.warn('⚠️ getPreferredTrainingLocation() is deprecated. Training is now exclusively remote.');
    return 'remote';
  }
}

const remoteTrainingConfig = new RemoteTrainingConfig();
export default remoteTrainingConfig;