import express, {Application, Request, Response} from 'express';
import multer from 'multer';
import {createReadStream, createWriteStream, existsSync, mkdirSync} from 'fs';
import {dirname, join, resolve} from 'path';
import {pipeline} from 'stream';
import {promisify} from 'util';
import {readFile, unlink, writeFile} from 'fs/promises';
import {MessageStorageService} from './MessageStorageService.js';
import {FineTuningService} from './FineTuningService.js';
import rustMLService from './RustMLService.js';

const pipelineAsync = promisify(pipeline);

// Simple logger for RemoteTrainingApiService
const logger = {
  info: (message: string, ...args: any[]) => console.log(`[INFO] ${message}`, ...args),
  warn: (message: string, ...args: any[]) => console.warn(`[WARN] ${message}`, ...args),
  error: (message: string, ...args: any[]) => console.error(`[ERROR] ${message}`, ...args)
};

interface RemoteTrainingConfig {
  port: number;
  authToken: string;
  maxFileSize: number;
  enableAuth: boolean;
}

export class RemoteTrainingApiService {
  private app: Application;
  private config: RemoteTrainingConfig;
  private messageStorageService: MessageStorageService;
  private fineTuningService: FineTuningService;
  private server: any = null;

  constructor(
    messageStorageService: MessageStorageService,
    fineTuningService: FineTuningService,
    config?: Partial<RemoteTrainingConfig>
  ) {
    this.messageStorageService = messageStorageService;
    this.fineTuningService = fineTuningService;

    // Default configuration
    this.config = {
      port: parseInt(process.env.REMOTE_API_PORT || '3000'),
      authToken: process.env.REMOTE_API_TOKEN || 'default-token-change-me',
      maxFileSize: 100 * 1024 * 1024, // 100MB
      enableAuth: process.env.REMOTE_API_AUTH !== 'false',
      ...config
    };

    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
  }

  private setupMiddleware(): void {
    // JSON parsing
    this.app.use(express.json({ limit: '50mb' }));

    // CORS for remote access
    this.app.use((req, res, next) => {
      res.header('Access-Control-Allow-Origin', '*');
      res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');

      if (req.method === 'OPTIONS') {
        res.sendStatus(200);
        return;
      }
      next();
    });

    // Request logging
    this.app.use((req, res, next) => {
      logger.info(`Remote API: ${req.method} ${req.url} from ${req.ip}`);
      next();
    });

    // Authentication middleware
    if (this.config.enableAuth) {
      this.app.use('/api', this.authenticateToken.bind(this));
    }
  }

  private authenticateToken(req: Request, res: Response, next: any): void {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
      res.status(401).json({ error: 'No token provided' });
      return;
    }

    if (token !== this.config.authToken) {
      res.status(403).json({ error: 'Invalid token' });
      return;
    }

    next();
  }

  private setupRoutes(): void {
    // Health check endpoint
    this.app.get('/api/health', (req: Request, res: Response) => {
      res.json({
        status: 'healthy',
        service: 'krokenheimer-remote-api',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      });
    });

    // Get training status
    this.app.get('/api/training/status', async (req: Request, res: Response) => {
      try {
        // Get current training status
        const trainingStatus = this.fineTuningService.getTrainingStatus();
        const totalMessages = await this.messageStorageService.getTotalMessageCount();
        const activeChannels = await this.messageStorageService.getActiveChannels();

        // Get training state from file
        const stateFilePath = resolve('./data/training_state.json');
        let trainingState = { lastTrainMessageCount: 0, modelVersion: 0, lastTrainDate: null };

        if (existsSync(stateFilePath)) {
          try {
            const stateData = await readFile(stateFilePath, 'utf-8');
            trainingState = JSON.parse(stateData);
          } catch (error) {
            logger.warn('Failed to read training state file:', error);
          }
        }

        res.json({
          training_in_progress: trainingStatus.isTraining,
          training_phase: trainingStatus.phase,
          current_step: trainingStatus.currentStep,
          total_steps: trainingStatus.totalSteps,
          current_epoch: trainingStatus.currentEpoch,
          total_epochs: trainingStatus.totalEpochs,
          current_loss: trainingStatus.currentLoss,
          elapsed_time: trainingStatus.elapsedTime,
          eta: trainingStatus.eta,
          total_messages: totalMessages,
          last_train_message_count: trainingState.lastTrainMessageCount,
          new_messages_since_last_train: totalMessages - trainingState.lastTrainMessageCount,
          model_version: trainingState.modelVersion,
          last_train_date: trainingState.lastTrainDate,
          active_channels: activeChannels.length,
          channels: activeChannels
        });
      } catch (error) {
        logger.error('Failed to get training status:', error);
        res.status(500).json({ error: 'Failed to get training status' });
      }
    });

    // Export training data
    this.app.post('/api/training/export', async (req: Request, res: Response) => {
      try {
        logger.info('Remote client requesting training data export...');

        const format = req.body.format || 'jsonl';
        if (format !== 'jsonl') {
          res.status(400).json({ error: 'Only JSONL format is currently supported' });
          return;
        }

        // Export training data
        const exportPath = await this.fineTuningService.exportTrainingData();
        if (!exportPath || !existsSync(exportPath)) {
          res.status(500).json({ error: 'Failed to export training data' });
          return;
        }

        // Stream the file to client
        res.setHeader('Content-Type', 'application/jsonl');
        res.setHeader('Content-Disposition', 'attachment; filename=training_data.jsonl');

        const fileStream = createReadStream(exportPath);
        await pipelineAsync(fileStream, res);

        logger.info('Training data exported successfully to remote client');
      } catch (error) {
        logger.error('Failed to export training data:', error);
        res.status(500).json({ error: 'Failed to export training data' });
      }
    });

    // Upload trained model
    const upload = multer({
      dest: './temp_uploads/',
      limits: { fileSize: this.config.maxFileSize },
      fileFilter: (req, file, cb) => {
        if (file.mimetype === 'application/zip' || file.originalname.endsWith('.zip')) {
          cb(null, true);
        } else {
          cb(new Error('Only ZIP files are allowed'));
        }
      }
    });

    this.app.post('/api/training/upload', upload.single('model'), async (req: Request, res: Response) => {
      try {
        if (!req.file) {
          res.status(400).json({ error: 'No model file provided' });
          return;
        }

        logger.info(`Receiving trained model upload: ${req.file.originalname}`);

        // Extract ZIP file to models directory
        const modelsDir = resolve('./data/models');
        const extractDir = join(modelsDir, 'krokenheimer');

        // Ensure directories exist
        if (!existsSync(modelsDir)) mkdirSync(modelsDir, { recursive: true });
        if (existsSync(extractDir)) {
          // Backup existing model
          const backupDir = join(modelsDir, `krokenheimer_backup_${Date.now()}`);
          await import('fs').then(fs => fs.promises.rename(extractDir, backupDir));
        }

        // Extract ZIP file
        const yauzl = await import('yauzl');
        await new Promise<void>((resolve, reject) => {
          yauzl.default.open(req.file!.path, { lazyEntries: true }, (err, zipfile) => {
            if (err) {
              reject(err);
              return;
            }

            mkdirSync(extractDir, { recursive: true });

            zipfile!.readEntry();
            zipfile!.on('entry', (entry) => {
              if (/\/$/.test(entry.fileName)) {
                // Directory entry
                try {
                  mkdirSync(join(extractDir, entry.fileName), { recursive: true });
                  zipfile!.readEntry();
                } catch (error) {
                  reject(error);
                }
              } else {
                // File entry
                zipfile!.openReadStream(entry, (err, readStream) => {
                  if (err) {
                    reject(err);
                    return;
                  }

                  try {
                    const outputPath = join(extractDir, entry.fileName);
                    const outputDir = dirname(outputPath);
                    mkdirSync(outputDir, { recursive: true });

                    const writeStream = createWriteStream(outputPath);
                    readStream!.pipe(writeStream);

                    writeStream.on('close', () => {
                      zipfile!.readEntry();
                    });

                    writeStream.on('error', (error) => {
                      reject(error);
                    });
                  } catch (error) {
                    reject(error);
                  }
                });
              }
            });

            zipfile!.on('end', () => {
              resolve();
            });

            zipfile!.on('error', (error) => {
              reject(error);
            });
          });
        });

        // Cleanup uploaded file
        await unlink(req.file.path);

        // Update training state
        const stateFilePath = resolve('./data/training_state.json');
        let trainingState = { lastTrainMessageCount: 0, modelVersion: 0, lastTrainDate: '' };

        if (existsSync(stateFilePath)) {
          try {
            const stateData = await readFile(stateFilePath, 'utf-8');
            trainingState = JSON.parse(stateData);
          } catch (error) {
            logger.warn('Failed to read existing training state');
          }
        }

        trainingState.lastTrainMessageCount = await this.messageStorageService.getTotalMessageCount();
        trainingState.modelVersion += 1;
        trainingState.lastTrainDate = new Date().toISOString();

        await writeFile(stateFilePath, JSON.stringify(trainingState, null, 2));

        logger.info(`✅ Model uploaded and extracted to ${extractDir}`);
        res.json({
          success: true,
          message: 'Model uploaded successfully',
          model_version: trainingState.modelVersion,
          extracted_to: extractDir
        });

      } catch (error) {
        logger.error('Failed to upload model:', error);
        res.status(500).json({ error: 'Failed to upload model' });

        // Cleanup on error
        if (req.file && existsSync(req.file.path)) {
          await unlink(req.file.path).catch(() => {});
        }
      }
    });

    // Training completion notification
    this.app.post('/api/training/complete', async (req: Request, res: Response) => {
      try {
        const { success, timestamp, model_path, gpu_type, client_version } = req.body;

        logger.info(`Remote training completed - Success: ${success}, GPU: ${gpu_type}, Client: ${client_version}`);

        // Log completion details
        const completion = {
          success,
          timestamp,
          model_path,
          gpu_type,
          client_version,
          received_at: new Date().toISOString()
        };

        // Store completion log
        const logPath = resolve('./data/remote_training_log.json');
        let logs: any[] = [];

        if (existsSync(logPath)) {
          try {
            const logData = await readFile(logPath, 'utf-8');
            logs = JSON.parse(logData);
          } catch (error) {
            logger.warn('Failed to read training log file');
          }
        }

        logs.push(completion);

        // Keep only last 50 entries
        if (logs.length > 50) {
          logs = logs.slice(-50);
        }

        await writeFile(logPath, JSON.stringify(logs, null, 2));

        res.json({
          success: true,
          message: 'Training completion received',
          logged: true
        });

      } catch (error) {
        logger.error('Failed to handle training completion:', error);
        res.status(500).json({ error: 'Failed to handle training completion' });
      }
    });

    // Get remote training logs
    this.app.get('/api/training/logs', async (req: Request, res: Response) => {
      try {
        const logPath = resolve('./data/remote_training_log.json');

        if (!existsSync(logPath)) {
          res.json({ logs: [] });
          return;
        }

        const logData = await readFile(logPath, 'utf-8');
        const logs = JSON.parse(logData);

        res.json({ logs });

      } catch (error) {
        logger.error('Failed to get training logs:', error);
        res.status(500).json({ error: 'Failed to get training logs' });
      }
    });

    // Check for force training flag
    this.app.get('/api/training/force-check', async (req: Request, res: Response) => {
      try {
        const forceFlagPath = resolve('./data/force_training.flag');

        if (!existsSync(forceFlagPath)) {
          res.json({ forceTraining: false, reason: 'No force training flag found' });
          return;
        }

        const flagData = await readFile(forceFlagPath, 'utf-8');
        const forceRequest = JSON.parse(flagData);

        // Check if flag is recent (within last hour to avoid stale requests)
        const requestTime = new Date(forceRequest.timestamp);
        const now = new Date();
        const hoursDiff = (now.getTime() - requestTime.getTime()) / (1000 * 60 * 60);

        if (hoursDiff > 1) {
          // Remove stale flag
          await unlink(forceFlagPath).catch(() => {});
          res.json({ forceTraining: false, reason: 'Force training flag expired (>1 hour old)' });
          return;
        }

        res.json({
          forceTraining: true,
          requestedBy: forceRequest.requestedByName,
          timestamp: forceRequest.timestamp,
          messageCount: forceRequest.messageCount
        });

        // Remove the flag after successful response so it's only used once
        await unlink(forceFlagPath).catch(() => {});

      } catch (error) {
        logger.error('Failed to check force training flag:', error);
        res.status(500).json({ error: 'Failed to check force training flag' });
      }
    });

    // NEW RUST ML ENDPOINTS

    // Get Rust ML service status and config
    this.app.get('/api/ml/status', async (req: Request, res: Response) => {
      try {
        const modelExists = await rustMLService.checkModelExists();
        const config = rustMLService.getConfig();
        const modelInfo = rustMLService.getModelInfo();
        const trainingStatus = await rustMLService.getTrainingStatus();

        res.json({
          service: 'rust-ml',
          model_exists: modelExists,
          model_info: modelInfo,
          config: config,
          training_status: trainingStatus,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        logger.error('Failed to get Rust ML status:', error);
        res.status(500).json({ error: 'Failed to get Rust ML status' });
      }
    });

    // Manually trigger local Rust training
    this.app.post('/api/ml/train', async (req: Request, res: Response) => {
      try {
        const { epochs = 3, force = false } = req.body;

        logger.info(`Manual Rust training requested - Epochs: ${epochs}, Force: ${force}`);

        // Check if training should start (unless forced)
        if (!force) {
          const totalMessages = await this.messageStorageService.getTotalMessageCount();

          // Get last training count
          const stateFilePath = resolve('./data/training_state.json');
          let lastTrainCount = 0;

          if (existsSync(stateFilePath)) {
            try {
              const stateData = await readFile(stateFilePath, 'utf-8');
              const state = JSON.parse(stateData);
              lastTrainCount = state.lastTrainMessageCount || 0;
            } catch (error) {
              logger.warn('Failed to read training state:', error);
            }
          }

          const shouldTrainingResult = await rustMLService.shouldStartTraining(
            totalMessages,
            lastTrainCount,
            1000
          );

          if (!shouldTrainingResult.shouldTrain) {
            res.status(400).json({
              success: false,
              reason: shouldTrainingResult.reason,
              message: 'Training criteria not met. Use force=true to override.',
              current_messages: totalMessages,
              last_train_count: lastTrainCount,
              new_messages: totalMessages - lastTrainCount
            });
            return;
          }
        }

        // Export training data
        const trainingDataPath = await this.fineTuningService.exportTrainingData();
        if (!trainingDataPath || !existsSync(trainingDataPath)) {
          res.status(500).json({
            success: false,
            error: 'Failed to export training data'
          });
          return;
        }

        // Start training (this is async)
        const trainingSuccess = await rustMLService.trainModel(trainingDataPath, epochs);

        if (trainingSuccess) {
          // Update training state
          const stateFilePath = resolve('./data/training_state.json');
          let trainingState = { lastTrainMessageCount: 0, modelVersion: 0, lastTrainDate: '' };

          if (existsSync(stateFilePath)) {
            try {
              const stateData = await readFile(stateFilePath, 'utf-8');
              trainingState = JSON.parse(stateData);
            } catch (error) {
              logger.warn('Failed to read existing training state');
            }
          }

          trainingState.lastTrainMessageCount = await this.messageStorageService.getTotalMessageCount();
          trainingState.modelVersion += 1;
          trainingState.lastTrainDate = new Date().toISOString();

          await writeFile(stateFilePath, JSON.stringify(trainingState, null, 2));

          logger.info('✅ Local Rust training completed successfully');
          res.json({
            success: true,
            message: 'Local Rust training completed successfully',
            model_version: trainingState.modelVersion,
            training_epochs: epochs,
            timestamp: new Date().toISOString()
          });
        } else {
          logger.error('❌ Local Rust training failed');
          res.status(500).json({
            success: false,
            error: 'Local Rust training failed'
          });
        }

      } catch (error) {
        logger.error('Failed to start local training:', error);
        res.status(500).json({
          success: false,
          error: 'Failed to start local training'
        });
      }
    });

    // Check if local training should start (for auto-training logic)
    this.app.get('/api/ml/should-train', async (req: Request, res: Response) => {
      try {
        const totalMessages = await this.messageStorageService.getTotalMessageCount();
        const threshold = parseInt(req.query.threshold as string) || 1000;

        // Get last training count
        const stateFilePath = resolve('./data/training_state.json');
        let lastTrainCount = 0;

        if (existsSync(stateFilePath)) {
          try {
            const stateData = await readFile(stateFilePath, 'utf-8');
            const state = JSON.parse(stateData);
            lastTrainCount = state.lastTrainMessageCount || 0;
          } catch (error) {
            logger.warn('Failed to read training state:', error);
          }
        }

        const result = await rustMLService.shouldStartTraining(totalMessages, lastTrainCount, threshold);

        res.json({
          should_train: result.shouldTrain,
          reason: result.reason,
          current_messages: totalMessages,
          last_train_count: lastTrainCount,
          new_messages: totalMessages - lastTrainCount,
          threshold: threshold,
          timestamp: new Date().toISOString()
        });

      } catch (error) {
        logger.error('Failed to check training criteria:', error);
        res.status(500).json({ error: 'Failed to check training criteria' });
      }
    });

    // Initialize Rust ML service
    this.app.post('/api/ml/initialize', async (req: Request, res: Response) => {
      try {
        const initialized = await rustMLService.initialize();

        res.json({
          success: true,
          initialized: initialized,
          message: initialized ? 'Rust ML service initialized successfully' : 'Rust ML service running in fallback mode',
          timestamp: new Date().toISOString()
        });

      } catch (error) {
        logger.error('Failed to initialize Rust ML service:', error);
        res.status(500).json({ error: 'Failed to initialize Rust ML service' });
      }
    });

    // 404 handler
    this.app.use((req: Request, res: Response) => {
      res.status(404).json({
        error: 'Endpoint not found',
        available_endpoints: [
          'GET /api/health',
          'GET /api/training/status',
          'POST /api/training/export',
          'POST /api/training/upload',
          'POST /api/training/complete',
          'GET /api/training/logs',
          'GET /api/training/force-check',
          'GET /api/ml/status',
          'POST /api/ml/train',
          'GET /api/ml/should-train',
          'POST /api/ml/initialize'
        ]
      });
    });
  }

  public async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.server = this.app.listen(this.config.port, '0.0.0.0', () => {
          logger.info(`🌐 Remote Training API started on port ${this.config.port}`);
          logger.info(`🔐 Authentication: ${this.config.enableAuth ? 'enabled' : 'disabled'}`);
          resolve();
        });

        this.server.on('error', (error: Error) => {
          logger.error('Remote Training API failed to start:', error);
          reject(error);
        });
      } catch (error) {
        reject(error);
      }
    });
  }

  public async stop(): Promise<void> {
    if (this.server) {
      return new Promise((resolve) => {
        this.server.close(() => {
          logger.info('Remote Training API stopped');
          resolve();
        });
      });
    }
  }

  public getConfig(): RemoteTrainingConfig {
    return { ...this.config };
  }

  public isRunning(): boolean {
    return this.server && this.server.listening;
  }
}