import { Client, Message, TextChannel } from 'discord.js';
import type { BotPlugin, BotCommand } from '../types/index.js';
import type { ExtensibleBot } from '../core/Bot.js';
import messageStorageService from '../services/MessageStorageService.js';
import vectorStoreService from '../services/VectorStoreService.js';
import rustMLService from '../services/RustMLService.js';
import fineTuningService from '../services/FineTuningService.js';
import trainingConfig from '../config/trainingConfig.js';
import cron from 'node-cron';
import { MessageGenerationConfig } from '../types/llm.js';

export class LLMPlugin implements BotPlugin {
  name = 'LLMPlugin';
  description = 'Automated message learning and generation using local LLM';
  version = '1.0.0';

  private client: Client | null = null;
  private bot: ExtensibleBot | null = null;
  private isInitialized = false;
  private scheduledTask: cron.ScheduledTask | null = null;
  private scanTask: cron.ScheduledTask | null = null;
  private config: MessageGenerationConfig;
  private lastGenerationTime = 0;
  private scanIntervalMinutes = 2;
  private lastScanTime = 0;
  private isScanning = false;

  commands: BotCommand[] = [
    {
      name: 'llmstats',
      description: 'Show LLM learning statistics',
      execute: this.showStats.bind(this),
    },
    {
      name: 'llmscan',
      description: 'Configure or trigger message scanning',
      usage: 'llmscan [interval_minutes|now|full]',
      execute: this.configureScan.bind(this),
    },
    {
      name: 'llmtrain',
      description: 'Manage model training',
      usage: 'llmtrain [status|force]',
      execute: this.manageTrain.bind(this),
    },
    {
      name: 'llmclear',
      description: 'Clear vector store (use after switching embedding methods)',
      execute: this.clearVectorStore.bind(this),
    },
      {
          name: 'llmstore',
          description: 'Store collected data into vector store',
          execute: this.embedVectorStore.bind(this),
      },
    {
      name: 'llmexclude',
      description: 'Manage channels excluded from training',
      usage: 'llmexclude [add|remove|list] [channelId]',
      execute: this.manageExclusions.bind(this),
    },
    {
      name: 'llmconfig',
      description: 'Show training configuration',
      execute: this.showTrainingConfig.bind(this),
    },
    {
      name: 'llmrust',
      description: 'Manage local Rust ML training system',
      usage: 'llmrust [status|logs]',
      execute: this.manageRustML.bind(this),
    }
  ];

  constructor() {
    // Load config from environment variables
    this.config = {
      enabled: process.env.LLM_ENABLED !== 'false', // Enabled by default
      minIntervalMinutes: parseInt(process.env.LLM_MIN_INTERVAL_MINUTES || '60'),
      maxIntervalMinutes: parseInt(process.env.LLM_MAX_INTERVAL_MINUTES || '180'),
      channelIds: process.env.LLM_CHANNEL_IDS?.split(',').filter(id => id.trim()),
      excludeChannelIds: process.env.LLM_EXCLUDE_CHANNEL_IDS?.split(',').filter(id => id.trim())
    };

    // Load scan interval from env or default to 2 minutes
    this.scanIntervalMinutes = parseInt(process.env.LLM_SCAN_INTERVAL_MINUTES || '2');
  }

  async initialize(client: Client, bot: ExtensibleBot): Promise<void> {
    this.client = client;
    this.bot = bot;

    console.log('🤖 Initializing LLM Plugin...');

    // Remote training API has been replaced with local Rust ML training
    console.log('🦀 Using local Rust ML training (remote training API disabled)');

    // Initialize Rust ML service and check for trained model
    console.log('1️⃣  Initializing Rust ML service...');
    const rustInitialized = await rustMLService.initialize();
    const modelExists = await rustMLService.checkModelExists();

    if (rustInitialized && modelExists) {
      console.log('✅ Rust ML service initialized with trained model');
    } else if (rustInitialized && !modelExists) {
      console.log('⚠️  Rust ML service running in fallback mode - no trained model found yet.');
      console.log('💡 Train your model with: !llmtrain now or POST /api/ml/train');
      console.log('   Model will be trained from scratch using YOUR Discord messages.');
      console.log('   Bot will respond once training is complete.');
    } else {
      console.log('⚠️  Rust ML service running in fallback mode - Rust module not compiled');
      console.log('💡 Compile Rust module for better performance: cd rust-ml && cargo build --release');
    }

    // Initialize vector store (ChromaDB should already be ready)
    console.log('3️⃣  Initializing ChromaDB...');
    try {
      await vectorStoreService.initialize();
      console.log('✅ ChromaDB initialized');
    } catch (error) {
      console.error('❌ Failed to initialize ChromaDB');
      console.error('Error:', error);
      return;
    }

    // Set up message collection listener
    client.on('messageCreate', async (message) => {
      // Handle mentions first (before collecting)
      if (message.mentions.has(client.user!.id) && !message.author.bot) {
        await this.handleMention(message);
      }

      await this.collectMessage(message);
    });

    this.isInitialized = true;
    console.log('✅ LLM Plugin initialized successfully');

    // Start scheduled message generation if enabled
    if (this.config.enabled) {
      this.startScheduledGeneration();
    }

    // Start scheduled channel scanning
    this.startChannelScanning();
  }

  private async handleMention(message: Message): Promise<void> {
    try {
      console.log(`🔔 Bot mentioned by ${message.author.username} in #${message.channel.isDMBased() ? 'DM' : (message.channel as TextChannel).name}`);

      // Check if channel is excluded from learning
      if (trainingConfig.isChannelExcluded(message.channel.id)) {
        const channelName = message.channel.isDMBased() ? 'DM' : (message.channel as TextChannel).name;
        console.log(`⚠️  Mention ignored - #${channelName} is excluded from training`);
        await message.reply('Sorry, I don\'t respond in this channel because it\'s excluded from my training.');
        return;
      }

      // Show typing indicator
      if ('sendTyping' in message.channel) {
        await message.channel.sendTyping().catch(() => {});
      }

      // Get recent messages for context
      const channelId = message.channel.id;
      const recentMessages = messageStorageService.getRecentMessages(channelId, 30);

      // If not enough context, use a simple response
      if (recentMessages.length < 5) {
        await message.reply('Hey! I need more conversation history in this channel to give you a relevant response. Keep chatting and try again in a bit!');
        return;
      }

      // Generate a response using YOUR trained model (this takes time)
      const response = await rustMLService.generateMentionResponse(recentMessages, message.content, message.author.username);

      console.log(response)

      // Validate response is not empty
      if (!response || response.trim().length === 0) {
        console.warn('⚠️  Model generated empty response, using fallback');
        await message.reply('Hmm, I\'m not sure what to say right now. My model might need retraining with more data!');
        return;
      }

      await message.reply(response);
      console.log(`✅ Replied to mention from ${message.author.username}`);

    } catch (error) {
      const channelName = message.channel.isDMBased() ? 'DM' : (message.channel as TextChannel).name;
      console.error(`Failed to handle mention in #${channelName}:`, error);

      // Provide helpful error message based on error type
      let errorMessage = 'Sorry, I encountered an error trying to respond.';

      if (error instanceof Error) {
        if (error.message.includes('Generation failed')) {
          errorMessage = 'I tried to respond but couldn\'t generate a meaningful reply. My model might need more training data overall - try `!llmtrain now` after collecting more messages across all channels.';
        } else if (error.message.includes('Model not found')) {
          errorMessage = 'My AI model hasn\'t been trained yet. Ask an admin to run `!llmtrain now` first!';
        }
      }

      await message.reply(errorMessage).catch(() => {});
    }
  }

  private async collectMessage(message: Message): Promise<void> {
    // Skip bot messages and empty messages
    if (message.author.bot || !message.content || message.content.trim() === '') {
      return;
    }

    // Skip command messages
    if (message.content.startsWith('!')) {
      return;
    }

    // Skip messages from excluded channels
    if (trainingConfig.isChannelExcluded(message.channel.id)) {
      return;
    }

    try {
      // Store in SQLite
      messageStorageService.storeMessage(message);

      // Store in vector database (async, don't wait)
      const storedMessage = {
        id: message.id,
        channelId: message.channel.id,
        channelName: message.channel.isDMBased() ? 'DM' : (message.channel as TextChannel).name || 'Unknown',
        authorId: message.author.id,
        authorName: message.author.username,
        content: message.content,
        timestamp: message.createdTimestamp,
        hasAttachments: message.attachments.size > 0,
        replyToId: message.reference?.messageId
      };

      vectorStoreService.storeMessage(storedMessage).catch(err => {
        console.error('Failed to store message in vector store:', err);
      });

      // Increment training counter
      fineTuningService.incrementMessageCount();
    } catch (error) {
      console.error('Failed to collect message:', error);
    }
  }

  private startScheduledGeneration(): void {
    console.log('📅 Starting scheduled message generation...');
    console.log(`   Interval: ${this.config.minIntervalMinutes}-${this.config.maxIntervalMinutes} minutes`);
    console.log(`   Active hours: 08:00 - 24:00 (German time)`);
    console.log(`   Activity requirement: Last message < 5 minutes ago`);

    // Schedule to run every minute, but use internal logic to randomize timing
    this.scheduledTask = cron.schedule('* * * * *', async () => {
      await this.tryGenerateMessage();
    });
  }

  private async tryGenerateMessage(): Promise<void> {
    if (!this.isInitialized || !this.client) {
      return;
    }

    // Check if current time is within allowed hours (8:00 - 24:00 German time)
    const germanTime = new Date().toLocaleString('en-US', { timeZone: 'Europe/Berlin' });
    const germanDate = new Date(germanTime);
    const currentHour = germanDate.getHours();

    if (currentHour < 8 || currentHour >= 24) {
      // Outside allowed hours, skip generation silently
      return;
    }

    const now = Date.now();
    const timeSinceLastGeneration = (now - this.lastGenerationTime) / 1000 / 60; // minutes

    // Random interval between min and max
    const targetInterval = Math.random() *
      (this.config.maxIntervalMinutes - this.config.minIntervalMinutes) +
      this.config.minIntervalMinutes;

    if (timeSinceLastGeneration < targetInterval) {
      return;
    }

    try {
      // Get active channels
      const activeChannels = messageStorageService.getActiveChannels();

      if (activeChannels.length === 0) {
        console.log('📝 No messages collected yet. Waiting for more data...');
        return;
      }

      // Filter channels based on config
      let eligibleChannels = activeChannels;

      // Filter out excluded channels from trainingConfig
      eligibleChannels = eligibleChannels.filter(ch =>
        !trainingConfig.isChannelExcluded(ch.channelId)
      );

      if (this.config.channelIds && this.config.channelIds.length > 0) {
        eligibleChannels = eligibleChannels.filter(ch =>
          this.config.channelIds!.includes(ch.channelId)
        );
      }

      if (this.config.excludeChannelIds && this.config.excludeChannelIds.length > 0) {
        eligibleChannels = eligibleChannels.filter(ch =>
          !this.config.excludeChannelIds!.includes(ch.channelId)
        );
      }

      if (eligibleChannels.length === 0) {
        console.log('📝 No eligible channels for message generation');
        return;
      }

      // Pick a random channel
      const targetChannel = eligibleChannels[Math.floor(Math.random() * eligibleChannels.length)];

      // Check if the channel has recent activity (last message < 5 minutes ago)
      // @ts-ignore
        const recentMessages = messageStorageService.getRecentMessages(targetChannel.channelId, 1);
      if (recentMessages.length === 0) {
        // @ts-ignore
          console.log(`📝 No messages in #${targetChannel.channelName}, skipping`);
        return;
      }

      // @ts-ignore
        const lastMessageAge = (now - recentMessages[0].timestamp) / 1000 / 60; // minutes
      if (lastMessageAge > 5) {
        // @ts-ignore
          console.log(`📝 #${targetChannel.channelName} is quiet (last message ${Math.round(lastMessageAge)} min ago), skipping`);
        return;
      }

      // Generate and send message
        // @ts-ignore
      await this.generateAndSendMessage(targetChannel.channelId, targetChannel.channelName);

      this.lastGenerationTime = now;

    } catch (error) {
      console.error('Failed to generate scheduled message:', error);
    }
  }

  private async generateAndSendMessage(channelId: string, channelName: string): Promise<void> {
    if (!this.client) {
      return;
    }

    try {
      // Get recent messages for context (increased to 100 for better context)
      const recentMessages = messageStorageService.getRecentMessages(channelId, 100);

      if (recentMessages.length < 10) {
        console.log(`📝 Not enough messages in #${channelName} to generate (need at least 10)`);
        return;
      }

      // Generate message using LLM with RAG (passes channelId for vector search)
      console.log(`🤖 Generating message for #${channelName} (using Rust ML)...`);
      const generatedContent = await rustMLService.generateMessage(recentMessages, channelName, channelId);

      // Validate response is not empty
      if (!generatedContent || generatedContent.trim().length === 0) {
        console.warn(`⚠️  Model generated empty response for #${channelName}, skipping message`);
        return;
      }

      // Get the channel and send message
      const channel = await this.client.channels.fetch(channelId);

      if (channel && channel.isTextBased()) {
        await (channel as TextChannel).send(generatedContent);
        console.log(`✅ Message sent to #${channelName}: "${generatedContent.substring(0, 50)}..."`);
      }

    } catch (error) {
      console.error(`Failed to generate message for #${channelName}:`, error);
    }
  }

  private startChannelScanning(): void {
    console.log('📅 Starting scheduled channel scanning...');
    console.log(`   Scan interval: ${this.scanIntervalMinutes} minutes`);

    // Run every minute and check if it's time to scan
    this.scanTask = cron.schedule('* * * * *', async () => {
      const now = Date.now();
      const timeSinceLastScan = (now - this.lastScanTime) / 1000 / 60; // minutes

      if (timeSinceLastScan >= this.scanIntervalMinutes) {
        if (this.isScanning) {
          console.log('⏭️  Skipping scheduled scan - scan already in progress');
          return;
        }
        console.log('🔍 Starting scheduled channel scan...');
        await this.scanAllChannels();
        this.lastScanTime = now;
      }
    });
  }

  private async fetchMessagesWithRetry(
    textChannel: TextChannel,
    options: { limit: number; before?: string },
    maxRetries: number = 3
  ): Promise<any> {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        // Increase timeout to 30 seconds for slow connections
        const messages = await textChannel.messages.fetch({
          ...options,
          // @ts-ignore - time property exists but not in types
          time: 30000
        });
        return messages;
      } catch (error: any) {
        lastError = error;
        console.log(`  ⚠️  Fetch attempt ${attempt}/${maxRetries} failed: ${error.message}`);

        if (attempt < maxRetries) {
          // Exponential backoff: 2s, 4s, 8s
          const backoffMs = Math.pow(2, attempt) * 1000;
          console.log(`  ⏳ Retrying in ${backoffMs / 1000}s...`);
          await new Promise(resolve => setTimeout(resolve, backoffMs));
        }
      }
    }

    // All retries failed
    throw lastError || new Error('Failed to fetch messages after retries');
  }

  private async scanAllChannels(forceFullScan: boolean = false): Promise<void> {
    if (!this.client || !this.isInitialized) {
      console.log('⚠️  Cannot scan: client not initialized');
      return;
    }

    // Check if already scanning
    if (this.isScanning) {
      console.log('⚠️  Scan already in progress, skipping');
      return;
    }

    // Set scanning flag
    this.isScanning = true;

    try {
      const guilds = this.client.guilds.cache;
      console.log(`📊 Found ${guilds.size} guild(s) to scan`);

      let totalScanned = 0;
      let totalCollected = 0;

      for (const [, guild] of guilds) {
        const channels = guild.channels.cache.filter(ch => ch.isTextBased());
        console.log(`📁 Guild "${guild.name}": ${channels.size} text channels`);

        for (const [, channel] of channels) {
          if (!channel.isTextBased()) continue;

          // Check if channel should be excluded (using trainingConfig)
          if (trainingConfig.isChannelExcluded(channel.id)) {
            console.log(`  ⏭️  Skipping excluded channel #${(channel as TextChannel).name}`);
            continue;
          }

          // Legacy check for config-based exclusions
          if (this.config.excludeChannelIds?.includes(channel.id)) {
            continue;
          }

          // Check if we should only scan specific channels
          if (this.config.channelIds && this.config.channelIds.length > 0) {
            if (!this.config.channelIds.includes(channel.id)) {
              continue;
            }
          }

          try {
            totalScanned++;
            const textChannel = channel as TextChannel;

            // Get last scanned timestamp for this channel (or 0 if forcing full scan)
            const lastScannedTimestamp = forceFullScan ? 0 : messageStorageService.getLastScannedTimestamp(textChannel.id);
            const timeSinceLastScan = lastScannedTimestamp > 0
              ? `${Math.round((Date.now() - lastScannedTimestamp) / 1000 / 60)} min`
              : 'never';

            console.log(`  📝 Scanning #${textChannel.name} (last: ${timeSinceLastScan} ago)${forceFullScan ? ' [FULL SCAN]' : ''}...`);

            // Fetch messages AFTER last scanned timestamp
            let lastMessageId: string | undefined;
            let totalFetched = 0;
            let channelCollected = 0;
            let hasMore = true;
            let reachedOldMessages = false;

            while (hasMore && !reachedOldMessages) {
              const options: { limit: number; before?: string } = { limit: 100 };
              if (lastMessageId) {
                options.before = lastMessageId;
              }

              // Fetch with retry logic and increased timeout
              const messages = await this.fetchMessagesWithRetry(textChannel, options);
              totalFetched += messages.size;

              if (messages.size === 0) {
                hasMore = false;
                break;
              }

              // Process messages in this batch
              for (const [, msg] of messages) {
                // Stop if we've reached messages older than last scan
                if (msg.createdTimestamp <= lastScannedTimestamp) {
                  reachedOldMessages = true;
                  break;
                }

                if (!msg.author.bot && msg.content && msg.content.trim() !== '' && !msg.content.startsWith('!')) {
                  await this.collectMessage(msg);
                  totalCollected++;
                  channelCollected++;
                }
              }

              // Get the last message ID for pagination
              lastMessageId = messages.last()?.id;

              // If we fetched fewer than 100 messages, we've reached the end
              if (messages.size < 100) {
                hasMore = false;
              }

              // Small delay between batches to avoid rate limiting
              await new Promise(resolve => setTimeout(resolve, 500));

              if (!reachedOldMessages) {
                console.log(`     Progress: ${totalFetched} fetched, ${channelCollected} new`);
              }
            }

            if (reachedOldMessages) {
              console.log(`     ⏭️  Stopped at previously scanned messages (${totalFetched} checked, ${channelCollected} new)`);
            } else {
              console.log(`     ✅ Reached channel start (${totalFetched} total, ${channelCollected} new)`);
            }

            if (channelCollected > 0) {
              console.log(`     ✅ Collected ${channelCollected} new messages from #${textChannel.name}`);
            } else {
              console.log(`     ℹ️  No new messages in #${textChannel.name}`);
            }

          } catch (error) {
            console.error(`     ❌ Failed to scan channel #${(channel as TextChannel).name}:`, error);
          }
        }
      }

      console.log(`✅ Channel scan complete: ${totalScanned} channels scanned, ${totalCollected} new messages collected`);

    } catch (error) {
      console.error('Failed to scan channels:', error);
    } finally {
      // Always clear the scanning flag, even if there was an error
      this.isScanning = false;
    }
  }

  private async configureScan(message: Message, args: string[]): Promise<void> {
    if (args.length === 0) {
      // Show current config
      const statusText = `
**🔍 Channel Scanning Configuration**

• Scan interval: ${this.scanIntervalMinutes} minutes
• Last scan: ${this.lastScanTime > 0 ? `${Math.round((Date.now() - this.lastScanTime) / 60000)} minutes ago` : 'Never'}
• Status: ${this.isScanning ? '🔄 Scan in progress...' : '✅ Idle'}

**Usage:**
\`!llmscan <minutes>\` - Set scan interval (1-60 minutes)
\`!llmscan now\` - Scan only new messages
\`!llmscan full\` - FULL scan (ignore timestamps, scan ALL messages)
      `;
      await message.reply(statusText);
      return;
    }

    const arg = args[0]?.toLowerCase() || '';

    // Trigger immediate scan
    if (arg === 'now' || arg === 'full' || arg === 'force') {
      if (this.isScanning) {
        await message.reply('⚠️  A scan is already in progress. Please wait for it to complete.');
        return;
      }

      const forceFullScan = (arg === 'full' || arg === 'force');

      if (forceFullScan) {
        await message.reply('🔍 Starting FULL channel scan (ignoring timestamps)...');
      } else {
        await message.reply('🔍 Starting channel scan...');
      }

      const startTime = Date.now();
      await this.scanAllChannels(forceFullScan);
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      await message.reply(`✅ Scan completed in ${duration}s`);
      return;
    }

    // Set scan interval
    const newInterval = parseInt(arg);
    if (isNaN(newInterval) || newInterval < 1 || newInterval > 60) {
      await message.reply('❌ Invalid interval. Please specify a number between 1 and 60 minutes.');
      return;
    }

    this.scanIntervalMinutes = newInterval;
    await message.reply(`✅ Scan interval set to ${newInterval} minutes`);
    console.log(`🔍 Scan interval updated to ${newInterval} minutes`);
  }

  private async showStats(message: Message): Promise<void> {
    try {
      const totalMessages = messageStorageService.getTotalMessageCount();
      const vectorCount = await vectorStoreService.getCollectionCount();
      const activeChannels = messageStorageService.getActiveChannels();
      const llmConfig = rustMLService.getConfig();
      const trainingStatus = fineTuningService.getTrainingStatus();

      // Get current German time
      const germanTime = new Date().toLocaleString('en-US', { timeZone: 'Europe/Berlin' });
      const germanDate = new Date(germanTime);
      const currentHour = germanDate.getHours();
      const isActiveHours = currentHour >= 8 && currentHour < 24;

      // Filter out excluded channels from stats
      const filteredChannels = activeChannels.filter(ch =>
        !trainingConfig.isChannelExcluded(ch.channelId)
      );

      const channelList = filteredChannels
        .slice(0, 10)
        .map(ch => `• #${ch.channelName}: ${ch.count} messages`)
        .join('\n');

      const statsText = `
**🤖 LLM Statistics**

**Message Collection:**
• Total messages stored: ${totalMessages}
• Vector embeddings: ${vectorCount}
• Active channels: ${activeChannels.length}

**Top Channels:**
${channelList}

**LLM Configuration:**
• Model: ${fineTuningService.getCurrentModelName()}
• Temperature: ${llmConfig.temperature}
• Max tokens: ${llmConfig.maxTokens}
• Context window: ${llmConfig.contextWindow}

**Auto-generation:**
• Status: ${this.config.enabled ? '✅ Enabled' : '❌ Disabled'}
• Active hours: 08:00 - 24:00 (German time)
• Current time: ${germanDate.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' })} ${isActiveHours ? '✅' : '🌙 (outside active hours)'}
• Activity requirement: Last message < 5 minutes ago
• Interval: ${this.config.minIntervalMinutes}-${this.config.maxIntervalMinutes} minutes
• Last generation: ${this.lastGenerationTime > 0 ? `${Math.round((Date.now() - this.lastGenerationTime) / 60000)} minutes ago` : 'Never'}

**Channel Scanning:**
• Scan interval: ${this.scanIntervalMinutes} minutes
• Last scan: ${this.lastScanTime > 0 ? `${Math.round((Date.now() - this.lastScanTime) / 60000)} minutes ago` : 'Never'}
• Status: ${this.isScanning ? '🔄 Scan in progress...' : '✅ Idle'}

**Training Status:**
• Training: ${trainingStatus.isTraining ? '🔄 In progress...' : '✅ Idle'}
      `;

      await message.reply(statsText);
    } catch (error) {
      console.error('Failed to show stats:', error);
      await message.reply('❌ Failed to retrieve statistics');
    }
  }

  private async manageTrain(message: Message, args: string[]): Promise<void> {
    if (args.length === 0 || args[0] === 'status') {
      // Show training status for remote architecture
      const status = fineTuningService.getTrainingStatus();
      const stats = await fineTuningService.getTrainingStats();
      // Using local Rust ML instead of remote training API

      let statusText = `**🦀 Rust ML Training Status**\n\n`;

      statusText += `• Current model: ${fineTuningService.getCurrentModelName()}\n`;
      statusText += `• Total messages: ${stats.totalMessages}\n`;
      statusText += `• New messages: ${stats.newMessages}\n`;
      statusText += `• Model version: v${stats.modelVersion}\n`;
      statusText += `• Training active: ${status.isTraining ? '🔄 Yes (local)' : '💤 No'}\n`;

      if (stats.lastTrainDate) {
        const lastTrainDate = new Date(stats.lastTrainDate).toLocaleDateString();
        statusText += `• Last training: ${lastTrainDate}\n`;
      }

      // If training is in progress, show detailed progress
      if (status.isTraining) {
        const progressBar = this.createProgressBar(
          status.totalSteps > 0 ? (status.currentStep / status.totalSteps) * 100 : 0,
          20
        );

        statusText += `\n**📊 Active Training:**\n`;
        statusText += `• Phase: ${this.getPhaseEmoji(status.phase)} ${status.phase}\n`;
        statusText += `• Step: ${status.currentStep}/${status.totalSteps}\n`;
        statusText += `• Epoch: ${status.currentEpoch}/${status.totalEpochs}\n`;
        if (status.currentLoss > 0) {
          statusText += `• Loss: ${status.currentLoss.toFixed(4)}\n`;
        }
        statusText += `• Progress: ${progressBar}\n`;
        statusText += `• Started: ${new Date(status.startTime).toLocaleTimeString()}\n`;
      }

      statusText += `\n**💡 Commands:**\n`;
      statusText += `• \`!llmtrain now\` - Start training\n`;
      statusText += `• \`!llmtrain status\` - Check status\n`;
      statusText += `• \`!llmstats\` - View statistics\n`;

      await message.reply(statusText);
      return;
    }

    // @ts-ignore
      const command = args[0].toLowerCase();

    if (command === 'now') {
      // Training is now handled by local Rust ML
      const stats = await fineTuningService.getTrainingStats();

      if (stats.totalMessages < 500) {
        await message.reply(
          `⚠️ **Not enough messages for training**\n\n` +
          `Currently have ${stats.totalMessages} messages, need at least 500.\n\n` +
          `💡 Collect more messages in Discord channels and try training again.`
        );
        return;
      }

      // Actually start the training
      try {
        const result = await rustMLService.startTraining();
        if (result.success) {
          await message.reply(
            '🦀 **Rust ML Training Started Successfully!**\n\n' +
            `📊 **Training Data:**\n` +
            `• Total messages: ${stats.totalMessages}\n` +
            `• New messages: ${stats.newMessages}\n` +
            `• Model version: v${stats.modelVersion}\n\n` +
            `⚡ **Training Info:**\n` +
            `• Method: CPU-based local training\n` +
            `• Duration: Variable (depending on data size)\n` +
            `• Epochs: 10 (default)\n\n` +
            `💡 **Check Progress:**\n` +
            `• Use \`!llmtrain status\` to monitor training\n` +
            `• Training logs will show progress\n` +
            `• Bot will use new model after completion`
          );
        } else {
          await message.reply(
            '❌ **Training Failed to Start**\n\n' +
            `Error: ${result.error || 'Unknown error occurred'}\n\n` +
            `💡 Check that the Rust ML module is properly compiled.\n` +
            `📊 Data: ${stats.totalMessages} messages available for training`
          );
        }
      } catch (error) {
        await message.reply('❌ Failed to start Rust ML training. Check bot logs for details.');
        console.error('Failed to start Rust ML training:', error);
      }
    } else if (command === 'force') {
      // Force immediate training using Rust ML
      const stats = await fineTuningService.getTrainingStats();
      // Start Rust ML training immediately
      try {
        const result = await rustMLService.startTraining();
        if (result.success) {
          await message.reply(
            '🦀 **Rust ML Training Started!**\n\n' +
            `📊 **Training Data:** ${stats.totalMessages} messages\n` +
            `⚡ **Method:** Local CPU training\n\n` +
            `💡 Use \`!llmtrain status\` to monitor progress.`)
        } else {
          await message.reply(
            '❌ **Training Failed to Start**\n\n' +
            `Error: ${result.error || 'Unknown error occurred'}\n\n` +
            `💡 Check that the Rust ML module is properly compiled.`
          );
        }

      } catch (error) {
        await message.reply('❌ Failed to start Rust ML training. Check bot logs for details.');
        console.error('Failed to start Rust ML training:', error);
      }
    } else {
      await message.reply('❌ Unknown command. Use `!llmtrain status` for training information.\n\n💡 Available commands:\n• `!llmtrain status` - Show training status\n• `!llmtrain now` - Start training\n• `!llmtrain force` - Force immediate training\n\n🦀 Training is handled by local Rust ML.');
    }
  }

  private createProgressBar(percent: number, width: number): string {
    const filled = Math.round((percent / 100) * width);
    const empty = width - filled;
    return '█'.repeat(filled) + '░'.repeat(empty);
  }

  private getPhaseEmoji(phase: string): string {
    const emojis: Record<string, string> = {
      'idle': '💤',
      'preparing': '📝',
      'training': '🔥',
      'saving': '💾',
      'importing': '📦'
    };
    return emojis[phase] || '❓';
  }

  private async clearVectorStore(message: Message, args: string[]): Promise<void> {
    await message.reply('🗑️ Clearing vector store and rebuilding with new embeddings...');

    try {
      await vectorStoreService.clear();
        await message.reply(`✅ Vector store rebuilt with TF-IDF embeddings!\n`);
    } catch (error) {
      console.error('Failed to clear vector store:', error);
      await message.reply('❌ Failed to clear vector store. Check console for details.');
    }
  }

  private async embedVectorStore(message: Message, args: string[]) : Promise<void> {
      const totalMessages = messageStorageService.getTotalMessageCount();

      // Get all active channels and re-embed their messages
      const channels = messageStorageService.getActiveChannels();
      let embeddedCount = 0;

      for (const channel of channels) {
          // Skip excluded channels
          if (trainingConfig.isChannelExcluded(channel.channelId)) {
              console.log(`  ⏭️  Skipping excluded channel #${channel.channelName}...`);
              continue;
          }

          const messages = messageStorageService.getMessagesByChannel(channel.channelId, 100000);
          console.log(`  📝 Re-embedding ${messages.length} messages from #${channel.channelName}...`);

          for (const msg of messages) {
              await vectorStoreService.storeMessage(msg);
              embeddedCount++;

              // Progress update every 100 messages
              if (embeddedCount % 100 === 0) {
                  console.log(`     Progress: ${embeddedCount}/${totalMessages} embedded...`);
              }
          }
      }

      const vectorCount = await vectorStoreService.getCollectionCount();
      console.log(`✅ Re-embedding complete! ${embeddedCount} messages processed, ${vectorCount} stored in vector DB.`);
      await message.reply(`✅ Vector store rebuilt with TF-IDF embeddings!\n📊 Stored ${vectorCount} message embeddings (${embeddedCount} processed).`);
  }

  private async manageExclusions(message: Message, args: string[]): Promise<void> {
    const action = args[0]?.toLowerCase();
    const channelId = args[1];

    if (!action || !['add', 'remove', 'list'].includes(action)) {
      await message.reply('Usage: `!llmexclude [add|remove|list] [channelId]`\nExample: `!llmexclude add 1234567890`');
      return;
    }

    if (action === 'list') {
      const excluded = trainingConfig.getExcludedChannels();
      if (excluded.length === 0) {
        await message.reply('📋 No channels are excluded from training.\n\n💡 The bot learns from ALL servers you invite it to!');
        return;
      }

      let response = '📋 **Channels excluded from training:**\n\n';
      for (const id of excluded) {
        const channel = await this.client?.channels.fetch(id).catch(() => null);
        const channelName = channel && 'name' in channel ? `#${channel.name}` : 'Unknown';
        response += `• ${channelName} (${id})\n`;
      }
      response += '\n💡 The bot learns from ALL servers you invite it to (except these channels)!';
      await message.reply(response);
      return;
    }

    if (!channelId) {
      await message.reply('❌ Please provide a channel ID.\nExample: `!llmexclude add 1234567890`\nTip: Right-click a channel → Copy Channel ID');
      return;
    }

    if (action === 'add') {
      trainingConfig.addExcludedChannel(channelId);
      const channel = await this.client?.channels.fetch(channelId).catch(() => null);
      const channelName = channel && 'name' in channel ? `#${channel.name}` : channelId;
      await message.reply(`✅ Added ${channelName} to training exclusion list.\n\n⚠️  This only affects FUTURE training. Run \`!llmtrain now\` to retrain without this channel.`);
    } else if (action === 'remove') {
      trainingConfig.removeExcludedChannel(channelId);
      const channel = await this.client?.channels.fetch(channelId).catch(() => null);
      const channelName = channel && 'name' in channel ? `#${channel.name}` : channelId;
      await message.reply(`✅ Removed ${channelName} from training exclusion list.\n\n💡 Run \`!llmtrain now\` to include this channel in training.`);
    }
  }

  private async showTrainingConfig(message: Message): Promise<void> {
    const config = trainingConfig.getStatusInfo();
    const channels = messageStorageService.getActiveChannels();
    const totalMessages = messageStorageService.getTotalMessageCount();
    const trainingStatus = fineTuningService.getTrainingStatus();

    // Count messages from different servers
    const serverCounts = new Map<string, number>();
    for (const channel of channels) {
      const fetchedChannel = await this.client?.channels.fetch(channel.channelId).catch(() => null);
      if (fetchedChannel && 'guild' in fetchedChannel && fetchedChannel.guild) {
        const serverName = fetchedChannel.guild.name;
        serverCounts.set(serverName, (serverCounts.get(serverName) || 0) + channel.count);
      }
    }

    let response = '⚙️ **Training Configuration**\n\n';
    response += `🤖 **Bot User ID:** ${config.botUserId || 'Not set yet'}\n`;
    response += `🌍 **Multi-Server Support:** ✅ Enabled (learns from ALL servers)\n`;
    response += `📊 **Total Messages:** ${totalMessages.toLocaleString()}\n`;
    response += `📁 **Active Channels:** ${channels.length}\n\n`;

    if (serverCounts.size > 0) {
      response += `🏠 **Messages per Server:**\n`;
      for (const [server, count] of serverCounts) {
        response += `• ${server}: ${count.toLocaleString()} messages\n`;
      }
      response += '\n';
    }

    if (config.excludedChannels.length > 0) {
      response += `🚫 **Excluded Channels:** ${config.excludedChannels.length}\n`;
      for (const id of config.excludedChannels) {
        const channel = await this.client?.channels.fetch(id).catch(() => null);
        const channelName = channel && 'name' in channel ? `#${channel.name}` : 'Unknown';
        response += `• ${channelName} (${id})\n`;
      }
      response += '\n';
    } else {
      response += `🚫 **Excluded Channels:** None\n\n`;
    }

    response += `🎓 **Training Filters:**\n`;
    response += `• Bot's own messages: ❌ Not included\n`;
    response += `• Other bots' messages: ❌ Not included\n`;
    response += `• User messages: ✅ Included (all servers)\n\n`;

    response += `📈 **Training Status:**\n`;
    response += `• Currently training: ${trainingStatus.isTraining ? '✅ Yes' : '❌ No'}\n`;
    response += `• Messages since last train: ${trainingStatus.messagesSinceLastTrain.toLocaleString()}\n\n`;

    response += `💡 **Tip:** Use \`!llmexclude add <channelId>\` to exclude channels from training.`;

    await message.reply(response);
  }

  /**
   * Handle local Rust ML commands
   */
  private async manageRustML(message: Message, args: string[]): Promise<void> {
    if (args.length === 0 || args[0] === 'status') {
      await this.showRustMLStatus(message);
      return;
    }

    // @ts-ignore
      const command = args[0].toLowerCase();

    if (command === 'logs') {
      await this.showRustMLLogs(message);
    } else {
      await message.reply(
        '❌ Unknown command.\n\n💡 Available commands:\n' +
        '• `!llmrust status` - Show Rust ML status\n' +
        '• `!llmrust logs` - Show training logs'
      );
    }
  }

  /**
   * Show Rust ML system status
   */
  private async showRustMLStatus(message: Message): Promise<void> {
    const rustInitialized = await rustMLService.initialize();
    const modelExists = await rustMLService.checkModelExists();
    const config = rustMLService.getConfig();

    let response = '**🦀 Rust ML System Status**\n\n';

    if (rustInitialized) {
      response += '✅ **Status:** Rust module loaded\n';
      response += `📦 **Model:** ${modelExists ? '✅ Trained model found' : '⚠️ No trained model'}\n`;
      response += `🔧 **Backend:** ${config.backend}\n`;
      response += `🌡️ **Temperature:** ${config.temperature}\n`;
      response += `📏 **Max Tokens:** ${config.maxTokens}\n`;
      response += `🪟 **Context Window:** ${config.contextWindow}\n\n`;

      if (modelExists) {
        response += '💡 **Ready to generate responses using trained model**\n';
      } else {
        response += '💡 **Use `!llmtrain now` to train your first model**\n';
      }
    } else {
      response += '❌ **Status:** Rust module not available\n';
      response += '🔧 **Backend:** Fallback mode\n\n';
      response += '💡 **To enable Rust ML:**\n';
      response += '• Compile the Rust module: `cd rust-ml && cargo build --release`\n';
      response += '• Restart the bot\n';
    }

    await message.reply(response);
  }

  /**
   * Show Rust ML training logs
   */
  private async showRustMLLogs(message: Message): Promise<void> {
    await message.reply(
      '📋 **Rust ML Training Logs**\n\n' +
      'Local training logs are output to the console.\n\n' +
      '💡 Check the bot\'s console output for training progress and errors.'
    );
  }

  async cleanup(): Promise<void> {
    if (this.scheduledTask) {
      this.scheduledTask.stop();
    }
    if (this.scanTask) {
      this.scanTask.stop();
    }
    console.log('🤖 LLM Plugin cleaned up');
  }
}
