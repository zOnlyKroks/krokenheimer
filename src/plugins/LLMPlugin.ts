import { Client, Message, TextChannel } from 'discord.js';
import type { BotPlugin, BotCommand } from '../types/index.js';
import type { ExtensibleBot } from '../core/Bot.js';
import messageStorageService from '../services/MessageStorageService.js';
import vectorStoreService from '../services/VectorStoreService.js';
import ollamaService from '../services/OllamaService.js';
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
      usage: 'llmscan [interval_minutes|now]',
      execute: this.configureScan.bind(this),
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

    // Check Ollama connection
    console.log('1️⃣  Checking Ollama connection...');
    const ollamaConnected = await ollamaService.checkConnection();
    if (!ollamaConnected) {
      console.error('❌ Ollama is not running. Please start Ollama first: https://ollama.ai');
      console.log('💡 After installing Ollama, run: ollama pull llama3.2:3b');
      return;
    }
    console.log('✅ Ollama is connected');

    // Ensure model exists
    console.log('2️⃣  Checking if model exists...');
    const modelExists = await ollamaService.ensureModelExists();
    if (!modelExists) {
      console.error('❌ Required model not found. Run: ollama pull llama3.2:3b');
      return;
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

      // Show typing indicator
      if ('sendTyping' in message.channel) {
        await message.channel.sendTyping().catch(() => {});
      }

      // Get recent messages for context
      const channelId = message.channel.id;
      const recentMessages = messageStorageService.getRecentMessages(channelId, 30);

      // If not enough context, use a simple response
      if (recentMessages.length < 5) {
        await message.reply('Hey! I need to learn more from this channel before I can have a proper conversation. Give me some time to collect messages!');
        return;
      }

      // Generate a response using the LLM (this takes time)
      const response = await ollamaService.generateMentionResponse(recentMessages, message.content, message.author.username);

      await message.reply(response);
      console.log(`✅ Replied to mention from ${message.author.username}`);

    } catch (error) {
      console.error('Failed to handle mention:', error);
      await message.reply('Sorry, I encountered an error trying to respond. Please try again later!').catch(() => {});
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

    } catch (error) {
      console.error('Failed to collect message:', error);
    }
  }

  private startScheduledGeneration(): void {
    console.log('📅 Starting scheduled message generation...');
    console.log(`   Interval: ${this.config.minIntervalMinutes}-${this.config.maxIntervalMinutes} minutes`);
    console.log(`   Active hours: 08:00 - 24:00 (German time)`);

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
      // Get recent messages for context
      const recentMessages = messageStorageService.getRecentMessages(channelId, 50);

      if (recentMessages.length < 10) {
        console.log(`📝 Not enough messages in #${channelName} to generate (need at least 10)`);
        return;
      }

      // Generate message using LLM
      console.log(`🤖 Generating message for #${channelName}...`);
      const generatedContent = await ollamaService.generateMessage(recentMessages, channelName);

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

  private async scanAllChannels(): Promise<void> {
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

          // Check if channel should be excluded
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

            console.log(`  📝 Scanning #${textChannel.name}...`);

            // Fetch ALL messages using pagination
            let lastMessageId: string | undefined;
            let totalFetched = 0;
            let channelCollected = 0;
            let hasMore = true;

            while (hasMore) {
              const options: { limit: number; before?: string } = { limit: 100 };
              if (lastMessageId) {
                options.before = lastMessageId;
              }

              const messages = await textChannel.messages.fetch(options);
              totalFetched += messages.size;

              if (messages.size === 0) {
                hasMore = false;
                break;
              }

              // Process messages in this batch
              for (const [, msg] of messages) {
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
              console.log(`     More to go, current: ${totalFetched}`)
            }

            console.log(`     Fetched ${totalFetched} total messages (${channelCollected} new, rest were duplicates or filtered)`);

            if (channelCollected > 0) {
              console.log(`     ✅ Collected ${channelCollected} new messages from #${textChannel.name}`);
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
\`!llmscan now\` - Trigger immediate scan
      `;
      await message.reply(statusText);
      return;
    }

    const arg = args[0]?.toLowerCase() || '';

    // Trigger immediate scan
    if (arg === 'now') {
      if (this.isScanning) {
        await message.reply('⚠️  A scan is already in progress. Please wait for it to complete.');
        return;
      }

      await message.reply('🔍 Starting channel scan...');
      const startTime = Date.now();
      await this.scanAllChannels();
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
      const llmConfig = ollamaService.getConfig();

      // Get current German time
      const germanTime = new Date().toLocaleString('en-US', { timeZone: 'Europe/Berlin' });
      const germanDate = new Date(germanTime);
      const currentHour = germanDate.getHours();
      const isActiveHours = currentHour >= 8 && currentHour < 24;

      const channelList = activeChannels
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
• Model: ${llmConfig.model}
• Temperature: ${llmConfig.temperature}
• Max tokens: ${llmConfig.maxTokens}

**Auto-generation:**
• Status: ${this.config.enabled ? '✅ Enabled' : '❌ Disabled'}
• Active hours: 08:00 - 24:00 (German time)
• Current time: ${germanDate.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' })} ${isActiveHours ? '✅' : '🌙 (outside active hours)'}
• Interval: ${this.config.minIntervalMinutes}-${this.config.maxIntervalMinutes} minutes
• Last generation: ${this.lastGenerationTime > 0 ? `${Math.round((Date.now() - this.lastGenerationTime) / 60000)} minutes ago` : 'Never'}

**Channel Scanning:**
• Scan interval: ${this.scanIntervalMinutes} minutes
• Last scan: ${this.lastScanTime > 0 ? `${Math.round((Date.now() - this.lastScanTime) / 60000)} minutes ago` : 'Never'}
• Status: ${this.isScanning ? '🔄 Scan in progress...' : '✅ Idle'}
      `;

      await message.reply(statsText);
    } catch (error) {
      console.error('Failed to show stats:', error);
      await message.reply('❌ Failed to retrieve statistics');
    }
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
