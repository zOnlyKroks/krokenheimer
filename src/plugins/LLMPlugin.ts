import { Client, Message, Collection, GuildTextBasedChannel } from "discord.js";
import type { BotPlugin, BotCommand } from "../types/index.js";
import type { ExtensibleBot } from "../core/Bot.js";
import { MessageDatabase } from "../services/MessageDatabase.js";
import { LLMService } from "../services/LLMService.js";

export class LLMPlugin implements BotPlugin {
  name = "LLMPlugin";
  description = "LLM integration with learning capabilities";
  version = "1.0.0";

  private messageDb: MessageDatabase;
  private llmService: LLMService;
  private client: Client | null = null;
  private randomResponseChance: number;
  private isTraining: boolean = false;

  constructor() {
    const baseModel = process.env.LLM_BASE_MODEL || 'gemma2:2b';
    const customModelName = process.env.LLM_CUSTOM_MODEL || 'discord-bot-custom';
    const ollamaHost = process.env.OLLAMA_HOST || 'http://localhost:11434';
    const responseChance = parseFloat(process.env.LLM_RANDOM_CHANCE || '0.05'); // 5% default

    this.messageDb = new MessageDatabase();
    this.llmService = new LLMService({
      baseModel,
      customModelName,
      ollamaHost,
      maxContextMessages: 20,
    });
    this.randomResponseChance = responseChance;
  }

  commands: BotCommand[] = [
    {
      name: "retrain",
      description: "Fine-tune the bot with server messages",
      execute: this.retrain.bind(this),
    },
    {
      name: "stats",
      description: "Show message learning statistics",
      aliases: ["llmstats"],
      execute: this.showStats.bind(this),
    },
    {
      name: "scan",
      description: "Scan historical messages from all servers",
      aliases: ["rescan"],
      execute: this.scanCommand.bind(this),
    },
  ];

  async initialize(client: Client, bot: ExtensibleBot): Promise<void> {
    this.client = client;
    console.log("LLM plugin initializing...");

    try {
      await this.llmService.initializeModel();
      console.log("LLM plugin initialized successfully");

      client.on("messageCreate", async (message) => {
        await this.handleAllMessages(message);
      });
      if (client.isReady()) {
        await this.scanAllHistoricalMessages();
      } else {
        client.once("ready", async () => {
          await this.scanAllHistoricalMessages();
        });
      }
    } catch (error) {
      console.error("Failed to initialize LLM plugin:", error);
      throw error;
    }
  }

  private async handleAllMessages(message: Message): Promise<void> {
    if (message.author.bot || !message.guild) return;

    this.messageDb.storeMessage(
      message.id,
      message.guild.id,
      message.channelId,
      message.author.id,
      message.author.username,
      message.content,
      message.createdTimestamp
    );

    const shouldRespond = this.shouldRespondToMessage(message);
    if (shouldRespond && !this.isTraining) {
      await this.generateAndSendResponse(message);
    }
  }

  private shouldRespondToMessage(message: Message): boolean {
    if (!this.client?.user) return false;
    if (message.content.startsWith('!')) return false;
    if (message.mentions.has(this.client.user.id)) return true;
    if (message.content.length < 10 || message.content.startsWith('http')) return false;
    return Math.random() < this.randomResponseChance;
  }

  private async scanAllHistoricalMessages(): Promise<void> {
    if (!this.client) return;

    console.log("🔍 Starting historical message scan...");
    let totalMessagesScanned = 0;
    let totalChannelsScanned = 0;

    try {
      const guilds = this.client.guilds.cache;
      console.log(`Found ${guilds.size} guild(s) to scan`);

      for (const [guildId, guild] of guilds) {
        console.log(`Scanning guild: ${guild.name} (${guildId})`);

        const channels = guild.channels.cache.filter(
          (channel) => channel.isTextBased() && !channel.isDMBased()
        );

        for (const [channelId, channel] of channels) {
          if (!channel.isTextBased()) continue;

          if (!('messages' in channel)) continue;

          const textChannel = channel as GuildTextBasedChannel;

          try {
            console.log(`  Scanning channel: ${textChannel.name || channelId}`);
            let messagesInChannel = 0;
            let lastMessageId: string | undefined;
            let hasMore = true;

            while (hasMore) {
              try {
                const fetchOptions = lastMessageId
                  ? { limit: 100, before: lastMessageId }
                  : { limit: 100 };

                const fetchedMessages = await textChannel.messages.fetch(fetchOptions);

                if (fetchedMessages.size === 0) {
                  hasMore = false;
                  break;
                }

                fetchedMessages.forEach((msg) => {
                  if (msg.author.bot || !msg.content) return;

                  this.messageDb.storeMessage(
                    msg.id,
                    guildId,
                    channelId,
                    msg.author.id,
                    msg.author.username,
                    msg.content,
                    msg.createdTimestamp
                  );

                  messagesInChannel++;
                  totalMessagesScanned++;
                });

                lastMessageId = fetchedMessages.last()?.id;

                if (fetchedMessages.size < 100) {
                  hasMore = false;
                }

                await new Promise((resolve) => setTimeout(resolve, 1000));
              } catch (channelError) {
                console.error(`    Error fetching messages from channel ${textChannel.name}:`, channelError);
                hasMore = false;
              }
            }

            console.log(`    Scanned ${messagesInChannel} messages from ${textChannel.name}`);
            totalChannelsScanned++;
          } catch (channelError) {
            console.error(`  Failed to scan channel ${textChannel.name}:`, channelError);
          }
        }
      }

      console.log(`✅ Historical scan complete! Scanned ${totalMessagesScanned} messages from ${totalChannelsScanned} channels across ${guilds.size} guilds.`);
    } catch (error) {
      console.error("Error during historical message scan:", error);
    }
  }

  private async generateAndSendResponse(message: Message): Promise<void> {
    try {
      const recentMessages = this.messageDb.getChannelMessages(message.channelId, 20);
      if (recentMessages.length === 0) return;

      if ('sendTyping' in message.channel) {
        await message.channel.sendTyping();
      }

      const botUsername = this.client?.user?.username || "Bot";
      const response = await this.llmService.generateResponse(
        `${message.author.username}: ${message.content}`,
        recentMessages,
        botUsername
      );

      await message.reply(response);
    } catch (error) {
      console.error("Error generating LLM response:", error);
    }
  }

  private async retrain(message: Message): Promise<void> {
    if (this.isTraining) {
      await message.reply("⏳ Training is already in progress. Please wait.");
      return;
    }

    if (!message.guild) {
      await message.reply("❌ This command can only be used in a server.");
      return;
    }

    try {
      this.isTraining = true;

      const totalMessages = this.messageDb.getTotalMessageCount();

      if (totalMessages < 100) {
        await message.reply(
          `⚠️ Not enough messages to train. Current: ${totalMessages}, Required: 100+`
        );
        this.isTraining = false;
        return;
      }

      await message.reply(
        `This will take a LONG time (30-60 minutes or more on CPU). Check console for progress!`
      );

      // Import spawn from child_process
      const { spawn } = await import('child_process');
      const path = await import('path');

      // Run the Python training script
      const pythonScript = path.join(process.cwd(), 'scripts', 'train_model.py');

      console.log(`\n${'='.repeat(60)}`);
      console.log('STARTING REAL MODEL TRAINING');
      console.log('This uses actual gradient descent to update model weights');
      console.log('Check console output below for training progress');
      console.log(`${'='.repeat(60)}\n`);

      const pythonProcess = spawn('python3', [pythonScript], {
        cwd: process.cwd(),
        stdio: 'inherit' // Show output in console
      });

      pythonProcess.on('close', async (code) => {
        if (code === 0) {
          await message.reply(
            `The model has actually learned from ${totalMessages} messages through gradient descent.`
          );
        } else {
          await message.reply(
            `Training failed with exit code ${code}. Check console for details.`
          );
        }
        this.isTraining = false;
      });

      pythonProcess.on('error', async (error) => {
        console.error('Failed to start training:', error);
        await message.reply(
          `Failed to start training: ${error.message}\n\nMake sure Python 3 and required packages are installed:\n\`cd scripts && pip install -r requirements.txt\``
        );
        this.isTraining = false;
      });

    } catch (error) {
      console.error("Error during retraining:", error);
      await message.reply(
        `Training failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
      this.isTraining = false;
    }
  }

  private async showStats(message: Message): Promise<void> {
    if (!message.guild) {
      await message.reply("This command can only be used in a server.");
      return;
    }

    const messageCount = this.messageDb.getMessageCount(message.guild.id);
    const randomChancePercent = (this.randomResponseChance * 100).toFixed(1);

    const statsText = `
**📊 LLM Statistics**

**Messages Stored:** ${messageCount}
**Random Response Chance:** ${randomChancePercent}%
**Training Status:** ${this.isTraining ? "🔄 In Progress" : "✅ Ready"}

The bot will respond when:
• Mentioned or replied to
• ${randomChancePercent}% random chance on any message

Use \`!retrain\` to fine-tune with stored messages (min 100 messages)
Use \`!scan\` to scan historical messages from all servers
    `;

    await message.reply(statsText);
  }

  private async scanCommand(message: Message): Promise<void> {
    await message.reply("🔍 Starting manual message scan... This may take a while. Check console for progress.");

    try {
      const startTime = Date.now();
      await this.scanAllHistoricalMessages();
      const duration = Math.round((Date.now() - startTime) / 1000);

      const totalMessages = this.messageDb.getTotalMessageCount();
      const guildMessages = message.guild ? this.messageDb.getMessageCount(message.guild.id) : 0;

      await message.reply(
        `✅ Scan complete! Took ${duration} seconds.\n**Total messages:** ${totalMessages}\n**This server:** ${guildMessages}`
      );
    } catch (error) {
      console.error("Error during manual scan:", error);
      await message.reply(
        `❌ Scan failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }

  async cleanup(): Promise<void> {
    this.messageDb.close();
  }
}