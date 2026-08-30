import { Client, Message } from "discord.js";
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
    const baseModel = process.env.LLM_BASE_MODEL || 'phi3:mini';
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
  ];

  async initialize(client: Client, bot: ExtensibleBot): Promise<void> {
    this.client = client;
    console.log("LLM plugin initializing...");

    try {
      await this.llmService.initializeModel();
      console.log("LLM plugin initialized successfully");

      // Register message handler for all messages (not just commands)
      client.on("messageCreate", async (message) => {
        await this.handleAllMessages(message);
      });

      // Wait for the client to be ready, then scan historical messages
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
    // Ignore bot messages
    if (message.author.bot) return;

    // Only handle guild messages
    if (!message.guild) return;

    // Store message in database
    this.messageDb.storeMessage(
      message.id,
      message.guild.id,
      message.channelId,
      message.author.id,
      message.author.username,
      message.content,
      message.createdTimestamp
    );

    // Determine if we should respond
    const shouldRespond = this.shouldRespondToMessage(message);

    if (shouldRespond && !this.isTraining) {
      await this.generateAndSendResponse(message);
    }
  }

  private shouldRespondToMessage(message: Message): boolean {
    if (!this.client?.user) return false;

    // Always respond if bot is mentioned
    if (message.mentions.has(this.client.user.id)) {
      return true;
    }

    // Always respond if replying to bot
    if (message.reference?.messageId) {
      // Check if referenced message is from bot (would need to fetch, simplified here)
      return true;
    }

    // Random chance response
    if (Math.random() < this.randomResponseChance) {
      return true;
    }

    return false;
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

          try {
            console.log(`  Scanning channel: ${channel.name || channelId}`);
            let messagesInChannel = 0;
            let lastMessageId: string | undefined;
            let hasMore = true;

            while (hasMore) {
              try {
                // Fetch messages in batches of 100 (Discord API limit)
                const options: any = { limit: 100 };
                if (lastMessageId) {
                  options.before = lastMessageId;
                }

                const messages = await channel.messages.fetch(options);

                if (messages.size === 0) {
                  hasMore = false;
                  break;
                }

                // Store each message
                for (const [msgId, msg] of messages) {
                  if (msg.author.bot || !msg.content) continue;

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
                }

                // Get the last message ID for pagination
                lastMessageId = messages.last()?.id;

                // If we got fewer than 100 messages, we've reached the end
                if (messages.size < 100) {
                  hasMore = false;
                }

                // Add a small delay to avoid rate limiting
                await new Promise((resolve) => setTimeout(resolve, 1000));
              } catch (channelError) {
                console.error(`    Error fetching messages from channel ${channel.name}:`, channelError);
                hasMore = false;
              }
            }

            console.log(`    Scanned ${messagesInChannel} messages from ${channel.name}`);
            totalChannelsScanned++;
          } catch (channelError) {
            console.error(`  Failed to scan channel ${channel.name}:`, channelError);
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
      // Get recent channel messages for context
      const recentMessages = this.messageDb.getChannelMessages(message.channelId, 20);

      if (recentMessages.length === 0) {
        return; // Need some context
      }

      // Show typing indicator
      if ('sendTyping' in message.channel) {
        await message.channel.sendTyping();
      }

      const botUsername = this.client?.user?.username || "Bot";
      const response = await this.llmService.generateResponse(
        `${message.author.username}: ${message.content}`,
        recentMessages,
        botUsername
      );

      // Send response
      await message.reply(response);
    } catch (error) {
      console.error("Error generating LLM response:", error);
      // Silently fail - don't spam errors in chat
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
      const guildId = message.guild.id;

      const messageCount = this.messageDb.getMessageCount(guildId);

      if (messageCount < 100) {
        await message.reply(
          `⚠️ Not enough messages to train. Current: ${messageCount}, Required: 100+`
        );
        this.isTraining = false;
        return;
      }

      await message.reply(
        `🔄 Starting training with ${messageCount} messages... This may take a few minutes.`
      );

      // Get all messages for training
      const allMessages = this.messageDb.getRecentMessages(guildId, 10000);

      // Fine-tune the model
      await this.llmService.fineTuneModel(allMessages, guildId);

      await message.reply(
        `✅ Training complete! The bot has learned from ${allMessages.length} messages.`
      );
    } catch (error) {
      console.error("Error during retraining:", error);
      await message.reply(
        `❌ Training failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    } finally {
      this.isTraining = false;
    }
  }

  private async showStats(message: Message): Promise<void> {
    if (!message.guild) {
      await message.reply("❌ This command can only be used in a server.");
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
    `;

    await message.reply(statsText);
  }

  async cleanup(): Promise<void> {
    this.messageDb.close();
  }
}