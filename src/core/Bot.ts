import { Client, GatewayIntentBits, Message } from "discord.js";
import type { BotPlugin, BotCommand, BotConfig } from "../types/index.ts";
import { Logger } from "./util/logger.js";
import trainingConfig from "../config/trainingConfig.js";

export class ExtensibleBot {
  private client: Client;
  private plugins: Map<string, BotPlugin> = new Map();
  private commands: Map<string, BotCommand> = new Map();
  private cooldowns: Map<string, Map<string, number>> = new Map();
  private logger: Logger;
  private config: BotConfig;

  constructor(config: BotConfig) {
    this.config = config;
    this.logger = new Logger();
    this.client = new Client({
      intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
      ],
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.client.once("ready", () => {
      this.logger.info(`✅ Bot logged in as ${this.client.user?.tag}`);

      // Set bot user ID for training filter
      if (this.client.user) {
        trainingConfig.setBotUserId(this.client.user.id);
      }
    });

    this.client.on("messageCreate", async (message) => {
      await this.handleMessage(message);
    });

    process.on("SIGINT", async () => {
      await this.shutdown();
    });
  }

  private async handleMessage(message: Message): Promise<void> {
    if (message.author.bot || !message.content.startsWith(this.config.prefix)) {
      return;
    }

    const args = message.content.slice(this.config.prefix.length).trim().split(/ +/);
    const commandName = args.shift()?.toLowerCase();

    if (!commandName) return;

    const command = this.commands.get(commandName) || 
      [...this.commands.values()].find(cmd => cmd.aliases?.includes(commandName));

    if (!command) return;

    if (await this.checkCooldown(message, command)) return;

    if (await this.checkPermissions(message, command)) return;

    try {
      this.logger.info(`Executing command: ${command.name} by ${message.author.tag}`);
      await command.execute(message, args, this.client);
    } catch (error) {
      this.logger.error(`Error executing command ${command.name}:`, error);
      await message.reply("❌ An error occurred while executing the command.");
    }
  }

  private async checkCooldown(message: Message, command: BotCommand): Promise<boolean> {
    if (!command.cooldown || command.cooldown <= 0) return false;

    const now = Date.now();
    const timestamps = this.cooldowns.get(command.name) || new Map();
    const cooldownAmount = command.cooldown * 1000;

    if (timestamps.has(message.author.id)) {
      const expirationTime = timestamps.get(message.author.id)! + cooldownAmount;

      if (now < expirationTime) {
        const timeLeft = (expirationTime - now) / 1000;
        await message.reply(`⏰ Please wait ${timeLeft.toFixed(1)} seconds before using \`${command.name}\` again.`);
        return true;
      }
    }

    timestamps.set(message.author.id, now);
    this.cooldowns.set(command.name, timestamps);

    setTimeout(() => timestamps.delete(message.author.id), cooldownAmount);
    return false;
  }

  private async checkPermissions(message: Message, command: BotCommand): Promise<boolean> {
    if (!command.permissions || command.permissions.length === 0) return false;

    const member = message.member;
    if (!member) return false;

    const hasPermission = command.permissions.every(permission => 
      member.permissions.has(permission as any)
    );

    if (!hasPermission) {
      await message.reply("❌ You don't have permission to use this command.");
      return true;
    }

    return false;
  }

  public async loadPlugin(plugin: BotPlugin): Promise<void> {
    try {
      this.logger.info(`Loading plugin: ${plugin.name} v${plugin.version}`);
      
      if (plugin.initialize) {
        await plugin.initialize(this.client, this);
      }

      for (const command of plugin.commands) {
        this.commands.set(command.name, command);
        this.logger.info(`Registered command: ${command.name}`);
      }

      this.plugins.set(plugin.name, plugin);
      this.logger.info(`✅ Plugin ${plugin.name} loaded successfully`);
    } catch (error) {
      this.logger.error(`Failed to load plugin ${plugin.name}:`, error);
      throw error;
    }
  }

  public async unloadPlugin(pluginName: string): Promise<void> {
    const plugin = this.plugins.get(pluginName);
    if (!plugin) {
      throw new Error(`Plugin ${pluginName} not found`);
    }

    try {
      // Cleanup plugin
      if (plugin.cleanup) {
        await plugin.cleanup();
      }

      // Unregister commands
      for (const command of plugin.commands) {
        this.commands.delete(command.name);
      }

      this.plugins.delete(pluginName);
      this.logger.info(`✅ Plugin ${pluginName} unloaded successfully`);
    } catch (error) {
      this.logger.error(`Failed to unload plugin ${pluginName}:`, error);
      throw error;
    }
  }

  public getLoadedPlugins(): string[] {
    return Array.from(this.plugins.keys());
  }

  public getCommands(): BotCommand[] {
    return Array.from(this.commands.values());
  }

  public async start(): Promise<void> {
    try {
      await this.client.login(this.config.token);
    } catch (error) {
      this.logger.error("Failed to start bot:", error);
      throw error;
    }
  }

  private async shutdown(): Promise<void> {
    this.logger.info("Shutting down bot...");
    
    // Cleanup all plugins
    for (const [name, plugin] of this.plugins) {
      if (plugin.cleanup) {
        try {
          await plugin.cleanup();
          this.logger.info(`Cleaned up plugin: ${name}`);
        } catch (error) {
          this.logger.error(`Error cleaning up plugin ${name}:`, error);
        }
      }
    }

    await this.client.destroy();
    process.exit(0);
  }
}