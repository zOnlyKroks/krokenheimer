import { Message, Client } from "discord.js";
import type { ExtensibleBot } from "../core/Bot.ts";

export interface BotCommand {
  name: string;
  description: string;
  aliases?: string[];
  usage?: string;
  cooldown?: number;
  permissions?: string[];
  execute: (message: Message, args: string[], client: Client) => Promise<void>;
}

export interface BotPlugin {
  name: string;
  description: string;
  version: string;
  commands: BotCommand[];
  initialize?: (client: Client, bot: ExtensibleBot) => Promise<void>;
  cleanup?: () => Promise<void>;
}

export interface BotConfig {
  prefix: string;
  token: string;
  owners?: string[];
  maxFileSize?: number;
  allowedFileTypes?: string[];
}