import dotenv from "dotenv";
import { ExtensibleBot } from "./core/Bot.ts";
import { GifPlugin } from "./plugins/GifPlugin.ts";
import { CorePlugin } from "./plugins/CorePlugin.ts";
import type { BotConfig } from "./types/index.ts";

dotenv.config();

async function main() {
  const config: BotConfig = {
    prefix: "!",
    token: process.env.BOT_TOKEN!,
    owners: process.env.BOT_OWNERS?.split(",") || [],
    maxFileSize: 80 * 1024 * 1024, // 80MB
    allowedFileTypes: [".png", ".jpg", ".jpeg", ".gif", ".webp"],
  };

  const bot = new ExtensibleBot(config);

  try {
    await bot.loadPlugin(new CorePlugin());
    await bot.loadPlugin(new GifPlugin());

    await bot.start();

    console.log("🚀 Bot started successfully!");
  } catch (error) {
    console.error("Failed to start bot:", error);
    process.exit(1);
  }
}

main().catch(console.error);
