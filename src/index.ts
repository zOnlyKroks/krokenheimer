import { Client, GatewayIntentBits } from "discord.js";
import { createCanvas, loadImage } from "canvas";
import { createWriteStream, unlinkSync } from "fs";
import * as path from "path";
import GIFEncoder from "gifencoder";
import fetch from "node-fetch";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config();

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

client.once("ready", () => {
  console.log(`✅ Logged in as ${client.user?.tag}`);
});

client.on("messageCreate", async (message) => {
  if (message.author.bot) return;

  if (message.content.startsWith("!gif")) {
    if (message.attachments.size <= 0) {
      await message.reply("❌ Please attach images to create a GIF");
      return;
    }
    try {
      console.log(`📥 Message from ${message.author.tag}: ${message.content}`);

      await message.reply("🔄 Creating GIF...");
      const urls = [...message.attachments.values()].map((att) => att.url);
      const gifPath = await createGifFromUrls(urls);
      await message.reply({
        content: "Here's your GIF 🎞️",
        files: [gifPath],
      });

      unlinkSync(gifPath);
    } catch (err) {
      console.error(err);
      await message.reply("❌ Failed to create GIF");
    }
  }
});

client.login(process.env.BOT_TOKEN);

async function createGifFromUrls(urls: string[]): Promise<string> {
  const images = await Promise.all(urls.map((url) => loadImageFromUrl(url)));

  if (images.length === 0) {
    throw new Error("No images loaded");
  }

  const encoder = new GIFEncoder(images[0]?.width ?? 1, images[0]?.height ?? 1);
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);

  const gifPath = path.join(__dirname, "output.gif");
  const stream = createWriteStream(gifPath);

  encoder.createReadStream().pipe(stream);
  encoder.start();
  encoder.setRepeat(0);
  encoder.setDelay(500);
  encoder.setQuality(10);

  const canvas = createCanvas(images[0]?.width ?? 1, images[0]?.height ?? 1);
  const ctx = canvas.getContext("2d");

  for (const img of images) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    encoder.addFrame(ctx as any);
  }

  encoder.finish();
  await new Promise<void>((resolve) => {
    stream.on("finish", () => resolve());
  });

  return gifPath;
}

async function loadImageFromUrl(url: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
  const buffer = await res.arrayBuffer();
  return loadImage(Buffer.from(buffer));
}
