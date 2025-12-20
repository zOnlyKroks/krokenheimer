import { createCanvas, loadImage } from "canvas";
import { createWriteStream, unlinkSync } from "fs";
import * as path from "path";
import GIFEncoder from "gifencoder";
import fetch from "node-fetch";
import { fileURLToPath } from "url";
export class GifPlugin {
    name = "GifPlugin";
    description = "Create GIFs from images";
    version = "1.0.0";
    commands = [
        {
            name: "gif",
            description: "Create a GIF from attached images",
            usage: "!gif (attach images)",
            cooldown: 10,
            execute: this.createGif.bind(this)
        },
        {
            name: "gifhelp",
            description: "Show help for GIF commands",
            aliases: ["gh"],
            execute: this.showHelp.bind(this)
        }
    ];
    async createGif(message, args) {
        if (message.attachments.size <= 0) {
            await message.reply("❌ Please attach images to create a GIF");
            return;
        }
        try {
            await message.reply("🔄 Creating GIF...");
            const urls = [...message.attachments.values()].map((att) => att.url);
            // Parse options from args
            const options = this.parseGifOptions(args);
            const gifPath = await this.createGifFromUrls(urls, options);
            await message.reply({
                content: "Here's your GIF 🎞️",
                files: [gifPath],
            });
            unlinkSync(gifPath);
        }
        catch (error) {
            console.error(error);
            await message.reply("❌ Failed to create GIF");
        }
    }
    async showHelp(message) {
        const helpText = `
**🎞️ GIF Plugin Help**

**Commands:**
\`!gif\` - Create a GIF from attached images
\`!gifhelp\` - Show this help message

**Options:**
\`--delay=500\` - Set frame delay in milliseconds (default: 500)
\`--quality=10\` - Set quality 1-20 (lower = better, default: 10)
\`--repeat=0\` - Set repeat count (0 = infinite, default: 0)
\`--width=300\` - Resize width (maintains aspect ratio)
\`--height=300\` - Resize height (maintains aspect ratio)

**Example:**
\`!gif --delay=200 --quality=5\` (attach images)
    `;
        await message.reply(helpText);
    }
    parseGifOptions(args) {
        const options = {
            delay: 500,
            quality: 10,
            repeat: 0
        };
        for (const arg of args) {
            if (arg.startsWith("--delay=")) {
                options.delay = parseInt(arg.split("=")[1] ?? "500") || 500;
            }
            else if (arg.startsWith("--quality=")) {
                options.quality = Math.max(1, Math.min(20, parseInt(arg.split("=")[1] ?? "10") || 10));
            }
            else if (arg.startsWith("--repeat=")) {
                options.repeat = parseInt(arg.split("=")[1] ?? "0") || 0;
            }
            else if (arg.startsWith("--width=")) {
                options.width = parseInt(arg.split("=")[1] ?? "0");
            }
            else if (arg.startsWith("--height=")) {
                options.height = parseInt(arg.split("=")[1] ?? "0");
            }
        }
        return options;
    }
    async createGifFromUrls(urls, options) {
        const images = await Promise.all(urls.map((url) => this.loadImageFromUrl(url)));
        if (images.length === 0) {
            throw new Error("No images loaded");
        }
        let { width, height } = this.calculateDimensions(images[0], options);
        const encoder = new GIFEncoder(width, height);
        const __filename = fileURLToPath(import.meta.url);
        const __dirname = path.dirname(__filename);
        const gifPath = path.join(__dirname, `output_${Date.now()}.gif`);
        const stream = createWriteStream(gifPath);
        encoder.createReadStream().pipe(stream);
        encoder.start();
        encoder.setRepeat(options.repeat);
        encoder.setDelay(options.delay);
        encoder.setQuality(options.quality);
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext("2d");
        for (const img of images) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, width, height);
            encoder.addFrame(ctx);
        }
        encoder.finish();
        await new Promise((resolve) => {
            stream.on("finish", () => resolve());
        });
        return gifPath;
    }
    calculateDimensions(firstImage, options) {
        let width = firstImage.width;
        let height = firstImage.height;
        if (options.width && options.height) {
            width = options.width;
            height = options.height;
        }
        else if (options.width) {
            width = options.width;
            height = Math.round((firstImage.height / firstImage.width) * options.width);
        }
        else if (options.height) {
            height = options.height;
            width = Math.round((firstImage.width / firstImage.height) * options.height);
        }
        return { width, height };
    }
    async loadImageFromUrl(url) {
        const res = await fetch(url);
        if (!res.ok)
            throw new Error(`Failed to fetch ${url}`);
        const buffer = await res.arrayBuffer();
        return loadImage(Buffer.from(buffer));
    }
}
//# sourceMappingURL=GifPlugin.js.map