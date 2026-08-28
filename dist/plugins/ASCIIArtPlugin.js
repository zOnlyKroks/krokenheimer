import { EmbedBuilder } from "discord.js";
import figlet from "figlet";
import { Logger } from "../core/util/logger.js";
export class ASCIIArtPlugin {
    name = "ASCIIArtPlugin";
    description = "Generate ASCII art from text";
    version = "1.0.0";
    defaultFont = "Standard";
    // Curated list of pure-ASCII fonts known to render reliably in Discord code blocks.
    availableFonts = [
        "Standard",
        "Big",
        "Block",
        "Doom",
        "Shadow",
        "Small Shadow",
    ];
    logger = new Logger();
    commands = [
        {
            name: "ascii",
            description: "Generate ASCII art from text",
            execute: this.generateAsciiArt.bind(this),
        },
    ];
    async initialize(client, bot) {
        this.logger.info("ASCIIArtPlugin initialized");
    }
    async cleanup() {
        this.logger.info("ASCIIArtPlugin cleanup completed");
    }
    findFont(name) {
        return this.availableFonts.find((font) => font.toLowerCase() === name.toLowerCase());
    }
    async generateAsciiArt(message, args) {
        if (args.length === 0) {
            const helpEmbed = new EmbedBuilder()
                .setTitle("🎨 ASCII Art Generator")
                .setColor(0x0099ff)
                .setDescription("Convert text to ASCII art")
                .addFields([
                {
                    name: "Usage",
                    value: "```\n" +
                        "!ascii <text>              - Generate ASCII art\n" +
                        "!ascii <font> <text>       - Generate ASCII art in a specific font\n" +
                        "!ascii fonts               - List available fonts\n" +
                        "```",
                    inline: false,
                },
                {
                    name: "Examples",
                    value: "```\n" +
                        "!ascii HELLO         - Creates ASCII art\n" +
                        "!ascii Big WELCOME   - Creates ASCII art using the Big font\n" +
                        "!ascii 2024          - Creates ASCII art with numbers\n" +
                        "```",
                    inline: false,
                },
                {
                    name: "Available Fonts",
                    value: this.availableFonts.join(", "),
                    inline: false,
                },
                {
                    name: "Supported Characters",
                    value: "A-Z, 0-9, space, !, ?",
                    inline: false,
                },
            ])
                .setFooter({
                text: "ASCII art will be displayed in a code block for best formatting",
            });
            await message.reply({ embeds: [helpEmbed] });
            return;
        }
        if (args[0].toLowerCase() === "fonts") {
            await message.reply(`🎨 Available fonts: ${this.availableFonts.join(", ")}\nUsage: \`!ascii <font> <text>\``);
            return;
        }
        let font = this.defaultFont;
        let textArgs = args;
        const requestedFont = this.findFont(args[0]);
        if (requestedFont) {
            font = requestedFont;
            textArgs = args.slice(1);
        }
        const text = textArgs.join(" ");
        if (!text || text.trim().length === 0) {
            await message.reply("❌ Please provide text to convert to ASCII art.");
            return;
        }
        // Limit text length
        if (text.length > 20) {
            await message.reply("❌ Text too long! Please limit to 20 characters or less.");
            return;
        }
        try {
            const asciiArt = figlet.textSync(text.toUpperCase(), { font });
            const codeBlock = "```\n" + asciiArt + "\n```";
            if (codeBlock.length > 2000) {
                await message.reply("❌ Generated ASCII art is too long for Discord. Try shorter text.");
                return;
            }
            // Sent as a plain message rather than an embed field: embed fields render
            // in a narrow column and word-wrap wide code blocks, mangling the art.
            await message.reply(codeBlock);
        }
        catch (error) {
            this.logger.error("ASCII art generation error:", error);
            await message.reply("❌ Error generating ASCII art. Please try again with different text.");
        }
    }
}
//# sourceMappingURL=ASCIIArtPlugin.js.map