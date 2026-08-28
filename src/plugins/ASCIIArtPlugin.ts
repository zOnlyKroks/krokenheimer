import { Client, Message, EmbedBuilder } from "discord.js";
import figlet from "figlet";
import type { BotPlugin, BotCommand } from "../types/index.js";
import type { ExtensibleBot } from "../core/Bot.js";
import { Logger } from "../core/util/logger.js";

type FigletFont = Exclude<NonNullable<Parameters<typeof figlet.textSync>[1]>["font"], undefined>;

export class ASCIIArtPlugin implements BotPlugin {
  name = "ASCIIArtPlugin";
  description = "Generate ASCII art from text";
  version = "1.0.0";

  private readonly defaultFont: FigletFont = "Standard";

  // Curated list of pure-ASCII fonts known to render reliably in Discord code blocks.
  private readonly availableFonts: readonly FigletFont[] = [
    "Standard",
    "Big",
    "Block",
    "Doom",
    "Shadow",
    "Small Shadow",
  ];

  private logger = new Logger();

  commands: BotCommand[] = [
    {
      name: "ascii",
      description: "Generate ASCII art from text",
      execute: this.generateAsciiArt.bind(this),
    },
  ];

  async initialize(client: Client, bot: ExtensibleBot): Promise<void> {
    this.logger.info("ASCIIArtPlugin initialized");
  }

  async cleanup(): Promise<void> {
    this.logger.info("ASCIIArtPlugin cleanup completed");
  }

  private findFont(name: string): FigletFont | undefined {
    return this.availableFonts.find(
      (font) => font.toLowerCase() === name.toLowerCase(),
    );
  }

  private async generateAsciiArt(
    message: Message,
    args: string[],
  ): Promise<void> {
    if (args.length === 0) {
      const helpEmbed = new EmbedBuilder()
        .setTitle("🎨 ASCII Art Generator")
        .setColor(0x0099ff)
        .setDescription("Convert text to ASCII art")
        .addFields([
          {
            name: "Usage",
            value:
              "```\n" +
              "!ascii <text>              - Generate ASCII art\n" +
              "!ascii <font> <text>       - Generate ASCII art in a specific font\n" +
              "!ascii fonts               - List available fonts\n" +
              "```",
            inline: false,
          },
          {
            name: "Examples",
            value:
              "```\n" +
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

    if (args[0]!.toLowerCase() === "fonts") {
      await message.reply(
        `🎨 Available fonts: ${this.availableFonts.join(", ")}\nUsage: \`!ascii <font> <text>\``,
      );
      return;
    }

    let font = this.defaultFont;
    let textArgs = args;

    const requestedFont = this.findFont(args[0]!);
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
      await message.reply(
        "❌ Text too long! Please limit to 20 characters or less.",
      );
      return;
    }

    try {
      const asciiArt = figlet.textSync(text.toUpperCase(), { font });
      const codeBlock = "```\n" + asciiArt + "\n```";

      if (codeBlock.length > 2000) {
        await message.reply(
          "❌ Generated ASCII art is too long for Discord. Try shorter text.",
        );
        return;
      }

      // Sent as a plain message rather than an embed field: embed fields render
      // in a narrow column and word-wrap wide code blocks, mangling the art.
      await message.reply(codeBlock);
    } catch (error) {
      this.logger.error("ASCII art generation error:", error);
      await message.reply(
        "❌ Error generating ASCII art. Please try again with different text.",
      );
    }
  }
}
