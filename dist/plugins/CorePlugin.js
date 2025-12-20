export class CorePlugin {
    name = "CorePlugin";
    description = "Core bot functionality";
    version = "1.0.0";
    bot = null;
    commands = [
        {
            name: "help",
            description: "Show available commands",
            aliases: ["h", "commands"],
            execute: this.showHelp.bind(this),
        },
        {
            name: "ping",
            description: "Check bot latency",
            execute: this.ping.bind(this),
        },
        {
            name: "plugins",
            description: "List loaded plugins",
            execute: this.listPlugins.bind(this),
        },
    ];
    async initialize(client, bot) {
        this.bot = bot;
        console.log("Core plugin initialized");
    }
    async showHelp(message, args) {
        if (!this.bot) {
            await message.reply("❌ Bot reference not available");
            return;
        }
        if (args.length > 0) {
            const commandName = args[0]?.toLowerCase() ?? "noCommand";
            const commands = this.bot.getCommands();
            const command = commands.find((cmd) => cmd.name === commandName || cmd.aliases?.includes(commandName));
            if (command) {
                const helpText = `
**📖 Command: ${command.name}**
${command.description}

**Usage:** \`!${command.usage || command.name}\`
${command.aliases ? `**Aliases:** ${command.aliases.join(", ")}` : ""}
${command.cooldown ? `**Cooldown:** ${command.cooldown}s` : ""}
        `;
                await message.reply(helpText);
                return;
            }
            else {
                await message.reply(`❌ Command \`${commandName}\` not found.`);
                return;
            }
        }
        const commands = this.bot.getCommands();
        const commandList = commands
            .map((cmd) => `• \`!${cmd.name}\` - ${cmd.description}`)
            .join("\n");
        const helpText = `
**🤖 Bot Commands**

Use \`!help <command>\` for detailed info about a specific command.

**Available Commands:**
${commandList}
    `;
        await message.reply(helpText);
    }
    async ping(message) {
        const sent = await message.reply("🏓 Pinging...");
        const latency = sent.createdTimestamp - message.createdTimestamp;
        const wsLatency = message.client.ws.ping;
        await sent.edit(`🏓 Pong!\n**Message Latency:** ${latency}ms\n**WebSocket Latency:** ${wsLatency}ms`);
    }
    async listPlugins(message) {
        if (!this.bot) {
            await message.reply("❌ Bot reference not available");
            return;
        }
        const plugins = this.bot.getLoadedPlugins();
        const pluginList = plugins
            .map((name) => `• ${name}`)
            .join("\n");
        await message.reply(`**📦 Loaded Plugins (${plugins.length})**\n${pluginList}`);
    }
}
//# sourceMappingURL=CorePlugin.js.map