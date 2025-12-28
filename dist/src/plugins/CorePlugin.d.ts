import { Client } from "discord.js";
import type { BotPlugin, BotCommand } from "../types/index.js";
import type { ExtensibleBot } from "../core/Bot.ts";
export declare class CorePlugin implements BotPlugin {
    name: string;
    description: string;
    version: string;
    private bot;
    commands: BotCommand[];
    initialize(client: Client, bot: ExtensibleBot): Promise<void>;
    private showHelp;
    private ping;
    private listPlugins;
}
//# sourceMappingURL=CorePlugin.d.ts.map