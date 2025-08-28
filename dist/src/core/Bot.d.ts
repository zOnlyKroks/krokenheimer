import type { BotPlugin, BotCommand, BotConfig } from "../types/index.ts";
export declare class ExtensibleBot {
    private client;
    private plugins;
    private commands;
    private cooldowns;
    private logger;
    private config;
    constructor(config: BotConfig);
    private setupEventHandlers;
    private handleMessage;
    private checkCooldown;
    private checkPermissions;
    loadPlugin(plugin: BotPlugin): Promise<void>;
    unloadPlugin(pluginName: string): Promise<void>;
    getLoadedPlugins(): string[];
    getCommands(): BotCommand[];
    start(): Promise<void>;
    private shutdown;
}
//# sourceMappingURL=Bot.d.ts.map