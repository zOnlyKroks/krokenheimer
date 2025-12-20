import { Client } from "discord.js";
import type { BotPlugin, BotCommand } from "../types/index.js";
import type { ExtensibleBot } from "../core/Bot.js";
export declare class BioinformaticsPlugin implements BotPlugin {
    name: string;
    description: string;
    version: string;
    private sequenceDetector;
    private blastClient;
    private blastRateLimiter;
    private cacheManager;
    private logger;
    private isInitialized;
    private client?;
    private bot?;
    private analysisOptions;
    commands: BotCommand[];
    /**
     * Initialize the plugin
     */
    initialize(client: Client, bot: ExtensibleBot): Promise<void>;
    /**
     * Cleanup plugin resources
     */
    cleanup(): Promise<void>;
    /**
     * Scan incoming messages for DNA sequences (automatic detection)
     */
    private scanMessage;
    /**
     * Process a DNA sequence for analysis
     */
    private processSequence;
    /**
     * Perform BLAST analysis on sequence
     */
    private analyzeSequence;
    /**
     * Calculate confidence score for a BLAST match
     */
    private calculateMatchConfidence;
    /**
     * Calculate overall analysis confidence
     */
    private calculateOverallConfidence;
    /**
     * Manual sequence analysis command
     */
    private manualAnalyze;
    /**
     * Show help command
     */
    private showHelp;
    /**
     * Show statistics command (admin only)
     */
    private showStats;
    /**
     * Calculate GC content percentage
     */
    private calculateGCContent;
}
//# sourceMappingURL=BioinformaticsPlugin.d.ts.map