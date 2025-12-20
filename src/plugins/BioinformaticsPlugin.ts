import { Message, Client, EmbedBuilder } from "discord.js";
import type { BotPlugin, BotCommand } from "../types/index.js";
import type { ExtensibleBot } from "../core/Bot.js";
import type {
    DNASequence,
    SpeciesIdentification,
    ConfidenceLevel,
    MessageContext,
    AnalysisOptions
} from "../types/bioinformatics.js";

import { SequenceDetector } from "../services/SequenceDetector.js";
import { BlastApiClient } from "../services/BlastApiClient.js";
import { SequenceCacheManager } from "../services/SequenceCache.js";
import { SequenceFormatter } from "../utils/SequenceFormatter.js";
import { Logger } from "../core/util/logger.js";

export class BioinformaticsPlugin implements BotPlugin {
    name = "BioinformaticsPlugin";
    description = "Automatic DNA sequence analysis and species identification";
    version = "1.0.1";

    private sequenceDetector = new SequenceDetector();
    private blastClient = new BlastApiClient();
    private cacheManager = new SequenceCacheManager();
    private logger = new Logger();

    // Analysis options
    private analysisOptions: AnalysisOptions = {
        minSequenceLength: 8,
        maxSequenceLength: 5000,
        enableCaching: true,
        enableRateLimit: true,
        extractionMethods: ['sequential', 'word-based', 'continuous'],
        gcContentRange: [15, 85],
        requiredComplexity: 0.3
    };

    commands: BotCommand[] = [
        {
            name: "analyze",
            description: "Manually analyze a DNA sequence",
            usage: "!analyze <DNA sequence>",
            aliases: ["seq", "dna"],
            cooldown: 10,
            execute: this.manualAnalyze.bind(this)
        },
        {
            name: "biohelp",
            description: "Show bioinformatics help",
            aliases: ["genomehelp", "seqhelp"],
            execute: this.showHelp.bind(this)
        },
        {
            name: "biostats",
            description: "Show analysis statistics (admin only)",
            execute: this.showStats.bind(this)
        }
    ];

    async initialize(client: Client, bot: ExtensibleBot): Promise<void> {
        client.on("messageCreate", (message) => {
            this.scanMessage(message).catch(error => {
                this.logger.error('Error in message scanning:', error);
            });
        });

        this.logger.info('BioinformaticsPlugin initialized - automatic DNA sequence detection active');
    }

    async cleanup(): Promise<void> {
        this.cacheManager.destroy();
        this.logger.info('BioinformaticsPlugin cleanup completed');
    }

    private async scanMessage(message: Message): Promise<void> {
        if (message.author.bot || message.content.startsWith('!')) return;
        if (message.content.length < 15) return;

        const context: MessageContext = {
            userId: message.author.id,
            channelId: message.channel.id,
            guildId: message.guild?.id,
            messageId: message.id,
            timestamp: message.createdTimestamp
        };

        try {
            const extractionResult = this.sequenceDetector.extractSequencesFromMessage(
                message.content,
                this.analysisOptions
            );

            if (extractionResult.sequences.length === 0) return;

            // Only process the best sequence automatically
            const bestSequence = extractionResult.sequences.sort((a, b) => b.length - a.length)[0];

            // @ts-ignore
            await this.processSequence(bestSequence, message, context, true);
        } catch (error) {
            this.logger.error('Error in automatic sequence scanning:', error);
        }
    }

    private async processSequence(
        sequence: DNASequence,
        message: Message,
        context: MessageContext,
        isAutomatic: boolean = false
    ): Promise<void> {
        try {
            const rateLimitCheck = this.cacheManager.canUserAnalyze(context);
            if (!rateLimitCheck.allowed) {
                if (!isAutomatic) {
                    const embed = SequenceFormatter.createRateLimitEmbed(rateLimitCheck.resetTime!);
                    await message.reply({ embeds: [embed] });
                }
                return;
            }

            let result = this.cacheManager.getCachedResult(sequence);
            if (result) {
                const embed = SequenceFormatter.createAnalysisEmbed(result);
                await message.reply({ embeds: [embed] });
                return;
            }

            if (isAutomatic) {
                const detectionEmbed = SequenceFormatter.createDetectionEmbed(sequence, message.content.substring(0, 100));
                const notificationMsg = await message.reply({ embeds: [detectionEmbed] });

                try {
                    result = await this.analyzeSequence(sequence, context);
                    const finalEmbed = SequenceFormatter.createAnalysisEmbed(result);
                    await notificationMsg.edit({ embeds: [finalEmbed] });
                } catch (error) {
                    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                    const errorEmbed = SequenceFormatter.createErrorEmbed(
                        `Analysis failed: ${errorMessage}`,
                        sequence
                    );
                    await notificationMsg.edit({ embeds: [errorEmbed] });
                }
            } else {
                const processingMsg = await message.reply("🔄 Analyzing sequence with NCBI BLAST...");
                try {
                    result = await this.analyzeSequence(sequence, context);
                    const embed = SequenceFormatter.createAnalysisEmbed(result);
                    await processingMsg.edit({ content: '', embeds: [embed] });
                } catch (error) {
                    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                    const errorEmbed = SequenceFormatter.createErrorEmbed(
                        `Analysis failed: ${errorMessage}`,
                        sequence
                    );
                    await processingMsg.edit({ content: '', embeds: [errorEmbed] });
                }
            }
        } catch (error) {
            this.logger.error('Error processing sequence:', error);
            if (!isAutomatic) {
                const errorEmbed = SequenceFormatter.createErrorEmbed(
                    "An unexpected error occurred during analysis."
                );
                await message.reply({ embeds: [errorEmbed] });
            }
        }
    }

    private async analyzeSequence(sequence: DNASequence, context: MessageContext): Promise<SpeciesIdentification> {
        const startTime = Date.now();

        // ✅ Use fixed BlastApiClient
        const blastResults = await this.blastClient.analyzeSequence(sequence);

        const topMatches = blastResults.hits.slice(0, 5).map(hit => ({
            species: hit.scientificName,
            commonName: hit.commonName,
            confidence: this.calculateMatchConfidence(hit),
            identity: hit.identity,
            eValue: hit.eValue,
            description: hit.description,
            taxonId: hit.taxonId
        }));

        const confidence = this.calculateOverallConfidence(sequence, blastResults, topMatches);

        const result: SpeciesIdentification = {
            sequence,
            blastResults,
            topMatches,
            confidence,
            processingTime: Date.now() - startTime,
            cacheHit: false
        };

        this.cacheManager.cacheResult(sequence, result);
        return result;
    }

    private calculateMatchConfidence(hit: any): number {
        let confidence = 0;
        confidence += Math.min(40, hit.identity * 0.4);

        if (hit.eValue <= 1e-50) confidence += 30;
        else if (hit.eValue <= 1e-20) confidence += 25;
        else if (hit.eValue <= 1e-10) confidence += 20;
        else if (hit.eValue <= 1e-5) confidence += 15;
        else if (hit.eValue <= 0.001) confidence += 10;
        else confidence += 5;

        confidence += Math.min(20, hit.bitScore / 10);
        confidence += Math.min(10, hit.coverage / 10);

        return Math.min(100, confidence);
    }

    private calculateOverallConfidence(sequence: DNASequence, blastResults: any, matches: any[]): ConfidenceLevel {
        let overall = 0;
        const extractionQuality = sequence.confidence * 30;
        overall += extractionQuality;

        let blastReliability = 0;
        if (matches.length > 0) blastReliability = matches[0].confidence * 0.4;
        overall += blastReliability;

        let sequenceValidity = 0;
        if (sequence.length >= 50) sequenceValidity += 15;
        else if (sequence.length >= 20) sequenceValidity += 10;
        else sequenceValidity += 5;

        if (sequence.gcContent >= 30 && sequence.gcContent <= 70) sequenceValidity += 15;
        else if (sequence.gcContent >= 20 && sequence.gcContent <= 80) sequenceValidity += 10;
        else sequenceValidity += 5;

        overall += sequenceValidity;

        let level: ConfidenceLevel['level'];
        if (overall >= 80) level = 'very-high';
        else if (overall >= 65) level = 'high';
        else if (overall >= 45) level = 'medium';
        else if (overall >= 25) level = 'low';
        else level = 'very-low';

        return { overall: Math.min(100, overall), extractionQuality, blastReliability, sequenceValidity, level };
    }

    private async manualAnalyze(message: Message, args: string[]): Promise<void> {
        if (args.length === 0) {
            await message.reply(
                "❌ Please provide a DNA sequence to analyze.\nUsage: `!analyze ATCGATCGATCG`"
            );
            return;
        }

        const sequenceText = args.join('').toUpperCase().replace(/[^ATCGWSMKRYBDHVN]/g, '');
        if (sequenceText.length < this.analysisOptions.minSequenceLength) {
            await message.reply(`❌ Sequence too short. Minimum length is ${this.analysisOptions.minSequenceLength} nucleotides.`);
            return;
        }

        const sequence: DNASequence = {
            raw: sequenceText,
            cleaned: sequenceText.replace(/[^ATCG]/g, ''),
            length: sequenceText.length,
            gcContent: this.calculateGCContent(sequenceText),
            isValid: true,
            extractionMethod: 'continuous',
            sourceText: message.content,
            confidence: 0.9
        };

        const context: MessageContext = {
            userId: message.author.id,
            channelId: message.channel.id,
            guildId: message.guild?.id,
            messageId: message.id,
            timestamp: message.createdTimestamp
        };

        await this.processSequence(sequence, message, context, false);
    }

    private async showHelp(message: Message): Promise<void> {
        const embed = SequenceFormatter.createHelpEmbed();
        await message.reply({ embeds: [embed] });
    }

    private async showStats(message: Message): Promise<void> {
        if (!message.member?.permissions.has('Administrator')) {
            await message.reply("❌ This command requires administrator permissions.");
            return;
        }

        const stats = this.cacheManager.getStats();
        const embed = new EmbedBuilder()
            .setTitle("📊 Bioinformatics Plugin Statistics")
            .setColor(0x0099ff)
            .addFields([
                {
                    name: "🗄️ Cache",
                    value: `**Entries**: ${stats.cache.size}\n` +
                        `**Total Hits**: ${stats.cache.hitCount}\n` +
                        `**Oldest Entry**: ${stats.cache.oldestEntry ? new Date(stats.cache.oldestEntry).toLocaleString() : 'None'}`,
                    inline: true
                },
                {
                    name: "⚡ Rate Limiting",
                    value: `**Active Users**: ${stats.rateLimit.activeUsers}\n` +
                        `**Total Requests**: ${stats.rateLimit.totalRequestsInWindow}\n` +
                        `**Avg per User**: ${stats.rateLimit.averageRequestsPerUser.toFixed(1)}`,
                    inline: true
                },
                {
                    name: "🔧 Settings",
                    value: `**Min Length**: ${this.analysisOptions.minSequenceLength} bp\n` +
                        `**Max Length**: ${this.analysisOptions.maxSequenceLength} bp\n` +
                        `**Methods**: ${this.analysisOptions.extractionMethods.join(', ')}`,
                    inline: true
                }
            ])
            .setFooter({ text: `Generated at ${new Date().toLocaleString()}` });

        await message.reply({ embeds: [embed] });
    }

    private calculateGCContent(sequence: string): number {
        const gcCount = (sequence.match(/[GC]/g) || []).length;
        return sequence.length > 0 ? (gcCount / sequence.length) * 100 : 0;
    }
}
