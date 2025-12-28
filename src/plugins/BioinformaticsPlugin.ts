import {Client, EmbedBuilder, Message} from "discord.js";
import type {BotCommand, BotPlugin} from "../types/index.js";
import type {ExtensibleBot} from "../core/Bot.js";
import type {
    AnalysisOptions,
    ConfidenceLevel,
    DNASequence,
    MessageContext,
    SpeciesIdentification
} from "../types/bioinformatics.js";

import {SequenceDetector} from "../services/SequenceDetector.js";
import {BlastApiClient} from "../services/BlastApiClient.js";
import {SequenceFormatter} from "../utils/SequenceFormatter.js";
import {Logger} from "../core/util/logger.js";

export class BioinformaticsPlugin implements BotPlugin {
    name = "BioinformaticsPlugin";
    description = "Automatic DNA sequence analysis and species identification";
    version = "1.0.1";

    private sequenceDetector = new SequenceDetector();
    private blastClient = new BlastApiClient();
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
            name: "biostats",
            description: "Show analysis statistics (admin only)",
            execute: this.showStats.bind(this)
        },
        {
            name: "alias",
            description: "Manage character aliases for DNA sequence processing",
            execute: this.manageAliases.bind(this)
        }
    ];

    async initialize(client: Client, bot: ExtensibleBot): Promise<void> {
        client.on("messageCreate", (message) => {
            if(Math.random() <= 0.1) {
                this.scanMessage(message).catch(error => {
                    this.logger.error('Error in message scanning:', error);
                });
            }
        });

        this.logger.info('BioinformaticsPlugin initialized - automatic DNA sequence detection active');
    }

    async cleanup(): Promise<void> {
        this.logger.info('BioinformaticsPlugin cleanup completed');
    }

    private async scanMessage(message: Message): Promise<void> {
        if (message.author.bot || message.content.startsWith('!')) return;

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
            if (extractionResult.totalAtcgCount < 10) return;

            // Only process the best sequence automatically
            const bestSequence = extractionResult.sequences.sort((a, b) => b.length - a.length)[0];

            // @ts-ignore
            await this.processSequence(bestSequence, message, context);
        } catch (error) {
            this.logger.error('Error in automatic sequence scanning:', error);
        }
    }

    private async processSequence(
        sequence: DNASequence,
        message: Message,
        context: MessageContext
    ): Promise<void> {
        try {
            let result;

            const detectionEmbed = SequenceFormatter.createDetectionEmbed(sequence, message.content.substring(0, 100));
            const notificationMsg = await message.reply({ embeds: [detectionEmbed] });

            try {
                result = await this.analyzeSequence(sequence, context);

                // If no matches found, delete the message to avoid clutter
                if (result.topMatches.length === 0) {
                    await notificationMsg.delete();
                    return;
                }

                const finalEmbed = await SequenceFormatter.createSimpleAnalysisEmbed(result);
                await notificationMsg.edit({ embeds: [finalEmbed] });
            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';

                // Log error instead of sending Discord message
                this.logger.error(`[BIOINFORMATICS] Automatic analysis failed for ${sequence.cleaned?.length || sequence.raw?.length}bp sequence:`, {
                    error: errorMessage,
                    sequence: sequence.cleaned?.substring(0, 50) || sequence.raw?.substring(0, 50),
                    method: sequence.extractionMethod,
                    user: context.userId || 'unknown',
                    channel: context.channelId || 'unknown',
                });

                // Delete the notification message instead of showing error
                try {
                    await notificationMsg.delete();
                } catch (deleteError) {
                    this.logger.error('[BIOINFORMATICS] Failed to delete notification message:', deleteError);
                }
            }
        } catch (error) {
            // Log all sequence processing errors without sending Discord messages
            this.logger.error(`[BIOINFORMATICS] Sequence processing error (automatic):`, {
                error: error instanceof Error ? error.message : String(error),
                sequence: sequence.cleaned?.substring(0, 50) || sequence.raw?.substring(0, 50),
                sequenceLength: sequence.cleaned?.length || sequence.raw?.length,
                method: sequence.extractionMethod,
                user: context.userId || 'unknown',
                channel: context.channelId || 'unknown'
            });
        }
    }

    private async analyzeSequence(sequence: DNASequence, context: MessageContext): Promise<SpeciesIdentification> {
        const startTime = Date.now();

        const blastResults = await this.blastClient.analyzeSequence(sequence);

        const topMatches = blastResults.hits.slice(0, 1).map(hit => ({
            species: hit.scientificName,
            commonName: hit.commonName,
            confidence: this.calculateMatchConfidence(hit),
            identity: hit.identity,
            eValue: hit.eValue,
            description: hit.description,
            taxonId: hit.taxonId
        }));

        const confidence = this.calculateOverallConfidence(sequence, blastResults, topMatches);

        return {
            sequence,
            blastResults,
            topMatches,
            confidence,
            processingTime: Date.now() - startTime
        };
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

        await this.processSequence(sequence, message, context);
    }

    private async showStats(message: Message): Promise<void> {
        if (!message.member?.permissions.has('Administrator')) {
            await message.reply("❌ This command requires administrator permissions.");
            return;
        }

        const embed = new EmbedBuilder()
            .setTitle("📊 Bioinformatics Plugin Statistics")
            .setColor(0x0099ff)
            .addFields([
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

    private async manageAliases(message: Message, args: string[]): Promise<void> {
        if (args.length === 0) {
            const helpEmbed = new EmbedBuilder()
                .setTitle("🔧 Character Alias Management")
                .setColor(0x0099ff)
                .setDescription("Manage character aliases for DNA sequence processing")
                .addFields([
                    {
                        name: "Commands",
                        value: "```\n" +
                            "!alias set <char> <replacement>  - Set character alias (ä → ae)\n" +
                            "!alias remove <char>             - Remove character alias\n" +
                            "!alias list                      - Show all current aliases\n" +
                            "!alias reset                     - Reset to default aliases\n" +
                            "```",
                        inline: false
                    },
                    {
                        name: "Examples",
                        value: "```\n" +
                            "!alias set ä ae      - Maps ä to ae\n" +
                            "!alias set ø o       - Maps ø to o\n" +
                            "!alias remove ä      - Removes ä mapping\n" +
                            "```",
                        inline: false
                    }
                ])
                .setFooter({ text: "Character aliases help improve DNA sequence detection from text containing accented characters." });

            await message.reply({ embeds: [helpEmbed] });
            return;
        }

        if(!args[0]) throw Error("Args0 empty!")

        const subcommand = args[0].toLowerCase();

        switch (subcommand) {
            case 'set':
                await this.setAlias(message, args.slice(1));
                break;
            case 'remove':
                await this.removeAlias(message, args.slice(1));
                break;
            case 'list':
                await this.listAliases(message);
                break;
            case 'reset':
                await this.resetAliases(message);
                break;
            default:
                await message.reply(`❌ Unknown alias command: ${subcommand}\n\nUse \`!alias\` to see available commands.`);
        }
    }

    private async setAlias(message: Message, args: string[]): Promise<void> {
        if (args.length !== 2) {
            await message.reply("❌ Usage: `!alias set <character> <replacement>`\n\nExample: `!alias set ä ae`");
            return;
        }

        const [fromChar, toReplacement] = args;

        if(!fromChar || !toReplacement) throw new Error("From or too chart empty!")

        // Validate input
        if (fromChar.length !== 1) {
            await message.reply("❌ The character to replace must be exactly one character.");
            return;
        }

        if (toReplacement.length > 5) {
            await message.reply("❌ Replacement string cannot be longer than 5 characters.");
            return;
        }

        // Prevent replacing standard DNA bases
        if (['A', 'T', 'C', 'G', 'a', 't', 'c', 'g'].includes(fromChar)) {
            await message.reply("❌ Cannot create aliases for standard DNA bases (A, T, C, G).");
            return;
        }

        this.sequenceDetector.setCharacterAlias(fromChar, toReplacement);

        const embed = new EmbedBuilder()
            .setTitle("✅ Character Alias Set")
            .setColor(0x00ff00)
            .setDescription(`Character **${fromChar}** will now be replaced with **${toReplacement}** before DNA sequence processing.`)
            .addFields([
                {
                    name: "Example",
                    value: `Text containing "${fromChar}" → Processed as "${toReplacement}"`,
                    inline: false
                }
            ])
            .setTimestamp();

        await message.reply({ embeds: [embed] });
    }

    private async removeAlias(message: Message, args: string[]): Promise<void> {
        if (args.length !== 1) {
            await message.reply("❌ Usage: `!alias remove <character>`\n\nExample: `!alias remove ä`");
            return;
        }

        const charToRemove = args[0];

        if(!charToRemove) throw new Error("Empty char!")

        if (charToRemove.length !== 1) {
            await message.reply("❌ Please specify exactly one character to remove.");
            return;
        }

        const wasRemoved = this.sequenceDetector.removeCharacterAlias(charToRemove);

        if (!wasRemoved) {
            await message.reply(`❌ No alias found for character **${charToRemove}**.`);
            return;
        }

        const embed = new EmbedBuilder()
            .setTitle("✅ Character Alias Removed")
            .setColor(0xff9900)
            .setDescription(`Alias for character **${charToRemove}** has been removed.`)
            .setTimestamp();

        await message.reply({ embeds: [embed] });
    }

    private async listAliases(message: Message): Promise<void> {
        const aliases = this.sequenceDetector.getCharacterAliases();

        if (aliases.size === 0) {
            await message.reply("📝 No character aliases are currently configured.");
            return;
        }

        // Group aliases for better display
        const aliasGroups = {
            "German Umlauts": [] as string[],
            "Accented Vowels": [] as string[],
            "Other Characters": [] as string[],
            "Custom": [] as string[]
        };

        const defaultAliases = new Set([
            'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'à', 'á', 'â', 'ã', 'å', 'À', 'Á', 'Â', 'Ã', 'Å',
            'è', 'é', 'ê', 'ë', 'È', 'É', 'Ê', 'Ë', 'ì', 'í', 'î', 'ï', 'Ì', 'Í', 'Î', 'Ï',
            'ò', 'ó', 'ô', 'õ', 'Ò', 'Ó', 'Ô', 'Õ', 'ù', 'ú', 'û', 'Ù', 'Ú', 'Û',
            'ç', 'Ç', 'ñ', 'Ñ', 'ß'
        ]);

        for (const [from, to] of aliases.entries()) {
            const aliasStr = `${from} → ${to}`;

            if (!defaultAliases.has(from)) {
                aliasGroups["Custom"].push(aliasStr);
            } else if (['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß'].includes(from)) {
                aliasGroups["German Umlauts"].push(aliasStr);
            } else if (/[àáâãåÀÁÂÃÅèéêëÈÉÊËìíîïÌÍÎÏòóôõÒÓÔÕùúûÙÚÛ]/.test(from)) {
                aliasGroups["Accented Vowels"].push(aliasStr);
            } else {
                aliasGroups["Other Characters"].push(aliasStr);
            }
        }

        const embed = new EmbedBuilder()
            .setTitle("📝 Character Aliases")
            .setColor(0x0099ff)
            .setDescription(`${aliases.size} character alias${aliases.size === 1 ? '' : 'es'} configured`)
            .setTimestamp();

        for (const [groupName, groupAliases] of Object.entries(aliasGroups)) {
            if (groupAliases.length > 0) {
                // Split into chunks of 20 to avoid field value limits
                const chunks = [];
                for (let i = 0; i < groupAliases.length; i += 20) {
                    chunks.push(groupAliases.slice(i, i + 20));
                }

                chunks.forEach((chunk, index) => {
                    const name = chunks.length === 1 ? groupName : `${groupName} (${index + 1}/${chunks.length})`;
                    embed.addFields([{
                        name: name,
                        value: "```\n" + chunk.join('\n') + "\n```",
                        inline: true
                    }]);
                });
            }
        }

        await message.reply({ embeds: [embed] });
    }

    private async resetAliases(message: Message): Promise<void> {
        if (!message.member?.permissions.has('Administrator')) {
            await message.reply("❌ This command requires administrator permissions.");
            return;
        }

        this.sequenceDetector.resetCharacterAliases();

        const embed = new EmbedBuilder()
            .setTitle("🔄 Character Aliases Reset")
            .setColor(0xff9900)
            .setDescription("All character aliases have been reset to default values.")
            .addFields([
                {
                    name: "Default Aliases Include",
                    value: "• German umlauts (ä→ae, ö→oe, ü→ue)\n" +
                           "• Accented vowels (á→a, é→e, í→i, etc.)\n" +
                           "• Common special characters (ç→c, ñ→n, ß→ss)",
                    inline: false
                }
            ])
            .setFooter({ text: "Use !alias list to see all default aliases" })
            .setTimestamp();

        await message.reply({ embeds: [embed] });
    }

    private calculateGCContent(sequence: string): number {
        const gcCount = (sequence.match(/[GC]/g) || []).length;
        return sequence.length > 0 ? (gcCount / sequence.length) * 100 : 0;
    }
}
