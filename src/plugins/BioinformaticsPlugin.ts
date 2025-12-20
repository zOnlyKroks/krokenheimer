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
import { BlastApiClient, BlastRateLimiter } from "../services/BlastApiClient.js";
import { SequenceCacheManager } from "../services/SequenceCache.js";
import { SequenceFormatter } from "../utils/SequenceFormatter.js";
import { Logger } from "../core/util/logger.js";

export class BioinformaticsPlugin implements BotPlugin {
  name = "BioinformaticsPlugin";
  description = "Automatic DNA sequence analysis and species identification";
  version = "1.0.0";

  private sequenceDetector = new SequenceDetector();
  private blastClient = new BlastApiClient();
  private blastRateLimiter = new BlastRateLimiter();
  private cacheManager = new SequenceCacheManager();
  private logger = new Logger();

  private isInitialized = false;
  private client?: Client;
  private bot?: ExtensibleBot;

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

  /**
   * Initialize the plugin
   */
  async initialize(client: Client, bot: ExtensibleBot): Promise<void> {
    this.client = client;
    this.bot = bot;

    // Add message event listener for automatic scanning
    client.on("messageCreate", (message) => {
      // Run async without blocking
      this.scanMessage(message).catch(error => {
        this.logger.error('Error in message scanning:', error);
      });
    });

    this.isInitialized = true;
    this.logger.info('BioinformaticsPlugin initialized - automatic DNA sequence detection active');
  }

  /**
   * Cleanup plugin resources
   */
  async cleanup(): Promise<void> {
    this.cacheManager.destroy();
    this.isInitialized = false;
    this.logger.info('BioinformaticsPlugin cleanup completed');
  }

  /**
   * Scan incoming messages for DNA sequences (automatic detection)
   */
  private async scanMessage(message: Message): Promise<void> {
    // Skip bot messages and commands
    if (message.author.bot || message.content.startsWith('!')) {
      return;
    }

    // Skip short messages
    if (message.content.length < 10) {
      return;
    }

    const context: MessageContext = {
      userId: message.author.id,
      channelId: message.channel.id,
      guildId: message.guild?.id,
      messageId: message.id,
      timestamp: message.createdTimestamp
    };

    try {
      // Extract sequences from the message
      const extractionResult = this.sequenceDetector.extractSequencesFromMessage(
        message.content,
        this.analysisOptions
      );

      // Skip if no valid sequences found
      if (extractionResult.sequences.length === 0) {
        return;
      }

      // Process each valid sequence
      for (const sequence of extractionResult.sequences) {
        await this.processSequence(sequence, message, context, true);
      }
    } catch (error) {
      this.logger.error('Error in automatic sequence scanning:', error);
    }
  }

  /**
   * Process a DNA sequence for analysis
   */
  private async processSequence(
    sequence: DNASequence,
    message: Message,
    context: MessageContext,
    isAutomatic: boolean = false
  ): Promise<void> {
    try {
      // Check rate limits
      const rateLimitCheck = this.cacheManager.canUserAnalyze(context);
      if (!rateLimitCheck.allowed) {
        if (!isAutomatic) { // Only show rate limit message for manual commands
          const embed = SequenceFormatter.createRateLimitEmbed(rateLimitCheck.resetTime!);
          await message.reply({ embeds: [embed] });
        }
        return;
      }

      // Check cache first
      let result = this.cacheManager.getCachedResult(sequence);

      if (result) {
        // Cache hit - send result immediately
        const embed = SequenceFormatter.createAnalysisEmbed(result);
        await message.reply({ embeds: [embed] });
        return;
      }

      // Send detection notification for automatic scans
      if (isAutomatic) {
        const detectionEmbed = SequenceFormatter.createDetectionEmbed(
          sequence,
          message.content.substring(0, 100)
        );
        const notificationMsg = await message.reply({ embeds: [detectionEmbed] });

        // Start analysis and update the notification
        try {
          result = await this.analyzeSequence(sequence, context);

          // Update with final results
          const finalEmbed = SequenceFormatter.createAnalysisEmbed(result);
          await notificationMsg.edit({ embeds: [finalEmbed] });
        } catch (error) {
          // Update with error
          const errorMessage = error instanceof Error ? error.message : 'Unknown error';
          const errorEmbed = SequenceFormatter.createErrorEmbed(
            `Analysis failed: ${errorMessage}`,
            sequence
          );
          await notificationMsg.edit({ embeds: [errorEmbed] });
        }
      } else {
        // Manual analysis - show processing message then result
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

  /**
   * Perform BLAST analysis on sequence
   */
  private async analyzeSequence(sequence: DNASequence, context: MessageContext): Promise<SpeciesIdentification> {
    const startTime = Date.now();

    // Submit to BLAST API
    const blastResults = await this.blastClient.analyzeSequence(sequence);

    // Process results into species matches
    const topMatches = blastResults.hits.slice(0, 5).map(hit => ({
      species: hit.scientificName,
      commonName: hit.commonName,
      confidence: this.calculateMatchConfidence(hit),
      identity: hit.identity,
      eValue: hit.eValue,
      description: hit.description,
      taxonId: hit.taxonId
    }));

    // Calculate overall confidence
    const confidence = this.calculateOverallConfidence(sequence, blastResults, topMatches);

    const result: SpeciesIdentification = {
      sequence,
      blastResults,
      topMatches,
      confidence,
      processingTime: Date.now() - startTime,
      cacheHit: false
    };

    // Cache the result
    this.cacheManager.cacheResult(sequence, result);

    return result;
  }

  /**
   * Calculate confidence score for a BLAST match
   */
  private calculateMatchConfidence(hit: any): number {
    let confidence = 0;

    // Identity-based score (0-40 points)
    confidence += Math.min(40, hit.identity * 0.4);

    // E-value based score (0-30 points)
    if (hit.eValue <= 1e-50) confidence += 30;
    else if (hit.eValue <= 1e-20) confidence += 25;
    else if (hit.eValue <= 1e-10) confidence += 20;
    else if (hit.eValue <= 1e-5) confidence += 15;
    else if (hit.eValue <= 0.001) confidence += 10;
    else confidence += 5;

    // Bit score based score (0-20 points)
    confidence += Math.min(20, hit.bitScore / 10);

    // Coverage based score (0-10 points)
    confidence += Math.min(10, hit.coverage / 10);

    return Math.min(100, confidence);
  }

  /**
   * Calculate overall analysis confidence
   */
  private calculateOverallConfidence(
    sequence: DNASequence,
    blastResults: any,
    matches: any[]
  ): ConfidenceLevel {
    let overall = 0;

    // Extraction quality (0-30 points)
    const extractionQuality = sequence.confidence * 30;
    overall += extractionQuality;

    // BLAST reliability (0-40 points)
    let blastReliability = 0;
    if (matches.length > 0) {
      const topMatch = matches[0];
      blastReliability = topMatch.confidence * 0.4;
    }
    overall += blastReliability;

    // Sequence validity (0-30 points)
    let sequenceValidity = 0;
    if (sequence.length >= 50) sequenceValidity += 15;
    else if (sequence.length >= 20) sequenceValidity += 10;
    else sequenceValidity += 5;

    if (sequence.gcContent >= 30 && sequence.gcContent <= 70) sequenceValidity += 15;
    else if (sequence.gcContent >= 20 && sequence.gcContent <= 80) sequenceValidity += 10;
    else sequenceValidity += 5;

    overall += sequenceValidity;

    // Determine confidence level
    let level: ConfidenceLevel['level'];
    if (overall >= 80) level = 'very-high';
    else if (overall >= 65) level = 'high';
    else if (overall >= 45) level = 'medium';
    else if (overall >= 25) level = 'low';
    else level = 'very-low';

    return {
      overall: Math.min(100, overall),
      extractionQuality,
      blastReliability,
      sequenceValidity,
      level
    };
  }

  /**
   * Manual sequence analysis command
   */
  private async manualAnalyze(message: Message, args: string[]): Promise<void> {
    if (args.length === 0) {
      await message.reply(
        "❌ Please provide a DNA sequence to analyze.\n" +
        "Usage: `!analyze ATCGATCGATCG`"
      );
      return;
    }

    const sequenceText = args.join('').toUpperCase().replace(/[^ATCGWSMKRYBDHVN]/g, '');

    if (sequenceText.length < this.analysisOptions.minSequenceLength) {
      await message.reply(
        `❌ Sequence too short. Minimum length is ${this.analysisOptions.minSequenceLength} nucleotides.`
      );
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

  /**
   * Show help command
   */
  private async showHelp(message: Message): Promise<void> {
    const embed = SequenceFormatter.createHelpEmbed();
    await message.reply({ embeds: [embed] });
  }

  /**
   * Show statistics command (admin only)
   */
  private async showStats(message: Message): Promise<void> {
    // Basic admin check (you might want to implement proper admin checking)
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

  /**
   * Calculate GC content percentage
   */
  private calculateGCContent(sequence: string): number {
    const gcCount = (sequence.match(/[GC]/g) || []).length;
    return sequence.length > 0 ? (gcCount / sequence.length) * 100 : 0;
  }
}