import { EmbedBuilder } from "discord.js";
import type { SpeciesIdentification, DNASequence, BlastHit } from "../types/bioinformatics.js";
/**
 * Formats bioinformatics analysis results for Discord
 */
export declare class SequenceFormatter {
    /**
     * Create main analysis result embed
     */
    static createAnalysisEmbed(result: SpeciesIdentification): EmbedBuilder;
    /**
     * Create a simple notification embed for automatic detection
     */
    static createDetectionEmbed(sequence: DNASequence, extractedFrom: string): EmbedBuilder;
    /**
     * Create error embed
     */
    static createErrorEmbed(error: string, sequence?: DNASequence): EmbedBuilder;
    /**
     * Create rate limit embed
     */
    static createRateLimitEmbed(resetTime: number): EmbedBuilder;
    /**
     * Create help embed
     */
    static createHelpEmbed(): EmbedBuilder;
    /**
     * Format sequence information
     */
    private static formatSequenceInfo;
    /**
     * Format processing information
     */
    private static formatProcessingInfo;
    /**
     * Format quality/confidence information
     */
    private static formatQualityInfo;
    /**
     * Format top species match
     */
    private static formatTopMatch;
    /**
     * Format additional matches
     */
    private static formatAdditionalMatches;
    /**
     * Format extraction method details
     */
    private static formatExtractionMethod;
    /**
     * Format confidence level with visual indicator
     */
    private static formatConfidenceLevel;
    /**
     * Get color based on confidence level
     */
    private static getConfidenceColor;
    /**
     * Get emoji for confidence level
     */
    private static getConfidenceEmoji;
    /**
     * Get emoji for extraction method
     */
    private static getMethodEmoji;
    /**
     * Get description for extraction method
     */
    private static getExtractionMethodDescription;
    /**
     * Create confidence bars visualization
     */
    private static getConfidenceBars;
    /**
     * Format E-value in scientific notation
     */
    private static formatEValue;
    /**
     * Create footer text with technical details
     */
    private static createFooter;
    /**
     * Create a detailed alignment view (for manual analysis)
     */
    static createAlignmentEmbed(result: SpeciesIdentification, hit: BlastHit): EmbedBuilder;
    /**
     * Format a sequence for display (with line breaks)
     */
    static formatSequenceDisplay(sequence: string, lineLength?: number): string;
}
//# sourceMappingURL=SequenceFormatter.d.ts.map