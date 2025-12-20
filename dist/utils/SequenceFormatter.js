import { EmbedBuilder } from "discord.js";
import { SpeciesImageService } from "../services/SpeciesImageService.js";
/**
 * Formats bioinformatics analysis results for Discord
 */
export class SequenceFormatter {
    /**
     * Create main analysis result embed
     */
    static async createAnalysisEmbed(result) {
        const embed = new EmbedBuilder()
            .setTitle("🧬 DNA Sequence Analysis")
            .setColor(this.getConfidenceColor(result.confidence))
            .setTimestamp();
        // Basic sequence info
        embed.addFields([
            {
                name: "📏 Sequence Info",
                value: this.formatSequenceInfo(result.sequence),
                inline: true
            },
            {
                name: "⚡ Processing",
                value: this.formatProcessingInfo(result),
                inline: true
            },
            {
                name: "📊 Quality",
                value: this.formatQualityInfo(result.confidence),
                inline: true
            }
        ]);
        // Top species match
        if (result.topMatches.length > 0) {
            const topMatch = result.topMatches[0];
            if (topMatch) {
                embed.addFields([
                    {
                        name: "🥇 Top Species Match",
                        value: this.formatTopMatch(topMatch),
                        inline: false
                    }
                ]);
                // Try to fetch and add species image
                try {
                    const imageUrl = await SpeciesImageService.getSpeciesImage(topMatch.species, topMatch.commonName || undefined);
                    if (imageUrl) {
                        // Set as thumbnail for a cleaner look (smaller image in top-right)
                        embed.setThumbnail(imageUrl);
                    }
                }
                catch (error) {
                    // Silently fail if image fetching fails - don't break the embed
                    console.warn(`Failed to fetch image for species ${topMatch.species}:`, error);
                }
            }
            // Additional matches if available
            if (result.topMatches.length > 1) {
                const otherMatches = result.topMatches.slice(1, 4); // Show up to 3 more
                embed.addFields([
                    {
                        name: "🔍 Other Matches",
                        value: this.formatAdditionalMatches(otherMatches),
                        inline: false
                    }
                ]);
            }
        }
        else {
            embed.addFields([
                {
                    name: "❌ No Matches Found",
                    value: "No significant species matches were found for this sequence.",
                    inline: false
                }
            ]);
        }
        // Extraction method details
        embed.addFields([
            {
                name: "🔬 Extraction Method",
                value: this.formatExtractionMethod(result.sequence),
                inline: true
            },
            {
                name: "📈 Confidence Level",
                value: this.formatConfidenceLevel(result.confidence),
                inline: true
            }
        ]);
        // Footer with technical details
        const footer = this.createFooter(result);
        embed.setFooter({ text: footer });
        return embed;
    }
    /**
     * Create a simple notification embed for automatic detection
     */
    static createDetectionEmbed(sequence, extractedFrom) {
        return new EmbedBuilder()
            .setTitle("🧬 DNA Sequence Detected!")
            .setColor(0x00ff00)
            .setDescription(`I found a DNA sequence in your message! Analyzing ${sequence.length} nucleotides...`)
            .addFields([
            {
                name: "Extracted Sequence",
                value: `\`${sequence.cleaned.substring(0, 50)}${sequence.cleaned.length > 50 ? '...' : ''}\``,
                inline: false
            },
            {
                name: "Extraction Method",
                value: this.getExtractionMethodDescription(sequence.extractionMethod),
                inline: true
            },
            {
                name: "Processing",
                value: "⏳ Querying NCBI BLAST...",
                inline: true
            }
        ])
            .setFooter({ text: "Results will be posted shortly • Powered by NCBI BLAST" });
    }
    /**
     * Create error embed
     */
    static createErrorEmbed(error, sequence) {
        const embed = new EmbedBuilder()
            .setTitle("❌ Analysis Error")
            .setColor(0xff0000)
            .setDescription(error);
        if (sequence) {
            embed.addFields([
                {
                    name: "Sequence Info",
                    value: `Length: ${sequence.length} bp\nMethod: ${sequence.extractionMethod}`,
                    inline: true
                }
            ]);
        }
        embed.setFooter({ text: "Try again later or contact support if the problem persists" });
        return embed;
    }
    /**
     * Create rate limit embed
     */
    static createRateLimitEmbed(resetTime) {
        const secondsUntilReset = Math.ceil((resetTime - Date.now()) / 1000);
        return new EmbedBuilder()
            .setTitle("⏰ Rate Limit Reached")
            .setColor(0xff9900)
            .setDescription("You've reached the limit for DNA sequence analysis.")
            .addFields([
            {
                name: "Limit",
                value: "3 sequences per minute",
                inline: true
            },
            {
                name: "Reset",
                value: `In ${secondsUntilReset} seconds`,
                inline: true
            }
        ])
            .setFooter({ text: "Rate limiting helps ensure fair access to analysis resources" });
    }
    /**
     * Create help embed
     */
    static createHelpEmbed() {
        return new EmbedBuilder()
            .setTitle("🧬 Bioinformatics Plugin Help")
            .setColor(0x0099ff)
            .setDescription("Automatic DNA sequence analysis with species identification")
            .addFields([
            {
                name: "🔍 Automatic Detection",
                value: "I automatically scan messages for DNA patterns using multiple methods:\n" +
                    "• **Sequential**: Extract A, T, C, G letters from any text\n" +
                    "• **Word-based**: Extract first ATCG letter from each word\n" +
                    "• **Continuous**: Find existing DNA sequences",
                inline: false
            },
            {
                name: "🧪 Examples",
                value: "**Normal text**: \"Albert told Catherine George\" → ATCG\n" +
                    "**Mixed content**: \"Amazing tigers, cool animals\" → ATCA\n" +
                    "**DNA sequences**: \"Check out ATCGATCG\" → ATCGATCG",
                inline: false
            },
            {
                name: "⚙️ Commands",
                value: "`!analyze <sequence>` - Manually analyze a DNA sequence\n" +
                    "`!biohelp` - Show this help message",
                inline: false
            },
            {
                name: "📊 Results Include",
                value: "• Species name (scientific & common)\n" +
                    "• Confidence scores and identity percentages\n" +
                    "• Multiple species matches\n" +
                    "• Alignment and sequence details",
                inline: false
            },
            {
                name: "🛡️ Rate Limits",
                value: "3 analyses per minute per user",
                inline: true
            },
            {
                name: "🔬 Powered By",
                value: "NCBI BLAST API",
                inline: true
            }
        ])
            .setFooter({ text: "Results are cached for 24 hours • Sequences are not stored permanently" });
    }
    /**
     * Format sequence information
     */
    static formatSequenceInfo(sequence) {
        return `**Length**: ${sequence.length} bp\n` +
            `**GC Content**: ${sequence.gcContent.toFixed(1)}%\n` +
            `**Preview**: \`${sequence.cleaned.substring(0, 20)}${sequence.cleaned.length > 20 ? '...' : ''}\``;
    }
    /**
     * Format processing information
     */
    static formatProcessingInfo(result) {
        const time = (result.processingTime / 1000).toFixed(1);
        const cacheStatus = result.cacheHit ? "Cache Hit" : "Fresh Analysis";
        return `**Time**: ${time}s\n` +
            `**Status**: ${cacheStatus}\n` +
            `**Database**: NCBI nt`;
    }
    /**
     * Format quality/confidence information
     */
    static formatQualityInfo(confidence) {
        return `**Overall**: ${confidence.overall.toFixed(0)}%\n` +
            `**Extraction**: ${confidence.extractionQuality.toFixed(0)}%\n` +
            `**Level**: ${this.getConfidenceEmoji(confidence.level)} ${confidence.level.replace('-', ' ').toUpperCase()}`;
    }
    /**
     * Format top species match
     */
    static formatTopMatch(match) {
        const commonNameText = match.commonName ? ` (${match.commonName})` : '';
        const confidence = match.confidence.toFixed(1);
        const identity = match.identity.toFixed(1);
        return `**${match.species}**${commonNameText}\n` +
            `🎯 **Identity**: ${identity}%\n` +
            `📊 **Confidence**: ${confidence}%\n` +
            `🔬 **E-value**: ${this.formatEValue(match.eValue)}\n` +
            `📝 ${match.description.substring(0, 100)}${match.description.length > 100 ? '...' : ''}`;
    }
    /**
     * Format additional matches
     */
    static formatAdditionalMatches(matches) {
        return matches.map((match, index) => {
            const commonNameText = match.commonName ? ` (${match.commonName})` : '';
            return `**${index + 2}.** ${match.species}${commonNameText} - ${match.identity.toFixed(1)}% identity`;
        }).join('\n');
    }
    /**
     * Format extraction method details
     */
    static formatExtractionMethod(sequence) {
        const methodEmoji = this.getMethodEmoji(sequence.extractionMethod);
        const methodName = sequence.extractionMethod.replace('-', ' ').toUpperCase();
        const confidence = (sequence.confidence * 100).toFixed(0);
        return `${methodEmoji} **${methodName}**\n` +
            `📈 ${confidence}% confidence`;
    }
    /**
     * Format confidence level with visual indicator
     */
    static formatConfidenceLevel(confidence) {
        const emoji = this.getConfidenceEmoji(confidence.level);
        const bars = this.getConfidenceBars(confidence.overall);
        return `${emoji} **${confidence.level.replace('-', ' ').toUpperCase()}**\n` +
            `${bars} ${confidence.overall.toFixed(0)}%`;
    }
    /**
     * Get color based on confidence level
     */
    static getConfidenceColor(confidence) {
        switch (confidence.level) {
            case 'very-high': return 0x00ff00; // Green
            case 'high': return 0x7fff00; // Light green
            case 'medium': return 0xffff00; // Yellow
            case 'low': return 0xff9900; // Orange
            case 'very-low': return 0xff0000; // Red
            default: return 0x808080; // Gray
        }
    }
    /**
     * Get emoji for confidence level
     */
    static getConfidenceEmoji(level) {
        switch (level) {
            case 'very-high': return '🟢';
            case 'high': return '🔵';
            case 'medium': return '🟡';
            case 'low': return '🟠';
            case 'very-low': return '🔴';
            default: return '⚫';
        }
    }
    /**
     * Get emoji for extraction method
     */
    static getMethodEmoji(method) {
        switch (method) {
            case 'continuous': return '📏';
            case 'sequential': return '🔗';
            case 'word-based': return '📝';
            case 'hybrid': return '🔀';
            default: return '🔬';
        }
    }
    /**
     * Get description for extraction method
     */
    static getExtractionMethodDescription(method) {
        switch (method) {
            case 'continuous': return '📏 Found existing DNA sequence';
            case 'sequential': return '🔗 Extracted letters sequentially';
            case 'word-based': return '📝 Extracted from word beginnings';
            case 'hybrid': return '🔀 Combined extraction methods';
            default: return '🔬 Custom extraction';
        }
    }
    /**
     * Create confidence bars visualization
     */
    static getConfidenceBars(percentage) {
        const filled = Math.round(percentage / 10);
        const empty = 10 - filled;
        return '█'.repeat(filled) + '░'.repeat(empty);
    }
    /**
     * Format E-value in scientific notation
     */
    static formatEValue(eValue) {
        if (eValue === 0)
            return '0';
        if (eValue >= 1)
            return eValue.toFixed(1);
        return eValue.toExponential(1);
    }
    /**
     * Create footer text with technical details
     */
    static createFooter(result) {
        const timestamp = new Date().toLocaleString();
        const cacheText = result.cacheHit ? ' • Cached result' : '';
        return `Generated • ${timestamp}${cacheText}`;
    }
}
//# sourceMappingURL=SequenceFormatter.js.map