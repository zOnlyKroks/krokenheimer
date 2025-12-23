import { EmbedBuilder } from "discord.js";
import { SpeciesImageService } from "../services/SpeciesImageService.js";
/**
 * Formats bioinformatics analysis results for Discord
 */
export class SequenceFormatter {
    /**
     * Create simplified analysis result embed (default)
     */
    static async createSimpleAnalysisEmbed(result) {
        const embed = new EmbedBuilder()
            .setTitle("DNA Analysis")
            .setColor(this.getConfidenceColor(result.confidence))
            .setTimestamp();
        // Show the scanned sequence
        const sequencePreview = result.sequence.cleaned.length > 100
            ? `${result.sequence.cleaned.substring(0, 100)}...`
            : result.sequence.cleaned;
        embed.addFields([
            {
                name: "Scanned Sequence:",
                value: `\`\`\`${sequencePreview}\`\`\`\n`,
                inline: false
            }
        ]);
        // Show only the top species match
        if (result.topMatches.length > 0) {
            const topMatch = result.topMatches[0];
            if (topMatch) {
                const commonName = topMatch.commonName ? ` (${topMatch.commonName})` : '';
                const description = topMatch.description ? ` - ${topMatch.description}` : '';
                const matchText = `**${topMatch.species}**${commonName} - ${topMatch.identity.toFixed(1)}%${description}`;
                embed.addFields([
                    {
                        name: "Species Match",
                        value: matchText,
                        inline: false
                    }
                ]);
            }
            // Add species image for top match
            try {
                const topMatch = result.topMatches[0];
                if (topMatch) {
                    const imageUrl = await SpeciesImageService.getSpeciesImage(topMatch.species, topMatch.commonName || undefined);
                    if (imageUrl) {
                        embed.setThumbnail(imageUrl);
                    }
                }
            }
            catch (error) {
                // Silently fail
            }
        }
        else {
            embed.addFields([
                {
                    name: "No Matches Found",
                    value: "No significant species matches were found for this sequence.",
                    inline: false
                }
            ]);
        }
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
            }
        ])
            .setFooter({ text: "Results will be posted shortly • Powered by NCBI BLAST" });
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
}
//# sourceMappingURL=SequenceFormatter.js.map