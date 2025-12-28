import type { ExtractionResult, AnalysisOptions } from "../types/bioinformatics.js";
export declare class SequenceDetector {
    private readonly defaultOptions;
    /**
     * Main extraction method - finds DNA sequences using multiple approaches
     */
    extractSequencesFromMessage(text: string, options?: Partial<AnalysisOptions>): ExtractionResult;
    /**
     * Sequential extraction: scan left-to-right, extract any A, T, C, G letters
     * Example: "Albert told Catherine George" → "ATCG"
     */
    private extractSequentialATCG;
    /**
     * Word-based extraction: extract first ATCG letter from each word
     * Example: "Apple Tree Cat Goat" → "ATCG"
     */
    private extractWordBasedATCG;
    /**
     * Continuous sequence detection: find existing ATCG sequences
     * Example: "Check out ATCGATCG sequence" → ["ATCGATCG"]
     */
    private extractContinuousSequences;
    /**
     * Count total A, T, C, G letters in text for pre-filtering
     */
    private countATCGLetters;
    /**
     * Create a DNASequence object with calculated properties
     */
    private createDNASequence;
    /**
     * Validate if a sequence is worth analyzing
     */
    private isValidSequence;
    /**
     * Calculate GC content percentage
     */
    private calculateGCContent;
    /**
     * Calculate sequence complexity to avoid simple repeats
     */
    private calculateComplexity;
    /**
     * Check if sequence is overly repetitive
     */
    private isRepetitive;
    /**
     * Calculate confidence score for extraction quality
     */
    private calculateExtractionConfidence;
    /**
     * Check if source text contains biological keywords
     */
    private containsBiologicalKeywords;
    /**
     * Check if word is obviously non-biological (URLs, codes, etc.)
     */
    private isNonBiological;
    /**
     * Check if sequence is likely a false positive
     */
    private isLikelyFalsePositive;
    /**
     * Convert IUPAC nucleotide codes to standard ATCG
     */
    private convertIUPACToStandard;
    /**
     * Remove duplicate sequences
     */
    private removeDuplicates;
    /**
     * Calculate overall validation score
     */
    private calculateValidationScore;
}
//# sourceMappingURL=SequenceDetector.d.ts.map