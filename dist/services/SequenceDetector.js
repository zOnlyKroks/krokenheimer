export class SequenceDetector {
    defaultOptions = {
        minSequenceLength: 8,
        maxSequenceLength: 10000,
        enableCaching: true,
        enableRateLimit: true,
        extractionMethods: ['sequential', 'word-based', 'continuous'],
        gcContentRange: [15, 85],
        requiredComplexity: 0.3
    };
    characterAliases = new Map([
        // German umlauts
        ['ä', 'ae'], ['ö', 'oe'], ['ü', 'ue'],
        ['Ä', 'AE'], ['Ö', 'OE'], ['Ü', 'UE'],
        // Accented vowels
        ['à', 'a'], ['á', 'a'], ['â', 'a'], ['ã', 'a'], ['å', 'a'],
        ['À', 'A'], ['Á', 'A'], ['Â', 'A'], ['Ã', 'A'], ['Å', 'A'],
        ['è', 'e'], ['é', 'e'], ['ê', 'e'], ['ë', 'e'],
        ['È', 'E'], ['É', 'E'], ['Ê', 'E'], ['Ë', 'E'],
        ['ì', 'i'], ['í', 'i'], ['î', 'i'], ['ï', 'i'],
        ['Ì', 'I'], ['Í', 'I'], ['Î', 'I'], ['Ï', 'I'],
        ['ò', 'o'], ['ó', 'o'], ['ô', 'o'], ['õ', 'o'],
        ['Ò', 'O'], ['Ó', 'O'], ['Ô', 'O'], ['Õ', 'O'],
        ['ù', 'u'], ['ú', 'u'], ['û', 'u'],
        ['Ù', 'U'], ['Ú', 'U'], ['Û', 'U'],
        // Other common characters
        ['ç', 'c'], ['Ç', 'C'],
        ['ñ', 'n'], ['Ñ', 'N'],
        ['ß', 'ss']
    ]);
    /**
     * Main extraction method - finds DNA sequences using multiple approaches
     */
    extractSequencesFromMessage(text, options) {
        const opts = { ...this.defaultOptions, ...options };
        const startTime = Date.now();
        // Apply character aliases before processing
        const processedText = this.applyCharacterAliases(text);
        // Pre-filter: check if message has enough ATCG letters to be worth analyzing
        const atcgCount = this.countATCGLetters(processedText);
        if (atcgCount < opts.minSequenceLength) {
            return {
                sequences: [],
                totalAtcgCount: atcgCount,
                messageLength: text.length,
                extractionMethods: [],
                processingTime: Date.now() - startTime
            };
        }
        const sequences = [];
        const usedMethods = [];
        // Method 1: Sequential extraction
        if (opts.extractionMethods.includes('sequential')) {
            const sequential = this.extractSequentialATCG(processedText);
            if (sequential.length >= opts.minSequenceLength) {
                const seq = this.createDNASequence(sequential, text, 'sequential');
                if (this.isValidSequence(seq, opts).isValid) {
                    sequences.push(seq);
                    usedMethods.push('sequential');
                }
            }
        }
        // Method 2: Word-based extraction
        if (opts.extractionMethods.includes('word-based')) {
            const wordBased = this.extractWordBasedATCG(processedText);
            if (wordBased.length >= opts.minSequenceLength) {
                const seq = this.createDNASequence(wordBased, text, 'word-based');
                if (this.isValidSequence(seq, opts).isValid) {
                    sequences.push(seq);
                    usedMethods.push('word-based');
                }
            }
        }
        // Method 3: Continuous sequence detection
        if (opts.extractionMethods.includes('continuous')) {
            const continuous = this.extractContinuousSequences(processedText);
            continuous.forEach(seq => {
                if (seq.length >= opts.minSequenceLength) {
                    const dnaSeq = this.createDNASequence(seq, text, 'continuous');
                    if (this.isValidSequence(dnaSeq, opts).isValid) {
                        sequences.push(dnaSeq);
                        if (!usedMethods.includes('continuous')) {
                            usedMethods.push('continuous');
                        }
                    }
                }
            });
        }
        return {
            sequences: this.removeDuplicates(sequences),
            totalAtcgCount: atcgCount,
            messageLength: text.length,
            extractionMethods: usedMethods,
            processingTime: Date.now() - startTime
        };
    }
    /**
     * Sequential extraction: scan left-to-right, extract any A, T, C, G letters
     * Example: "Albert told Catherine George" → "ATCG"
     */
    extractSequentialATCG(text) {
        const cleanText = text.toUpperCase();
        let sequence = '';
        for (const char of cleanText) {
            if (['A', 'T', 'C', 'G'].includes(char)) {
                sequence += char;
            }
        }
        return sequence;
    }
    /**
     * Word-based extraction: extract first ATCG letter from each word
     * Example: "Apple Tree Cat Goat" → "ATCG"
     */
    extractWordBasedATCG(text) {
        const words = text.toUpperCase().split(/\s+/);
        let sequence = '';
        for (const word of words) {
            // Skip very short words and obvious non-biological content
            if (word.length < 2 || this.isNonBiological(word))
                continue;
            // Find first ATCG letter in this word
            for (const char of word) {
                if (['A', 'T', 'C', 'G'].includes(char)) {
                    sequence += char;
                    break; // Only take first ATCG letter from each word
                }
            }
        }
        return sequence;
    }
    /**
     * Continuous sequence detection: find existing ATCG sequences
     * Example: "Check out ATCGATCG sequence" → ["ATCGATCG"]
     */
    extractContinuousSequences(text) {
        const sequences = [];
        const cleanText = text.toUpperCase();
        // Look for continuous ATCG patterns (including IUPAC codes)
        const dnaPattern = /[ATCGWSMKRYBDHVN]{4,}/g;
        let match;
        while ((match = dnaPattern.exec(cleanText)) !== null) {
            const seq = match[0];
            // Filter out obvious false positives
            if (!this.isLikelyFalsePositive(seq)) {
                // Convert IUPAC codes to standard ATCG where possible
                const standardSeq = this.convertIUPACToStandard(seq);
                sequences.push(standardSeq);
            }
        }
        return sequences;
    }
    /**
     * Count total A, T, C, G letters in text for pre-filtering
     */
    countATCGLetters(text) {
        const cleanText = text.toUpperCase();
        let count = 0;
        for (const char of cleanText) {
            if (['A', 'T', 'C', 'G'].includes(char)) {
                count++;
            }
        }
        return count;
    }
    /**
     * Create a DNASequence object with calculated properties
     */
    createDNASequence(sequence, sourceText, method) {
        const cleaned = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        const gcContent = this.calculateGCContent(cleaned);
        const confidence = this.calculateExtractionConfidence(cleaned, sourceText, method);
        return {
            raw: sequence,
            cleaned,
            length: cleaned.length,
            gcContent,
            isValid: false, // Will be set by validation
            extractionMethod: method,
            sourceText: sourceText.substring(0, 200), // Limit source text for privacy
            confidence
        };
    }
    /**
     * Validate if a sequence is worth analyzing
     */
    isValidSequence(sequence, options) {
        const errors = [];
        const warnings = [];
        // Length check
        const lengthCheck = sequence.length >= options.minSequenceLength &&
            sequence.length <= options.maxSequenceLength;
        if (!lengthCheck) {
            errors.push(`Sequence length ${sequence.length} outside acceptable range`);
        }
        // GC content check
        const [minGC, maxGC] = options.gcContentRange;
        const gcContentCheck = sequence.gcContent >= minGC && sequence.gcContent <= maxGC;
        if (!gcContentCheck) {
            warnings.push(`GC content ${sequence.gcContent}% outside typical range`);
        }
        // Base composition check - ensure all 4 bases present for longer sequences
        const bases = new Set(sequence.cleaned.split(''));
        const baseCompositionCheck = sequence.length <= 12 || bases.size >= 3;
        if (!baseCompositionCheck) {
            warnings.push('Sequence lacks base diversity');
        }
        // Complexity check - avoid simple repeats
        const complexity = this.calculateComplexity(sequence.cleaned);
        const complexityCheck = complexity >= options.requiredComplexity;
        if (!complexityCheck) {
            warnings.push('Sequence appears to be repetitive');
        }
        // Repetitive pattern check
        const repetitiveCheck = !this.isRepetitive(sequence.cleaned);
        if (!repetitiveCheck) {
            warnings.push('Sequence contains repetitive patterns');
        }
        const score = this.calculateValidationScore({
            lengthCheck,
            gcContentCheck,
            baseCompositionCheck,
            complexityCheck,
            repetitiveCheck
        });
        return {
            isValid: errors.length === 0 && score >= 50,
            errors,
            warnings,
            score,
            details: {
                lengthCheck,
                gcContentCheck,
                baseCompositionCheck,
                complexityCheck,
                repetitiveCheck
            }
        };
    }
    /**
     * Calculate GC content percentage
     */
    calculateGCContent(sequence) {
        if (sequence.length === 0)
            return 0;
        const gcCount = (sequence.match(/[GC]/g) || []).length;
        return (gcCount / sequence.length) * 100;
    }
    /**
     * Calculate sequence complexity to avoid simple repeats
     */
    calculateComplexity(sequence) {
        if (sequence.length <= 1)
            return 0;
        const uniqueBigrams = new Set();
        for (let i = 0; i < sequence.length - 1; i++) {
            uniqueBigrams.add(sequence.substring(i, i + 2));
        }
        const maxPossibleBigrams = Math.min(16, sequence.length - 1); // 4^2 = 16 possible bigrams
        return uniqueBigrams.size / maxPossibleBigrams;
    }
    /**
     * Check if sequence is overly repetitive
     */
    isRepetitive(sequence) {
        if (sequence.length < 8)
            return false;
        // Check for simple repeats (AA, AAAA, etc.)
        const simpleRepeats = [
            /A{4,}/, /T{4,}/, /C{4,}/, /G{4,}/, // Single base repeats
            /(AT){3,}/, /(TA){3,}/, /(GC){3,}/, /(CG){3,}/ // Dinucleotide repeats
        ];
        return simpleRepeats.some(pattern => pattern.test(sequence));
    }
    /**
     * Calculate confidence score for extraction quality
     */
    calculateExtractionConfidence(sequence, sourceText, method) {
        let confidence = 0.5; // Base confidence
        // Method-based adjustments
        switch (method) {
            case 'continuous':
                confidence = 0.9; // Continuous sequences are most reliable
                break;
            case 'sequential':
                confidence = 0.6; // Sequential extraction is moderately reliable
                break;
            case 'word-based':
                confidence = 0.4; // Word-based is less reliable but creative
                break;
        }
        // Length adjustments
        if (sequence.length >= 20)
            confidence += 0.1;
        if (sequence.length >= 50)
            confidence += 0.1;
        if (sequence.length < 10)
            confidence -= 0.2;
        // Source context adjustments
        if (this.containsBiologicalKeywords(sourceText)) {
            confidence += 0.2;
        }
        return Math.max(0, Math.min(1, confidence));
    }
    /**
     * Check if source text contains biological keywords
     */
    containsBiologicalKeywords(text) {
        const biologicalTerms = [
            'dna', 'gene', 'sequence', 'nucleotide', 'genome', 'chromosome',
            'mutation', 'genetics', 'biology', 'species', 'organism', 'pcr',
            'blast', 'genbank', 'protein', 'amino', 'phylogeny'
        ];
        const lowerText = text.toLowerCase();
        return biologicalTerms.some(term => lowerText.includes(term));
    }
    /**
     * Check if word is obviously non-biological (URLs, codes, etc.)
     */
    isNonBiological(word) {
        const nonBiologicalPatterns = [
            /^https?:/i, // URLs
            /\d{3,}/, // Long numbers
            /[!@#$%^&*()]/, // Special characters
            /^[0-9A-F]{6,}$/i // Hex codes
        ];
        return nonBiologicalPatterns.some(pattern => pattern.test(word));
    }
    /**
     * Check if sequence is likely a false positive
     */
    isLikelyFalsePositive(sequence) {
        // Common false positives
        const falsePositives = [
            /^[ATCG]{1,3}$/, // Too short to be meaningful
            /AAAA+|TTTT+|CCCC+|GGGG+/, // Simple repeats
        ];
        return falsePositives.some(pattern => pattern.test(sequence));
    }
    /**
     * Apply character aliases to text before processing
     * Example: "ärgerlich täglich" → "aergerlich taeglich"
     */
    applyCharacterAliases(text) {
        let processedText = text;
        for (const [from, to] of this.characterAliases.entries()) {
            processedText = processedText.replace(new RegExp(from, 'g'), to);
        }
        return processedText;
    }
    /**
     * Convert IUPAC nucleotide codes to standard ATCG
     */
    convertIUPACToStandard(sequence) {
        const iupacMap = {
            'W': 'A', // A or T → choose A
            'S': 'C', // G or C → choose C
            'M': 'A', // A or C → choose A
            'K': 'G', // G or T → choose G
            'R': 'A', // A or G → choose A
            'Y': 'C', // C or T → choose C
            'B': 'C', // C, G or T → choose C
            'D': 'A', // A, G or T → choose A
            'H': 'A', // A, C or T → choose A
            'V': 'A', // A, C or G → choose A
            'N': 'A' // Any base → choose A
        };
        return sequence.replace(/[WSMKRYBDHVN]/g, match => iupacMap[match] || match);
    }
    /**
     * Set a character alias
     */
    setCharacterAlias(from, to) {
        this.characterAliases.set(from, to);
    }
    /**
     * Remove a character alias
     */
    removeCharacterAlias(from) {
        return this.characterAliases.delete(from);
    }
    /**
     * Get all character aliases
     */
    getCharacterAliases() {
        return new Map(this.characterAliases);
    }
    /**
     * Reset character aliases to defaults
     */
    resetCharacterAliases() {
        this.characterAliases = new Map([
            // German umlauts
            ['ä', 'ae'], ['ö', 'oe'], ['ü', 'ue'],
            ['Ä', 'AE'], ['Ö', 'OE'], ['Ü', 'UE'],
            // Accented vowels
            ['à', 'a'], ['á', 'a'], ['â', 'a'], ['ã', 'a'], ['å', 'a'],
            ['À', 'A'], ['Á', 'A'], ['Â', 'A'], ['Ã', 'A'], ['Å', 'A'],
            ['è', 'e'], ['é', 'e'], ['ê', 'e'], ['ë', 'e'],
            ['È', 'E'], ['É', 'E'], ['Ê', 'E'], ['Ë', 'E'],
            ['ì', 'i'], ['í', 'i'], ['î', 'i'], ['ï', 'i'],
            ['Ì', 'I'], ['Í', 'I'], ['Î', 'I'], ['Ï', 'I'],
            ['ò', 'o'], ['ó', 'o'], ['ô', 'o'], ['õ', 'o'],
            ['Ò', 'O'], ['Ó', 'O'], ['Ô', 'O'], ['Õ', 'O'],
            ['ù', 'u'], ['ú', 'u'], ['û', 'u'],
            ['Ù', 'U'], ['Ú', 'U'], ['Û', 'U'],
            // Other common characters
            ['ç', 'c'], ['Ç', 'C'],
            ['ñ', 'n'], ['Ñ', 'N'],
            ['ß', 'ss']
        ]);
    }
    /**
     * Remove duplicate sequences
     */
    removeDuplicates(sequences) {
        const seen = new Set();
        return sequences.filter(seq => {
            const key = `${seq.cleaned}-${seq.extractionMethod}`;
            if (seen.has(key)) {
                return false;
            }
            seen.add(key);
            return true;
        });
    }
    /**
     * Calculate overall validation score
     */
    calculateValidationScore(details) {
        const weights = {
            lengthCheck: 30,
            gcContentCheck: 20,
            baseCompositionCheck: 20,
            complexityCheck: 15,
            repetitiveCheck: 15
        };
        let score = 0;
        for (const [check, weight] of Object.entries(weights)) {
            if (details[check]) {
                score += weight;
            }
        }
        return score;
    }
}
//# sourceMappingURL=SequenceDetector.js.map