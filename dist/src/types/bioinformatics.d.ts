export interface DNASequence {
    raw: string;
    cleaned: string;
    length: number;
    gcContent: number;
    isValid: boolean;
    extractionMethod: 'sequential' | 'word-based' | 'continuous' | 'hybrid';
    sourceText: string;
    confidence: number;
}
export interface BlastHit {
    accession: string;
    description: string;
    scientificName: string;
    commonName?: string;
    eValue: number;
    bitScore: number;
    identity: number;
    coverage: number;
    alignmentLength: number;
    taxonId?: number;
}
export interface BlastResults {
    requestId: string;
    querySequence: string;
    queryLength: number;
    database: string;
    program: string;
    hits: BlastHit[];
    timestamp: number;
    executionTime: number;
    status: 'pending' | 'running' | 'completed' | 'failed';
}
export interface SpeciesIdentification {
    sequence: DNASequence;
    blastResults?: BlastResults;
    topMatches: SpeciesMatch[];
    confidence: ConfidenceLevel;
    processingTime: number;
    cacheHit: boolean;
}
export interface SpeciesMatch {
    species: string;
    commonName?: string;
    confidence: number;
    identity: number;
    eValue: number;
    description: string;
    taxonId?: number;
}
export interface ConfidenceLevel {
    overall: number;
    extractionQuality: number;
    blastReliability: number;
    sequenceValidity: number;
    level: 'very-low' | 'low' | 'medium' | 'high' | 'very-high';
}
export interface ExtractionResult {
    sequences: DNASequence[];
    totalAtcgCount: number;
    messageLength: number;
    extractionMethods: string[];
    processingTime: number;
}
export interface CacheEntry {
    key: string;
    result: SpeciesIdentification;
    timestamp: number;
    expiresAt: number;
    hitCount: number;
}
export interface RateLimitInfo {
    userId: string;
    requestCount: number;
    windowStart: number;
    lastRequest: number;
}
export interface BlastRequest {
    sequence: DNASequence;
    userId: string;
    messageId: string;
    timestamp: number;
    priority: number;
}
export interface ValidationResult {
    isValid: boolean;
    errors: string[];
    warnings: string[];
    score: number;
    details: {
        lengthCheck: boolean;
        gcContentCheck: boolean;
        complexityCheck: boolean;
        baseCompositionCheck: boolean;
        repetitiveCheck: boolean;
    };
}
export interface MessageContext {
    userId: string;
    channelId: string;
    guildId?: string;
    messageId: string;
    timestamp: number;
}
export interface AnalysisOptions {
    minSequenceLength: number;
    maxSequenceLength: number;
    enableCaching: boolean;
    enableRateLimit: boolean;
    extractionMethods: Array<'sequential' | 'word-based' | 'continuous' | 'hybrid'>;
    gcContentRange: [number, number];
    requiredComplexity: number;
}
//# sourceMappingURL=bioinformatics.d.ts.map