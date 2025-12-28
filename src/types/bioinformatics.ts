export interface DNASequence {
  raw: string;                    // Original extracted sequence
  cleaned: string;                // Processed sequence (uppercase, only ATCG)
  length: number;                 // Number of nucleotides
  gcContent: number;              // GC content percentage (0-100)
  isValid: boolean;               // Whether sequence passes validation
  extractionMethod: 'sequential' | 'word-based' | 'continuous' | 'hybrid';
  sourceText: string;             // Original message text
  confidence: number;             // Confidence score (0-1) based on extraction quality
}

export interface BlastHit {
  accession: string;              // GenBank accession number
  description: string;            // Full description from BLAST
  scientificName: string;         // Species scientific name
  commonName?: string;            // Common species name (if available)
  eValue: number;                 // Expect value (significance)
  bitScore: number;               // Bit score
  identity: number;               // Percent identity (0-100)
  coverage: number;               // Query coverage percentage (0-100)
  alignmentLength: number;        // Length of alignment
  taxonId?: number;               // NCBI taxonomy ID
}

export interface BlastResults {
  requestId: string;              // BLAST RID (Request ID)
  querySequence: string;          // Input sequence sent to BLAST
  queryLength: number;            // Length of query sequence
  database: string;               // Database searched (e.g., "nt")
  program: string;                // BLAST program used (e.g., "blastn")
  hits: BlastHit[];               // Array of matching results
  timestamp: number;              // When results were obtained
  executionTime: number;          // Time taken for BLAST search (milliseconds)
  status: 'pending' | 'running' | 'completed' | 'failed';
}

export interface SpeciesIdentification {
  sequence: DNASequence;          // The analyzed sequence
  blastResults?: BlastResults;    // BLAST results (optional if failed)
  topMatches: SpeciesMatch[];     // Processed top species matches
  confidence: ConfidenceLevel;    // Overall confidence assessment
  processingTime: number;         // Total time for analysis (milliseconds)
}

export interface SpeciesMatch {
  species: string;                // Scientific name
  commonName?: string;            // Common name
  confidence: number;             // Confidence score (0-100)
  identity: number;               // Percent identity
  eValue: number;                 // Expect value
  description: string;            // Human-readable description
  taxonId?: number;               // NCBI taxonomy ID
}

export interface ConfidenceLevel {
  overall: number;                // Overall confidence (0-100)
  extractionQuality: number;      // Quality of sequence extraction (0-100)
  blastReliability: number;       // Reliability of BLAST results (0-100)
  sequenceValidity: number;       // How "DNA-like" the sequence is (0-100)
  level: 'very-low' | 'low' | 'medium' | 'high' | 'very-high';
}

export interface ExtractionResult {
  sequences: DNASequence[];       // All extracted sequences
  totalAtcgCount: number;         // Total A,T,C,G letters found
  messageLength: number;          // Original message length
  extractionMethods: string[];    // Methods that found sequences
  processingTime: number;         // Time taken for extraction (milliseconds)
}

export interface CacheEntry {
  key: string;                    // Hash of the sequence
  result: SpeciesIdentification;  // Cached result
  timestamp: number;              // When cached
  expiresAt: number;              // Expiration timestamp
  hitCount: number;               // Number of times accessed
}

export interface RateLimitInfo {
  userId: string;                 // Discord user ID
  requestCount: number;           // Current request count
  windowStart: number;            // Rate limit window start time
  lastRequest: number;            // Last request timestamp
}

export interface BlastRequest {
  sequence: DNASequence;          // Sequence to analyze
  userId: string;                 // Discord user ID for rate limiting
  messageId: string;              // Discord message ID
  timestamp: number;              // Request timestamp
  priority: number;               // Request priority (1-10)
}

export interface ValidationResult {
  isValid: boolean;               // Whether sequence passes all checks
  errors: string[];               // Validation error messages
  warnings: string[];             // Validation warnings
  score: number;                  // Validation score (0-100)
  details: {
    lengthCheck: boolean;         // Minimum length requirement
    gcContentCheck: boolean;      // GC content within acceptable range
    complexityCheck: boolean;     // Sufficient sequence complexity
    baseCompositionCheck: boolean; // Contains varied bases
    repetitiveCheck: boolean;     // Not overly repetitive
  };
}

// Utility types for Discord integration
export interface MessageContext {
  userId: string;                 // Discord user ID
  channelId: string;              // Discord channel ID
  guildId?: string;               // Discord guild ID (if in server)
  messageId: string;              // Discord message ID
  timestamp: number;              // Message timestamp
}

export interface AnalysisOptions {
  minSequenceLength: number;      // Minimum sequence length to analyze
  maxSequenceLength: number;      // Maximum sequence length to analyze
  enableCaching: boolean;         // Whether to use caching
  enableRateLimit: boolean;       // Whether to enforce rate limits
  extractionMethods: Array<'sequential' | 'word-based' | 'continuous' | 'hybrid'>;
  gcContentRange: [number, number]; // Acceptable GC content range
  requiredComplexity: number;     // Minimum complexity score
}