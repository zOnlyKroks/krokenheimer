import type { DNASequence, BlastResults, BlastRequest } from "../types/bioinformatics.js";
export declare class BlastApiClient {
    private readonly baseUrl;
    private readonly statusCheckInterval;
    private readonly maxWaitTime;
    private readonly maxRetries;
    /**
     * Submit a DNA sequence for BLAST analysis
     */
    submitSequence(sequence: DNASequence): Promise<string>;
    /**
     * Check the status of a BLAST job
     */
    checkStatus(rid: string): Promise<'WAITING' | 'READY' | 'UNKNOWN'>;
    /**
     * Retrieve BLAST results once ready
     */
    getResults(rid: string): Promise<BlastResults>;
    /**
     * Submit sequence and wait for results (convenience method)
     */
    analyzeSequence(sequence: DNASequence): Promise<BlastResults>;
    /**
     * Build POST parameters for BLAST submission
     */
    private buildBlastParams;
    /**
     * Extract RID from BLAST submission response
     */
    private extractRID;
    /**
     * Parse BLAST JSON results into our format
     */
    private parseBlastResults;
    /**
     * Parse individual BLAST hit
     */
    private parseBlastHit;
    /**
     * Extract scientific and common names from BLAST description
     */
    private extractSpeciesNames;
    /**
     * Calculate percent identity
     */
    private calculateIdentity;
    /**
     * Calculate query coverage
     */
    private calculateCoverage;
    /**
     * Utility method for delays
     */
    private sleep;
}
/**
 * Rate limiter for BLAST API calls
 */
export declare class BlastRateLimiter {
    private requestQueue;
    private isProcessing;
    private readonly maxRequestsPerMinute;
    private readonly requestWindowMs;
    private requestTimes;
    /**
     * Add a request to the rate-limited queue
     */
    queueRequest(request: BlastRequest): Promise<BlastResults>;
    /**
     * Process queued requests with rate limiting
     */
    private processQueue;
    /**
     * Wait if necessary to respect rate limits
     */
    private waitForRateLimit;
}
//# sourceMappingURL=BlastApiClient.d.ts.map