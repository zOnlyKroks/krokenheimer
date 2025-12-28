import type { SpeciesIdentification, DNASequence, MessageContext } from "../types/bioinformatics.js";
/**
 * In-memory cache for sequence analysis results
 */
export declare class SequenceCache {
    private cache;
    private readonly defaultTTL;
    private readonly maxCacheSize;
    private cleanupInterval;
    constructor();
    /**
     * Get cached result for a sequence
     */
    get(sequence: DNASequence): SpeciesIdentification | null;
    /**
     * Store analysis result in cache
     */
    set(sequence: DNASequence, result: SpeciesIdentification, ttl?: number): void;
    /**
     * Check if sequence exists in cache (without retrieving)
     */
    has(sequence: DNASequence): boolean;
    /**
     * Generate cache key for a sequence
     */
    private generateKey;
    /**
     * Remove expired entries and enforce size limits
     */
    private cleanup;
    /**
     * Evict oldest entries when cache is full
     */
    private evictOldestEntries;
    /**
     * Get cache statistics
     */
    getStats(): {
        size: number;
        hitCount: number;
        oldestEntry: number | null;
    };
    /**
     * Clear all cache entries
     */
    clear(): void;
    /**
     * Cleanup on shutdown
     */
    destroy(): void;
}
/**
 * User rate limiter to prevent spam
 */
export declare class UserRateLimiter {
    private userLimits;
    readonly requestsPerMinute = 3;
    private readonly windowMs;
    private readonly cleanupInterval;
    private cleanupTimer;
    constructor();
    /**
     * Check if user can make a request
     */
    canMakeRequest(userId: string): boolean;
    /**
     * Get time until user can make next request (in milliseconds)
     */
    getTimeUntilReset(userId: string): number;
    /**
     * Get current rate limit status for user
     */
    getLimitStatus(userId: string): {
        requestsUsed: number;
        requestsRemaining: number;
        resetTime: number;
    };
    /**
     * Manually reset a user's rate limit (admin function)
     */
    resetUserLimit(userId: string): void;
    /**
     * Clean up old rate limit entries
     */
    private cleanup;
    /**
     * Get rate limiter statistics
     */
    getStats(): {
        activeUsers: number;
        totalRequestsInWindow: number;
        averageRequestsPerUser: number;
    };
    /**
     * Cleanup on shutdown
     */
    destroy(): void;
}
/**
 * Combined service for cache and rate limiting management
 */
export declare class SequenceCacheManager {
    private cache;
    private rateLimiter;
    /**
     * Check if user can analyze a sequence (rate limit check)
     */
    canUserAnalyze(context: MessageContext): {
        allowed: boolean;
        reason?: string;
        resetTime?: number;
    };
    /**
     * Get cached result or null if not found/expired
     */
    getCachedResult(sequence: DNASequence): SpeciesIdentification | null;
    /**
     * Cache analysis result
     */
    cacheResult(sequence: DNASequence, result: SpeciesIdentification): void;
    /**
     * Check if sequence is in cache without retrieving
     */
    isInCache(sequence: DNASequence): boolean;
    /**
     * Get user rate limit status
     */
    getUserLimitStatus(userId: string): {
        requestsUsed: number;
        requestsRemaining: number;
        resetTime: number;
    };
    /**
     * Get comprehensive statistics
     */
    getStats(): {
        cache: {
            size: number;
            hitCount: number;
            oldestEntry: number | null;
        };
        rateLimit: {
            activeUsers: number;
            totalRequestsInWindow: number;
            averageRequestsPerUser: number;
        };
        timestamp: number;
    };
    /**
     * Admin functions for management
     */
    admin: {
        clearCache: () => void;
        resetUserLimit: (userId: string) => void;
        getStats: () => {
            cache: {
                size: number;
                hitCount: number;
                oldestEntry: number | null;
            };
            rateLimit: {
                activeUsers: number;
                totalRequestsInWindow: number;
                averageRequestsPerUser: number;
            };
            timestamp: number;
        };
    };
    /**
     * Cleanup resources
     */
    destroy(): void;
}
//# sourceMappingURL=SequenceCache.d.ts.map