import crypto from "crypto";
/**
 * In-memory cache for sequence analysis results
 */
export class SequenceCache {
    cache = new Map();
    defaultTTL = 24 * 60 * 60 * 1000; // 24 hours
    maxCacheSize = 1000; // Maximum number of cached entries
    cleanupInterval;
    constructor() {
        // Run cleanup every hour
        this.cleanupInterval = setInterval(() => this.cleanup(), 60 * 60 * 1000);
    }
    /**
     * Get cached result for a sequence
     */
    get(sequence) {
        const key = this.generateKey(sequence);
        const entry = this.cache.get(key);
        if (!entry) {
            return null;
        }
        // Check if entry has expired
        if (Date.now() > entry.expiresAt) {
            this.cache.delete(key);
            return null;
        }
        // Update hit count and access time
        entry.hitCount++;
        entry.timestamp = Date.now();
        return {
            ...entry.result,
            cacheHit: true
        };
    }
    /**
     * Store analysis result in cache
     */
    set(sequence, result, ttl) {
        const key = this.generateKey(sequence);
        const actualTTL = ttl || this.defaultTTL;
        // Ensure we don't mark this as a cache hit when storing
        const resultToStore = { ...result, cacheHit: false };
        const entry = {
            key,
            result: resultToStore,
            timestamp: Date.now(),
            expiresAt: Date.now() + actualTTL,
            hitCount: 0
        };
        this.cache.set(key, entry);
        // Enforce cache size limit
        if (this.cache.size > this.maxCacheSize) {
            this.evictOldestEntries(this.maxCacheSize * 0.1); // Remove 10% of entries
        }
    }
    /**
     * Check if sequence exists in cache (without retrieving)
     */
    has(sequence) {
        const key = this.generateKey(sequence);
        const entry = this.cache.get(key);
        if (!entry) {
            return false;
        }
        // Check if expired
        if (Date.now() > entry.expiresAt) {
            this.cache.delete(key);
            return false;
        }
        return true;
    }
    /**
     * Generate cache key for a sequence
     */
    generateKey(sequence) {
        // Create hash based on cleaned sequence and extraction method
        const data = `${sequence.cleaned}-${sequence.extractionMethod}`;
        return crypto.createHash('sha256').update(data).digest('hex');
    }
    /**
     * Remove expired entries and enforce size limits
     */
    cleanup() {
        const now = Date.now();
        const expiredKeys = [];
        // Find expired entries
        for (const [key, entry] of this.cache.entries()) {
            if (now > entry.expiresAt) {
                expiredKeys.push(key);
            }
        }
        // Remove expired entries
        for (const key of expiredKeys) {
            this.cache.delete(key);
        }
        console.log(`Cache cleanup: removed ${expiredKeys.length} expired entries, ${this.cache.size} entries remaining`);
    }
    /**
     * Evict oldest entries when cache is full
     */
    evictOldestEntries(count) {
        const entries = Array.from(this.cache.entries())
            .sort(([, a], [, b]) => a.timestamp - b.timestamp)
            .slice(0, count);
        for (const [key] of entries) {
            this.cache.delete(key);
        }
    }
    /**
     * Get cache statistics
     */
    getStats() {
        let totalHits = 0;
        let oldestTimestamp = null;
        for (const entry of this.cache.values()) {
            totalHits += entry.hitCount;
            if (oldestTimestamp === null || entry.timestamp < oldestTimestamp) {
                oldestTimestamp = entry.timestamp;
            }
        }
        return {
            size: this.cache.size,
            hitCount: totalHits,
            oldestEntry: oldestTimestamp
        };
    }
    /**
     * Clear all cache entries
     */
    clear() {
        this.cache.clear();
    }
    /**
     * Cleanup on shutdown
     */
    destroy() {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
        }
        this.clear();
    }
}
/**
 * User rate limiter to prevent spam
 */
export class UserRateLimiter {
    userLimits = new Map();
    requestsPerMinute = 3; // Max 3 analysis requests per user per minute
    windowMs = 60000; // 1 minute window
    cleanupInterval = 5 * 60 * 1000; // Cleanup every 5 minutes
    cleanupTimer;
    constructor() {
        this.cleanupTimer = setInterval(() => this.cleanup(), this.cleanupInterval);
    }
    /**
     * Check if user can make a request
     */
    canMakeRequest(userId) {
        const now = Date.now();
        const userLimit = this.userLimits.get(userId);
        if (!userLimit) {
            // First request for this user
            this.userLimits.set(userId, {
                userId,
                requestCount: 1,
                windowStart: now,
                lastRequest: now
            });
            return true;
        }
        // Check if we're in a new window
        if (now - userLimit.windowStart >= this.windowMs) {
            // Reset window
            userLimit.requestCount = 1;
            userLimit.windowStart = now;
            userLimit.lastRequest = now;
            return true;
        }
        // Check if under limit
        if (userLimit.requestCount < this.requestsPerMinute) {
            userLimit.requestCount++;
            userLimit.lastRequest = now;
            return true;
        }
        // Rate limited
        return false;
    }
    /**
     * Get time until user can make next request (in milliseconds)
     */
    getTimeUntilReset(userId) {
        const userLimit = this.userLimits.get(userId);
        if (!userLimit) {
            return 0; // No limit set, can make request immediately
        }
        const timeInWindow = Date.now() - userLimit.windowStart;
        const timeLeft = this.windowMs - timeInWindow;
        return Math.max(0, timeLeft);
    }
    /**
     * Get current rate limit status for user
     */
    getLimitStatus(userId) {
        const userLimit = this.userLimits.get(userId);
        if (!userLimit) {
            return {
                requestsUsed: 0,
                requestsRemaining: this.requestsPerMinute,
                resetTime: 0
            };
        }
        const now = Date.now();
        const timeInWindow = now - userLimit.windowStart;
        // If window has expired, reset
        if (timeInWindow >= this.windowMs) {
            return {
                requestsUsed: 0,
                requestsRemaining: this.requestsPerMinute,
                resetTime: 0
            };
        }
        return {
            requestsUsed: userLimit.requestCount,
            requestsRemaining: Math.max(0, this.requestsPerMinute - userLimit.requestCount),
            resetTime: userLimit.windowStart + this.windowMs
        };
    }
    /**
     * Manually reset a user's rate limit (admin function)
     */
    resetUserLimit(userId) {
        this.userLimits.delete(userId);
    }
    /**
     * Clean up old rate limit entries
     */
    cleanup() {
        const now = Date.now();
        const expiredUsers = [];
        for (const [userId, limitInfo] of this.userLimits.entries()) {
            // Remove entries that are older than 2 windows
            if (now - limitInfo.lastRequest > this.windowMs * 2) {
                expiredUsers.push(userId);
            }
        }
        for (const userId of expiredUsers) {
            this.userLimits.delete(userId);
        }
        if (expiredUsers.length > 0) {
            console.log(`Rate limiter cleanup: removed ${expiredUsers.length} expired user entries`);
        }
    }
    /**
     * Get rate limiter statistics
     */
    getStats() {
        const now = Date.now();
        let totalRequests = 0;
        let activeUsers = 0;
        for (const limitInfo of this.userLimits.values()) {
            const timeInWindow = now - limitInfo.windowStart;
            if (timeInWindow < this.windowMs) {
                activeUsers++;
                totalRequests += limitInfo.requestCount;
            }
        }
        return {
            activeUsers,
            totalRequestsInWindow: totalRequests,
            averageRequestsPerUser: activeUsers > 0 ? totalRequests / activeUsers : 0
        };
    }
    /**
     * Cleanup on shutdown
     */
    destroy() {
        if (this.cleanupTimer) {
            clearInterval(this.cleanupTimer);
        }
        this.userLimits.clear();
    }
}
/**
 * Combined service for cache and rate limiting management
 */
export class SequenceCacheManager {
    cache = new SequenceCache();
    rateLimiter = new UserRateLimiter();
    /**
     * Check if user can analyze a sequence (rate limit check)
     */
    canUserAnalyze(context) {
        const canMakeRequest = this.rateLimiter.canMakeRequest(context.userId);
        if (!canMakeRequest) {
            const resetTime = this.rateLimiter.getTimeUntilReset(context.userId);
            return {
                allowed: false,
                reason: `Rate limited. You can analyze ${this.rateLimiter.requestsPerMinute} sequences per minute.`,
                resetTime
            };
        }
        return { allowed: true };
    }
    /**
     * Get cached result or null if not found/expired
     */
    getCachedResult(sequence) {
        return this.cache.get(sequence);
    }
    /**
     * Cache analysis result
     */
    cacheResult(sequence, result) {
        this.cache.set(sequence, result);
    }
    /**
     * Check if sequence is in cache without retrieving
     */
    isInCache(sequence) {
        return this.cache.has(sequence);
    }
    /**
     * Get user rate limit status
     */
    getUserLimitStatus(userId) {
        return this.rateLimiter.getLimitStatus(userId);
    }
    /**
     * Get comprehensive statistics
     */
    getStats() {
        const cacheStats = this.cache.getStats();
        const rateLimitStats = this.rateLimiter.getStats();
        return {
            cache: cacheStats,
            rateLimit: rateLimitStats,
            timestamp: Date.now()
        };
    }
    /**
     * Admin functions for management
     */
    admin = {
        clearCache: () => this.cache.clear(),
        resetUserLimit: (userId) => this.rateLimiter.resetUserLimit(userId),
        getStats: () => this.getStats()
    };
    /**
     * Cleanup resources
     */
    destroy() {
        this.cache.destroy();
        this.rateLimiter.destroy();
    }
}
//# sourceMappingURL=SequenceCache.js.map