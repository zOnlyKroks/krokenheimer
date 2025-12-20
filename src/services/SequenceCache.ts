import crypto from "crypto";
import type {
  SpeciesIdentification,
  DNASequence,
  CacheEntry,
  RateLimitInfo,
  MessageContext
} from "../types/bioinformatics.js";

/**
 * In-memory cache for sequence analysis results
 */
export class SequenceCache {
  private cache = new Map<string, CacheEntry>();
  private readonly defaultTTL = 24 * 60 * 60 * 1000; // 24 hours
  private readonly maxCacheSize = 1000; // Maximum number of cached entries
  private readonly cleanupInterval: NodeJS.Timeout;

  constructor() {
    // Run cleanup every hour
    this.cleanupInterval = setInterval(() => this.cleanup(), 60 * 60 * 1000);
  }

  /**
   * Get cached result for a sequence
   */
  public get(sequence: DNASequence): SpeciesIdentification | null {
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
  public set(sequence: DNASequence, result: SpeciesIdentification, ttl?: number): void {
    const key = this.generateKey(sequence);
    const actualTTL = ttl || this.defaultTTL;

    // Ensure we don't mark this as a cache hit when storing
    const resultToStore = { ...result, cacheHit: false };

    const entry: CacheEntry = {
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
  public has(sequence: DNASequence): boolean {
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
  private generateKey(sequence: DNASequence): string {
    // Create hash based on cleaned sequence and extraction method
    const data = `${sequence.cleaned}-${sequence.extractionMethod}`;
    return crypto.createHash('sha256').update(data).digest('hex');
  }

  /**
   * Remove expired entries and enforce size limits
   */
  private cleanup(): void {
    const now = Date.now();
    const expiredKeys: string[] = [];

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
  private evictOldestEntries(count: number): void {
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
  public getStats(): { size: number; hitCount: number; oldestEntry: number | null } {
    let totalHits = 0;
    let oldestTimestamp: number | null = null;

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
  public clear(): void {
    this.cache.clear();
  }

  /**
   * Cleanup on shutdown
   */
  public destroy(): void {
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
  private userLimits = new Map<string, RateLimitInfo>();
  public readonly requestsPerMinute = 3; // Max 3 analysis requests per user per minute
  private readonly windowMs = 60000; // 1 minute window
  private readonly cleanupInterval = 5 * 60 * 1000; // Cleanup every 5 minutes
  private cleanupTimer: NodeJS.Timeout;

  constructor() {
    this.cleanupTimer = setInterval(() => this.cleanup(), this.cleanupInterval);
  }

  /**
   * Check if user can make a request
   */
  public canMakeRequest(userId: string): boolean {
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
  public getTimeUntilReset(userId: string): number {
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
  public getLimitStatus(userId: string): {
    requestsUsed: number;
    requestsRemaining: number;
    resetTime: number;
  } {
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
  public resetUserLimit(userId: string): void {
    this.userLimits.delete(userId);
  }

  /**
   * Clean up old rate limit entries
   */
  private cleanup(): void {
    const now = Date.now();
    const expiredUsers: string[] = [];

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
  public getStats(): {
    activeUsers: number;
    totalRequestsInWindow: number;
    averageRequestsPerUser: number;
  } {
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
  public destroy(): void {
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
  private cache = new SequenceCache();
  private rateLimiter = new UserRateLimiter();

  /**
   * Check if user can analyze a sequence (rate limit check)
   */
  public canUserAnalyze(context: MessageContext): {
    allowed: boolean;
    reason?: string;
    resetTime?: number;
  } {
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
  public getCachedResult(sequence: DNASequence): SpeciesIdentification | null {
    return this.cache.get(sequence);
  }

  /**
   * Cache analysis result
   */
  public cacheResult(sequence: DNASequence, result: SpeciesIdentification): void {
    this.cache.set(sequence, result);
  }

  /**
   * Check if sequence is in cache without retrieving
   */
  public isInCache(sequence: DNASequence): boolean {
    return this.cache.has(sequence);
  }

  /**
   * Get user rate limit status
   */
  public getUserLimitStatus(userId: string) {
    return this.rateLimiter.getLimitStatus(userId);
  }

  /**
   * Get comprehensive statistics
   */
  public getStats() {
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
  public admin = {
    clearCache: () => this.cache.clear(),
    resetUserLimit: (userId: string) => this.rateLimiter.resetUserLimit(userId),
    getStats: () => this.getStats()
  };

  /**
   * Cleanup resources
   */
  public destroy(): void {
    this.cache.destroy();
    this.rateLimiter.destroy();
  }
}