/**
 * Service for fetching species images from various sources
 */
export class SpeciesImageService {
    private static readonly WIKIPEDIA_API_BASE = "https://en.wikipedia.org/w/api.php";
    private static readonly WIKIMEDIA_API_BASE = "https://commons.wikimedia.org/w/api.php";

    // Cache for image URLs to avoid repeated API calls
    private static readonly imageCache = new Map<string, string | null>();

    // Cache TTL: 24 hours
    private static readonly CACHE_TTL = 24 * 60 * 60 * 1000;
    private static readonly cacheTimestamps = new Map<string, number>();

    /**
     * Get species image URL for a given scientific name
     * @param scientificName The scientific name of the species (e.g., "Mus musculus")
     * @param commonName Optional common name for fallback search
     * @returns Image URL or null if not found
     */
    public static async getSpeciesImage(scientificName: string, commonName?: string): Promise<string | null> {
        // Normalize the scientific name for caching
        const cacheKey = scientificName.toLowerCase().trim();

        // Check cache first
        if (this.isCached(cacheKey)) {
            return this.imageCache.get(cacheKey) || null;
        }

        try {

            // Try Wikipedia first (most reliable for species pages)
            let imageUrl = await this.getImageFromWikipedia(scientificName);

            // If Wikipedia fails, try with common name
            if (!imageUrl && commonName) {
                imageUrl = await this.getImageFromWikipedia(commonName);
            }

            if (!imageUrl) {
                imageUrl = await this.getImageFromWikimediaCommons(scientificName);
            }

            // Cache the result (even if null)
            this.cacheImage(cacheKey, imageUrl);

            return imageUrl;

        } catch (error) {
            this.cacheImage(cacheKey, null);
            return null;
        }
    }

    /**
     * Get image from Wikipedia page
     */
    private static async getImageFromWikipedia(searchTerm: string): Promise<string | null> {
        try {
            // First, search for the page
            const searchUrl = new URL(this.WIKIPEDIA_API_BASE);
            searchUrl.searchParams.set('action', 'query');
            searchUrl.searchParams.set('format', 'json');
            searchUrl.searchParams.set('list', 'search');
            searchUrl.searchParams.set('srsearch', searchTerm);
            searchUrl.searchParams.set('srlimit', '1');
            searchUrl.searchParams.set('srnamespace', '0'); // Main namespace only
            searchUrl.searchParams.set('origin', '*'); // CORS

            const searchResponse = await fetch(searchUrl.toString(), {
                headers: { 'User-Agent': 'GenomeSequencerBot/1.0 (bioinformatics research)' }
            });

            if (!searchResponse.ok) {
                return null;
            }

            const searchData = await searchResponse.json();
            const searchResults = searchData.query?.search;

            if (!searchResults || searchResults.length === 0) {
                return null;
            }

            const pageTitle = searchResults[0].title;

            // Get page info including main image
            const pageUrl = new URL(this.WIKIPEDIA_API_BASE);
            pageUrl.searchParams.set('action', 'query');
            pageUrl.searchParams.set('format', 'json');
            pageUrl.searchParams.set('titles', pageTitle);
            pageUrl.searchParams.set('prop', 'pageimages|images');
            pageUrl.searchParams.set('pithumbsize', '300');
            pageUrl.searchParams.set('pilimit', '1');
            pageUrl.searchParams.set('imlimit', '5');
            pageUrl.searchParams.set('origin', '*'); // CORS

            const pageResponse = await fetch(pageUrl.toString(), {
                headers: { 'User-Agent': 'GenomeSequencerBot/1.0 (bioinformatics research)' }
            });

            if (!pageResponse.ok) {
                return null;
            }

            const pageData = await pageResponse.json();
            const pages = pageData.query?.pages;

            if (!pages) {
                return null;
            }

            const page = Object.values(pages)[0] as any;

            // Try to get the main page image (thumbnail)
            if (page.thumbnail?.source) {
                return page.thumbnail.source;
            }

            // If no main image, try to get the first suitable image from the page
            if (page.images && page.images.length > 0) {
                // Look for typical biological image files
                for (const image of page.images) {
                    const title = image.title.toLowerCase();
                    // Skip common non-biological images
                    if (title.includes('commons-logo') ||
                        title.includes('edit-icon') ||
                        title.includes('wikimedia') ||
                        title.includes('symbol') ||
                        title.includes('icon')) {
                        continue;
                    }

                    // Get image info
                    const imageUrl = await this.getImageUrlFromTitle(image.title);
                    if (imageUrl) {
                        return imageUrl;
                    }
                }
            }

            return null;

        } catch (error) {
            return null;
        }
    }

    /**
     * Get image URL from Wikimedia Commons
     */
    private static async getImageFromWikimediaCommons(scientificName: string): Promise<string | null> {
        try {
            // Search for images on Commons using scientific name
            const searchUrl = new URL(this.WIKIMEDIA_API_BASE);
            searchUrl.searchParams.set('action', 'query');
            searchUrl.searchParams.set('format', 'json');
            searchUrl.searchParams.set('list', 'search');
            searchUrl.searchParams.set('srsearch', `${scientificName} filetype:bitmap`);
            searchUrl.searchParams.set('srlimit', '5');
            searchUrl.searchParams.set('srnamespace', '6'); // File namespace
            searchUrl.searchParams.set('origin', '*'); // CORS

            const response = await fetch(searchUrl.toString(), {
                headers: { 'User-Agent': 'GenomeSequencerBot/1.0 (bioinformatics research)' }
            });

            if (!response.ok) {
                return null;
            }

            const data = await response.json();
            const searchResults = data.query?.search;

            if (!searchResults || searchResults.length === 0) {
                return null;
            }

            // Try to get URL for the first suitable image
            for (const result of searchResults) {
                const imageUrl = await this.getImageUrlFromTitle(result.title, true);
                if (imageUrl) {
                    return imageUrl;
                }
            }

            return null;

        } catch (error) {
            return null;
        }
    }

    /**
     * Get actual image URL from file title
     */
    private static async getImageUrlFromTitle(title: string, isCommons: boolean = false): Promise<string | null> {
        try {
            const apiBase = isCommons ? this.WIKIMEDIA_API_BASE : this.WIKIPEDIA_API_BASE;
            const url = new URL(apiBase);
            url.searchParams.set('action', 'query');
            url.searchParams.set('format', 'json');
            url.searchParams.set('titles', title);
            url.searchParams.set('prop', 'imageinfo');
            url.searchParams.set('iiprop', 'url|size');
            url.searchParams.set('iiurlwidth', '300');
            url.searchParams.set('origin', '*'); // CORS

            const response = await fetch(url.toString(), {
                headers: { 'User-Agent': 'GenomeSequencerBot/1.0 (bioinformatics research)' }
            });

            if (!response.ok) {
                return null;
            }

            const data = await response.json();
            const pages = data.query?.pages;

            if (!pages) {
                return null;
            }

            const page = Object.values(pages)[0] as any;
            const imageinfo = page.imageinfo?.[0];

            if (imageinfo) {
                // Prefer thumburl for smaller size, fallback to url
                return imageinfo.thumburl || imageinfo.url || null;
            }

            return null;

        } catch (error) {
            return null;
        }
    }

    /**
     * Check if image is cached and not expired
     */
    private static isCached(cacheKey: string): boolean {
        const timestamp = this.cacheTimestamps.get(cacheKey);
        if (!timestamp) {
            return false;
        }

        const now = Date.now();
        if (now - timestamp > this.CACHE_TTL) {
            // Cache expired, remove it
            this.imageCache.delete(cacheKey);
            this.cacheTimestamps.delete(cacheKey);
            return false;
        }

        return this.imageCache.has(cacheKey);
    }

    /**
     * Cache image URL with timestamp
     */
    private static cacheImage(cacheKey: string, imageUrl: string | null): void {
        this.imageCache.set(cacheKey, imageUrl);
        this.cacheTimestamps.set(cacheKey, Date.now());

        // Clean up old cache entries if cache gets too large
        if (this.imageCache.size > 1000) {
            this.cleanupCache();
        }
    }

    /**
     * Clean up old cache entries
     */
    private static cleanupCache(): void {
        const now = Date.now();
        const keysToDelete: string[] = [];

        for (const [key, timestamp] of this.cacheTimestamps.entries()) {
            if (now - timestamp > this.CACHE_TTL) {
                keysToDelete.push(key);
            }
        }

        for (const key of keysToDelete) {
            this.imageCache.delete(key);
            this.cacheTimestamps.delete(key);
        }
    }

    /**
     * Clear all cached images (useful for testing or maintenance)
     */
    public static clearCache(): void {
        this.imageCache.clear();
        this.cacheTimestamps.clear();
    }

    /**
     * Get cache statistics
     */
    public static getCacheStats(): { size: number; entries: number } {
        return {
            size: this.imageCache.size,
            entries: this.cacheTimestamps.size
        };
    }
}