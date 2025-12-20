import fetch from "node-fetch";
export class BlastApiClient {
    baseUrl = "https://blast.ncbi.nlm.nih.gov/Blast.cgi";
    statusCheckInterval = 5000; // 5 seconds between status checks
    maxWaitTime = 300000; // 5 minutes max wait
    maxRetries = 3;
    /**
     * Submit a DNA sequence for BLAST analysis
     */
    async submitSequence(sequence) {
        const params = this.buildBlastParams(sequence.cleaned);
        for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
            try {
                const response = await fetch(this.baseUrl, {
                    method: 'POST',
                    body: params,
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'User-Agent': 'Discord-Bot-Genome-Sequencer/1.0 (contact: your-email@example.com)'
                    }
                });
                if (!response.ok) {
                    throw new Error(`BLAST submission failed: ${response.status} ${response.statusText}`);
                }
                const result = await response.text();
                const rid = this.extractRID(result);
                if (!rid) {
                    throw new Error('Failed to extract RID from BLAST response');
                }
                return rid;
            }
            catch (error) {
                if (attempt === this.maxRetries) {
                    throw new Error(`BLAST submission failed after ${this.maxRetries} attempts: ${error}`);
                }
                // Exponential backoff
                const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
                await this.sleep(delay);
            }
        }
        throw new Error('BLAST submission failed');
    }
    /**
     * Check the status of a BLAST job
     */
    async checkStatus(rid) {
        try {
            const params = new URLSearchParams({
                CMD: 'Get',
                RID: rid,
                FORMAT_TYPE: 'XML'
            });
            const response = await fetch(`${this.baseUrl}?${params}`, {
                headers: {
                    'User-Agent': 'Discord-Bot-Genome-Sequencer/1.0 (contact: your-email@example.com)'
                }
            });
            if (!response.ok) {
                return 'UNKNOWN';
            }
            const result = await response.text();
            if (result.includes('Status=WAITING')) {
                return 'WAITING';
            }
            else if (result.includes('Status=READY')) {
                return 'READY';
            }
            else {
                return 'UNKNOWN';
            }
        }
        catch (error) {
            console.error('Failed to check BLAST status:', error);
            return 'UNKNOWN';
        }
    }
    /**
     * Retrieve BLAST results once ready
     */
    async getResults(rid) {
        const params = new URLSearchParams({
            CMD: 'Get',
            RID: rid,
            FORMAT_TYPE: 'JSON2'
        });
        const response = await fetch(`${this.baseUrl}?${params}`, {
            headers: {
                'User-Agent': 'Discord-Bot-Genome-Sequencer/1.0 (contact: your-email@example.com)'
            }
        });
        if (!response.ok) {
            throw new Error(`Failed to retrieve BLAST results: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        return this.parseBlastResults(rid, data);
    }
    /**
     * Submit sequence and wait for results (convenience method)
     */
    async analyzeSequence(sequence) {
        const startTime = Date.now();
        // Submit the BLAST job
        const rid = await this.submitSequence(sequence);
        // Poll for completion
        const endTime = startTime + this.maxWaitTime;
        let status = 'WAITING';
        while (Date.now() < endTime && status === 'WAITING') {
            await this.sleep(this.statusCheckInterval);
            status = await this.checkStatus(rid);
        }
        if (status !== 'READY') {
            throw new Error(`BLAST job did not complete within ${this.maxWaitTime / 1000} seconds`);
        }
        // Retrieve results
        const results = await this.getResults(rid);
        results.executionTime = Date.now() - startTime;
        return results;
    }
    /**
     * Build POST parameters for BLAST submission
     */
    buildBlastParams(sequence) {
        const params = new URLSearchParams({
            CMD: 'Put',
            PROGRAM: 'blastn', // Nucleotide BLAST
            DATABASE: 'nt', // Nucleotide collection
            QUERY: sequence,
            FORMAT_TYPE: 'JSON2', // JSON format for easier parsing
            EXPECT: '0.001', // Stricter e-value threshold
            HITLIST_SIZE: '10', // Top 10 hits
            ALIGNMENTS: '10', // Show alignments for top 10
            DESCRIPTIONS: '10', // Descriptions for top 10
            AUTO_FORMAT: 'Semi', // Semi-automatic formatting
            COMPOSITION_BASED_STATISTICS: 'yes',
            FILTER: 'L', // Low complexity filter
            GAPCOSTS: '5 2', // Gap opening and extension costs
            WORD_SIZE: '11', // Word size for initial matches
            MATCH_REWARD: '2', // Match reward
            MISMATCH_PENALTY: '-3' // Mismatch penalty
        });
        return params.toString();
    }
    /**
     * Extract RID from BLAST submission response
     */
    extractRID(response) {
        // Look for RID in the response
        const ridMatch = response.match(/RID\s*=\s*([A-Z0-9-]+)/i);
        return ridMatch && ridMatch[1] ? ridMatch[1] : null;
    }
    /**
     * Parse BLAST JSON results into our format
     */
    parseBlastResults(rid, data) {
        const results = {
            requestId: rid,
            querySequence: '',
            queryLength: 0,
            database: 'nt',
            program: 'blastn',
            hits: [],
            timestamp: Date.now(),
            executionTime: 0,
            status: 'completed'
        };
        try {
            // Navigate the BLAST JSON structure
            const blastOutput = data.BlastOutput2?.[0]?.report;
            if (!blastOutput) {
                throw new Error('Invalid BLAST output format');
            }
            const search = blastOutput.results?.search;
            if (!search) {
                throw new Error('No search results found');
            }
            results.querySequence = search.query_title || '';
            results.queryLength = search.query_len || 0;
            // Extract hits
            const hits = search.hits || [];
            results.hits = hits.slice(0, 10).map((hit) => this.parseBlastHit(hit))
                .filter((hit) => hit !== null);
        }
        catch (error) {
            console.error('Failed to parse BLAST results:', error);
            results.status = 'failed';
        }
        return results;
    }
    /**
     * Parse individual BLAST hit
     */
    parseBlastHit(hitData) {
        try {
            const description = hitData.description?.[0];
            const hsps = hitData.hsps?.[0]; // High-scoring segment pairs
            if (!description || !hsps) {
                return null;
            }
            // Extract species name from description
            const fullDescription = description.title || '';
            const { scientificName, commonName } = this.extractSpeciesNames(fullDescription);
            return {
                accession: description.accession || 'Unknown',
                description: fullDescription,
                scientificName,
                commonName,
                eValue: hsps.evalue || 1.0,
                bitScore: hsps.bit_score || 0,
                identity: this.calculateIdentity(hsps.identities, hsps.align_len),
                coverage: this.calculateCoverage(hsps.align_len, hsps.query_from, hsps.query_to),
                alignmentLength: hsps.align_len || 0,
                taxonId: description.taxid
            };
        }
        catch (error) {
            console.error('Failed to parse BLAST hit:', error);
            return null;
        }
    }
    /**
     * Extract scientific and common names from BLAST description
     */
    extractSpeciesNames(description) {
        // BLAST descriptions typically format: "Species name [Common name] gene/description"
        const speciesMatch = description.match(/^([A-Z][a-z]+ [a-z]+)/);
        const commonNameMatch = description.match(/\[([^\]]+)\]/);
        const scientificName = (speciesMatch && speciesMatch[1]) ? speciesMatch[1] : 'Unknown species';
        let commonName = (commonNameMatch && commonNameMatch[1]) ? commonNameMatch[1] : undefined;
        // Clean up common name if it's too generic
        if (commonName && (commonName.length < 3 || commonName.toLowerCase().includes('gene'))) {
            commonName = undefined;
        }
        return { scientificName, commonName };
    }
    /**
     * Calculate percent identity
     */
    calculateIdentity(identities, alignmentLength) {
        if (alignmentLength === 0)
            return 0;
        return (identities / alignmentLength) * 100;
    }
    /**
     * Calculate query coverage
     */
    calculateCoverage(alignmentLength, queryFrom, queryTo) {
        const queryLength = Math.abs(queryTo - queryFrom) + 1;
        if (queryLength === 0)
            return 0;
        return (alignmentLength / queryLength) * 100;
    }
    /**
     * Utility method for delays
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
/**
 * Rate limiter for BLAST API calls
 */
export class BlastRateLimiter {
    requestQueue = [];
    isProcessing = false;
    maxRequestsPerMinute = 6; // NCBI allows ~10 seconds between requests
    requestWindowMs = 60000; // 1 minute
    requestTimes = [];
    /**
     * Add a request to the rate-limited queue
     */
    async queueRequest(request) {
        return new Promise((resolve, reject) => {
            this.requestQueue.push({
                ...request,
                resolve,
                reject
            });
            this.processQueue();
        });
    }
    /**
     * Process queued requests with rate limiting
     */
    async processQueue() {
        if (this.isProcessing || this.requestQueue.length === 0) {
            return;
        }
        this.isProcessing = true;
        while (this.requestQueue.length > 0) {
            // Check if we're within rate limits
            await this.waitForRateLimit();
            const request = this.requestQueue.shift();
            const blastClient = new BlastApiClient();
            try {
                const result = await blastClient.analyzeSequence(request.sequence);
                request.resolve(result);
            }
            catch (error) {
                request.reject(error);
            }
            // Record the request time
            this.requestTimes.push(Date.now());
        }
        this.isProcessing = false;
    }
    /**
     * Wait if necessary to respect rate limits
     */
    async waitForRateLimit() {
        const now = Date.now();
        // Remove old request times outside the window
        this.requestTimes = this.requestTimes.filter(time => now - time < this.requestWindowMs);
        // If we've hit the limit, wait
        if (this.requestTimes.length >= this.maxRequestsPerMinute) {
            const oldestRequest = this.requestTimes[0];
            if (oldestRequest !== undefined) {
                const waitTime = this.requestWindowMs - (now - oldestRequest) + 1000; // Add 1 second buffer
                if (waitTime > 0) {
                    await new Promise(resolve => setTimeout(resolve, waitTime));
                }
            }
        }
        // Add minimum delay between requests (NCBI recommends 10+ seconds)
        if (this.requestTimes.length > 0) {
            const lastRequest = this.requestTimes[this.requestTimes.length - 1];
            if (lastRequest !== undefined) {
                const timeSinceLastRequest = now - lastRequest;
                const minInterval = 10000; // 10 seconds
                if (timeSinceLastRequest < minInterval) {
                    await new Promise(resolve => setTimeout(resolve, minInterval - timeSinceLastRequest));
                }
            }
        }
    }
}
//# sourceMappingURL=BlastApiClient.js.map