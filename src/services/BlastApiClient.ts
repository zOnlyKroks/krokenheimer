import fetch from "node-fetch";
import * as zlib from "zlib";
import {promisify} from "util";
import type {BlastHit, BlastResults, DNASequence} from "../types/bioinformatics.js";

interface BlastParameters {
    wordSize: number;
    expect: string;
    hitlistSize: string;
    filter: string;
    dust: string;
    matchScore?: number;
    mismatchScore?: number;
    gapCosts?: string;
}

export class BlastApiClient {
    private readonly baseUrl = "https://blast.ncbi.nlm.nih.gov/Blast.cgi";
    private readonly statusCheckRetries = 3;

    public async analyzeSequence(sequence: DNASequence): Promise<BlastResults> {
        // 1. Submit sequence
        const rid = await this.submitSequence(sequence);

        // 2. Wait until BLAST job is ready (with timeout)
        const startTime = Date.now();
        const maxWaitTime = 300000;
        const pollInterval = 12000;

        let status: "WAITING" | "READY" | "UNKNOWN" = "WAITING";
        let unknownCount = 0;
        const maxUnknownAttempts = 3;

        while (Date.now() - startTime < maxWaitTime) {
            await this.sleep(pollInterval);

            try {
                status = await this.checkStatus(rid);

                if (status === "READY") {
                    break;
                } else if (status === "WAITING") {
                    unknownCount = 0;
                } else if (status === "UNKNOWN") {
                    unknownCount++;
                    console.warn(`[BLAST] Unknown status ${unknownCount}/${maxUnknownAttempts}`);

                    if (unknownCount >= maxUnknownAttempts) {
                        throw new Error(`Status check failed after ${unknownCount} consecutive UNKNOWN responses`);
                    }
                }
            } catch (error) {
                console.error(`[BLAST] Status check error:`, error);
                throw error;
            }
        }

        if (status !== "READY") {
            const elapsed = (Date.now() - startTime) / 1000;
            throw new Error(`BLAST job timeout: ${status} after ${elapsed.toFixed(1)}s`);
        }

        // 3. Retrieve results
        return await this.getResults(rid);
    }

    /**
     * Calculate optimal BLAST parameters based on NCBI's algorithm
     * This implements the same logic used by the NCBI web interface
     */
    private getOptimalParameters(seqLength: number): BlastParameters {
        let wordSize: number;
        let expect: string;
        let matchScore: number = 2;
        let mismatchScore: number = -3;
        let gapCosts: string = "5 2"; // Gap open, gap extend

        // NCBI's parameter selection logic based on sequence length
        if (seqLength < 30) {
            // Very short sequences - highly sensitive
            wordSize = 7;
            expect = "1000";
            matchScore = 1;
            mismatchScore = -3;
            gapCosts = "5 2";
        } else if (seqLength <= 50) {
            // Short sequences - optimized for somewhat similar sequences
            wordSize = 7;
            expect = "1000";
            matchScore = 1;
            mismatchScore = -3;
        } else if (seqLength <= 100) {
            // Medium-short sequences
            wordSize = 11;
            expect = "10";
            matchScore = 2;
            mismatchScore = -3;
        } else if (seqLength <= 200) {
            // Medium sequences
            wordSize = 11;
            expect = "10";
        } else if (seqLength <= 500) {
            // Medium-long sequences
            wordSize = 11;
            expect = "10";
        } else if (seqLength <= 1000) {
            // Long sequences
            wordSize = 11;
            expect = "10";
        } else if (seqLength <= 5000) {
            // Very long sequences - can use larger word size for speed
            wordSize = 28;
            expect = "10";
        } else {
            // Extremely long sequences - maximize speed
            wordSize = 28;
            expect = "0.01";
        }

        const params: BlastParameters = {
            wordSize,
            expect,
            hitlistSize: "100", // Increased from 50 for better results
            filter: "L", // Use low complexity filter (default NCBI setting)
            dust: "yes", // Enable DUST filtering for nucleotide sequences
            matchScore,
            mismatchScore,
            gapCosts
        };

        console.log(`[BLAST] Auto-adjusted parameters for sequence length ${seqLength}:`, {
            wordSize: params.wordSize,
            expect: params.expect,
            matchScore: params.matchScore,
            mismatchScore: params.mismatchScore,
            gapCosts: params.gapCosts
        });

        return params;
    }

    public async submitSequence(sequence: DNASequence): Promise<string> {
        const seq = sequence.cleaned || sequence.raw;

        // Enforce minimum length
        if (!seq || seq.length < 20) {
            throw new Error("Sequence must be at least 20 nucleotides long");
        }

        // Get optimal parameters based on sequence length
        const params = this.getOptimalParameters(seq.length);

        const urlParams = new URLSearchParams({
            CMD: "Put",
            PROGRAM: "blastn",
            DATABASE: "nt",
            QUERY: seq,
            FORMAT_TYPE: "JSON2",
            EXPECT: params.expect,
            HITLIST_SIZE: params.hitlistSize,
            WORD_SIZE: params.wordSize.toString(),
            FILTER: params.filter,
            DUST: params.dust,
            REWARD: params.matchScore?.toString() || "2",
            PENALTY: params.mismatchScore?.toString() || "-3",
            GAPCOSTS: params.gapCosts || "5 2"
        });

        const response = await fetch(`${this.baseUrl}?${urlParams}`, {
            method: "POST",
            headers: { "User-Agent": "Discord-Bot-Genome-Sequencer/1.0" }
        });

        if (!response.ok) {
            throw new Error(`Failed to submit BLAST sequence (${response.status})`);
        }

        const text = await response.text();
        const match = text.match(/RID\s*=\s*(\S+)/);
        if (!match) {
            throw new Error("No RID returned from BLAST");
        }

        // @ts-ignore
        return match[1];
    }


    public async checkStatus(
        rid: string
    ): Promise<"WAITING" | "READY" | "UNKNOWN"> {
        for (let attempt = 1; attempt <= this.statusCheckRetries; attempt++) {
            try {
                const params = new URLSearchParams({
                    CMD: "Get",
                    RID: rid,
                    FORMAT_TYPE: "XML"
                });

                const response = await fetch(`${this.baseUrl}?${params}`, {
                    headers: { "User-Agent": "Discord-Bot-Genome-Sequencer/1.0" }
                });

                if (!response.ok) {
                    throw new Error(`Status check failed (${response.status})`);
                }

                const text = await response.text();

                if (text.includes("Status=WAITING")) {
                    return "WAITING";
                }
                if (text.includes("Status=READY")) {
                    return "READY";
                }
                // Check if we got the actual XML results instead of status
                if (text.includes("BlastOutput") && text.includes("<?xml")) {
                    return "READY";
                }
                if (text.includes("FAILED")) {
                    throw new Error("BLAST job failed");
                }
            } catch (err) {
                if (attempt === this.statusCheckRetries) {
                    console.error(`[BLAST] All status check attempts failed, returning UNKNOWN`);
                    return "UNKNOWN";
                }

                await this.sleep(2000 * attempt);
            }
        }

        return "UNKNOWN";
    }

    /* ------------------------------------------------------------------ */
    /* RESULTS RETRIEVAL                                                   */
    /* ------------------------------------------------------------------ */

    public async getResults(rid: string): Promise<BlastResults> {
        const params = new URLSearchParams({
            CMD: "Get",
            RID: rid,
            FORMAT_TYPE: "JSON2"
        });

        const response = await fetch(`${this.baseUrl}?${params}`, {
            headers: { "User-Agent": "Discord-Bot-Genome-Sequencer/1.0" }
        });

        if (!response.ok) {
            throw new Error(`Result retrieval failed (${response.status})`);
        }

        const contentType = response.headers.get("content-type") || "";

        if (contentType.includes("application/zip")) {
            const zipBuffer = Buffer.from(await response.arrayBuffer());

            const manifestText = await this.extractFirstJsonFromZip(zipBuffer);

            const manifest = JSON.parse(manifestText);

            return this.parseBlastResults(rid, manifest);
        }

        const text = await response.text();

        const json = JSON.parse(text);

        return this.parseBlastResults(rid, json);
    }

    /* ------------------------------------------------------------------ */
    /* CORE PARSING                                                        */
    /* ------------------------------------------------------------------ */

    private async parseBlastResults(
        rid: string,
        data: any
    ): Promise<BlastResults> {
        const results: BlastResults = {
            requestId: rid,
            querySequence: "",
            queryLength: 0,
            database: "nt",
            program: "blastn",
            hits: [],
            timestamp: Date.now(),
            executionTime: 0,
            status: "completed"
        };

        let search: any | null = null;

        if (data.BlastJSON?.[0]?.File) {
            const file = data.BlastJSON[0].File;

            try {
                const fetched = await this.fetchBlastJsonFile(rid, file);

                // Check if external file has actual results
                search = fetched.BlastOutput2?.[0]?.report?.results?.search;

                if (fetched.BlastJSON) {

                    const baseFile = file.replace('_1.json', '.json');

                    try {
                        search = (await this.fetchBlastJsonFile(rid, baseFile)).BlastOutput2?.[0]?.report?.results?.search;
                    } catch (baseError) {
                        console.error(`[BLAST] Base file ${baseFile} also failed:`, baseError);
                    }

                    if (!search) {
                        try {
                            const xmlResults = await this.getXMLResults(rid);
                            if (xmlResults) {
                                return xmlResults;
                            }
                        } catch (xmlError) {
                            console.error(`[BLAST] XML fallback also failed:`, xmlError);
                        }
                    }
                }
            } catch (error) {
                console.error(`[BLAST] Failed to fetch external file ${file}:`, error);
                throw error;
            }
        }

        if (!search) {
            throw new Error("No BLAST search results found");
        }

        results.querySequence =
            search.query_title || search["query-def"] || "";
        results.queryLength = search.query_len || search["query-len"] || 0;

        const hits = search.hits || [];

        results.hits = hits
            .slice(0, 10)
            .map((h: any) => {
                return this.parseBlastHit(h);
            })
            .filter(Boolean) as BlastHit[];


        return results;
    }

    /**
     * Get BLAST results in XML format as fallback for short sequences
     */
    private async getXMLResults(rid: string): Promise<BlastResults | null> {
        const params = new URLSearchParams({
            CMD: "Get",
            RID: rid,
            FORMAT_TYPE: "XML"
        });

        const response = await fetch(`${this.baseUrl}?${params}`, {
            headers: { "User-Agent": "Discord-Bot-Genome-Sequencer/1.0" }
        });

        if (!response.ok) {
            return null;
        }

        const xmlText = await response.text();

        // Parse XML to extract actual hits
        const results: BlastResults = {
            requestId: rid,
            querySequence: "Sequence",
            queryLength: 0,
            database: "nt",
            program: "blastn",
            hits: [],
            timestamp: Date.now(),
            executionTime: 0,
            status: "completed"
        };

        try {
            const hitMatches = xmlText.match(/<Hit>(.*?)<\/Hit>/gs);

            if (hitMatches) {
                for (const hitXml of hitMatches.slice(0, 10)) {
                    const accession = hitXml.match(/<Hit_accession>(.*?)<\/Hit_accession>/)?.[1] || 'Unknown';
                    const def = hitXml.match(/<Hit_def>(.*?)<\/Hit_def>/)?.[1] || 'Unknown';

                    const hspMatch = hitXml.match(/<Hsp>(.*?)<\/Hsp>/s);
                    if (hspMatch) {
                        // @ts-ignore
                        const evalue = parseFloat(hspMatch[1].match(/<Hsp_evalue>(.*?)<\/Hsp_evalue>/)?.[1] || '1');
                        // @ts-ignore
                        const bitScore = parseFloat(hspMatch[1].match(/<Hsp_bit-score>(.*?)<\/Hsp_bit-score>/)?.[1] || '0');
                        // @ts-ignore
                        const identity = parseInt(hspMatch[1].match(/<Hsp_identity>(\d+)<\/Hsp_identity>/)?.[1] || '0');
                        // @ts-ignore
                        const alignLen = parseInt(hspMatch[1].match(/<Hsp_align-len>(\d+)<\/Hsp_align-len>/)?.[1] || '0');

                        const hit = {
                            accession,
                            description: def,
                            scientificName: this.extractSpeciesNames(def).scientificName,
                            commonName: this.extractSpeciesNames(def).commonName,
                            eValue: evalue,
                            bitScore,
                            identity: alignLen > 0 ? (identity / alignLen) * 100 : 0,
                            coverage: 50, // Rough estimate
                            alignmentLength: alignLen,
                            taxonId: undefined
                        };

                        results.hits.push(hit);
                    }
                }
            }

            const queryDefMatch = xmlText.match(/<BlastOutput_query-def>(.*?)<\/BlastOutput_query-def>/);
            if (queryDefMatch) {
                // @ts-ignore
                results.querySequence = queryDefMatch[1];
            }

            const queryLenMatch = xmlText.match(/<BlastOutput_query-len>(\d+)<\/BlastOutput_query-len>/);
            if (queryLenMatch) {
                // @ts-ignore
                results.queryLength = parseInt(queryLenMatch[1]);
            }
        } catch (error) {
            console.error(`[BLAST] Error parsing XML results:`, error);
            console.log(`[BLAST] XML content that failed to parse:`, xmlText.substring(0, 1000));
        }

        return results;
    }

    /* ------------------------------------------------------------------ */
    /* FETCH REFERENCED RESULT FILE                                        */
    /* ------------------------------------------------------------------ */

    private async fetchBlastJsonFile(
        rid: string,
        file: string
    ): Promise<any> {
        const params = new URLSearchParams({
            CMD: "Get",
            RID: rid,
            FORMAT_TYPE: "JSON2",
            FORMAT_FILE: file
        });

        const response = await fetch(`${this.baseUrl}?${params}`, {
            headers: { "User-Agent": "Discord-Bot-Genome-Sequencer/1.0" }
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch BLAST file ${file}`);
        }

        const contentType = response.headers.get("content-type") || "";

        if (contentType.includes("application/zip")) {
            const zipBuffer = Buffer.from(await response.arrayBuffer());

            const manifestText = await this.extractFirstJsonFromZip(zipBuffer);

            return JSON.parse(manifestText);
        }

        const text = await response.text();

        // Check if text response is actually a ZIP file (starts with PK)
        if (text.charCodeAt(0) === 80 && text.charCodeAt(1) === 75) {
            const zipBuffer = Buffer.from(text, 'latin1');
            const manifestText = await this.extractFirstJsonFromZip(zipBuffer);

            return JSON.parse(manifestText);
        }

        return JSON.parse(text);
    }

    /* ------------------------------------------------------------------ */
    /* HIT PARSING                                                         */
    /* ------------------------------------------------------------------ */

    private parseBlastHit(hit: any): BlastHit | null {
        try {
            const desc = hit.description?.[0];
            const hsp = hit.hsps?.[0];
            if (!desc || !hsp) return null;

            const { scientificName, commonName } =
                this.extractSpeciesNames(desc.title || "");

            return {
                accession: desc.accession,
                description: desc.title,
                scientificName,
                commonName,
                eValue: hsp.evalue,
                bitScore: hsp.bit_score,
                identity:
                    (hsp.identities / hsp.align_len) * 100,
                coverage:
                    (hsp.align_len /
                        (Math.abs(hsp.query_to - hsp.query_from) + 1)) *
                    100,
                alignmentLength: hsp.align_len,
                taxonId: desc.taxid
            };
        } catch {
            return null;
        }
    }

    private extractSpeciesNames(desc: string) {
        const sci = desc.match(/^([A-Z][a-z]+ [a-z]+)/)?.[1] || "Unknown species";
        const common = desc.match(/\[([^\]]+)\]/)?.[1];
        return { scientificName: sci, commonName: common };
    }

    /* ------------------------------------------------------------------ */
    /* ZIP UTIL (MANIFEST ONLY)                                            */
    /* ------------------------------------------------------------------ */

    private async extractFirstJsonFromZip(buffer: Buffer): Promise<string> {
        const inflate = promisify(zlib.inflateRaw);

        let offset = 0;
        while (offset < buffer.length - 30) {
            if (buffer.readUInt32LE(offset) === 0x04034b50) {
                const nameLen = buffer.readUInt16LE(offset + 26);
                const extraLen = buffer.readUInt16LE(offset + 28);
                const dataStart = offset + 30 + nameLen + extraLen;

                let next = buffer.indexOf(
                    Buffer.from([0x50, 0x4b, 0x03, 0x04]),
                    dataStart
                );
                if (next === -1) next = buffer.length;

                const compressed = buffer.subarray(dataStart, next);
                const decompressed = await inflate(compressed);
                return decompressed.toString("utf8");
            }
            offset++;
        }

        throw new Error("No JSON found in ZIP");
    }


    private sleep(ms: number) {
        return new Promise(r => setTimeout(r, ms));
    }
}