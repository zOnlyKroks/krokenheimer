import fetch from "node-fetch";
import * as zlib from "zlib";
import { promisify } from "util";
export class BlastApiClient {
    baseUrl = "https://blast.ncbi.nlm.nih.gov/Blast.cgi";
    statusCheckRetries = 3;
    async analyzeSequence(sequence) {
        // 1. Submit sequence
        const rid = await this.submitSequence(sequence);
        // 2. Wait until BLAST job is ready
        let status = "WAITING";
        while (status === "WAITING") {
            await this.sleep(5000); // wait 5 seconds between checks
            status = await this.checkStatus(rid);
        }
        if (status !== "READY") {
            throw new Error("BLAST job did not complete successfully");
        }
        // 3. Retrieve results
        return await this.getResults(rid);
    }
    async submitSequence(sequence) {
        const params = new URLSearchParams({
            CMD: "Put",
            PROGRAM: "blastn",
            DATABASE: "nt",
            QUERY: sequence.cleaned || sequence.raw,
            FORMAT_TYPE: "JSON2"
        });
        const response = await fetch(`${this.baseUrl}?${params}`, {
            method: "POST",
            headers: { "User-Agent": "Discord-Bot-Genome-Sequencer/1.0" }
        });
        if (!response.ok) {
            throw new Error(`Failed to submit BLAST sequence (${response.status})`);
        }
        const text = await response.text();
        const match = text.match(/RID\s*=\s*(\S+)/);
        if (!match)
            throw new Error("No RID returned from BLAST");
        // @ts-ignore
        return match[1];
    }
    async checkStatus(rid) {
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
                if (text.includes("Status=WAITING"))
                    return "WAITING";
                if (text.includes("Status=READY"))
                    return "READY";
                if (text.includes("FAILED"))
                    throw new Error("BLAST job failed");
            }
            catch (err) {
                if (attempt === this.statusCheckRetries)
                    return "UNKNOWN";
                await this.sleep(2000 * attempt);
            }
        }
        return "UNKNOWN";
    }
    /* ------------------------------------------------------------------ */
    /* RESULTS RETRIEVAL                                                   */
    /* ------------------------------------------------------------------ */
    async getResults(rid) {
        console.log(`[BLAST] Retrieving results for ${rid}`);
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
    async parseBlastResults(rid, data) {
        const results = {
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
        let search = null;
        // Normal JSON2
        if (data.BlastOutput2) {
            search = data.BlastOutput2[0]?.report?.results?.search;
        }
        // Manifest-only JSON
        else if (data.BlastJSON?.[0]?.File) {
            const file = data.BlastJSON[0].File;
            console.log(`[BLAST] Manifest points to ${file}, fetching externally`);
            const fetched = await this.fetchBlastJsonFile(rid, file);
            search = fetched.BlastOutput2?.[0]?.report?.results?.search;
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
            .map((h) => this.parseBlastHit(h))
            .filter(Boolean);
        return results;
    }
    /* ------------------------------------------------------------------ */
    /* FETCH REFERENCED RESULT FILE                                        */
    /* ------------------------------------------------------------------ */
    async fetchBlastJsonFile(rid, file) {
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
        return JSON.parse(await response.text());
    }
    /* ------------------------------------------------------------------ */
    /* HIT PARSING                                                         */
    /* ------------------------------------------------------------------ */
    parseBlastHit(hit) {
        try {
            const desc = hit.description?.[0];
            const hsp = hit.hsps?.[0];
            if (!desc || !hsp)
                return null;
            const { scientificName, commonName } = this.extractSpeciesNames(desc.title || "");
            return {
                accession: desc.accession,
                description: desc.title,
                scientificName,
                commonName,
                eValue: hsp.evalue,
                bitScore: hsp.bit_score,
                identity: (hsp.identities / hsp.align_len) * 100,
                coverage: (hsp.align_len /
                    (Math.abs(hsp.query_to - hsp.query_from) + 1)) *
                    100,
                alignmentLength: hsp.align_len,
                taxonId: desc.taxid
            };
        }
        catch {
            return null;
        }
    }
    extractSpeciesNames(desc) {
        const sci = desc.match(/^([A-Z][a-z]+ [a-z]+)/)?.[1] || "Unknown species";
        const common = desc.match(/\[([^\]]+)\]/)?.[1];
        return { scientificName: sci, commonName: common };
    }
    /* ------------------------------------------------------------------ */
    /* ZIP UTIL (MANIFEST ONLY)                                            */
    /* ------------------------------------------------------------------ */
    async extractFirstJsonFromZip(buffer) {
        const inflate = promisify(zlib.inflateRaw);
        let offset = 0;
        while (offset < buffer.length - 30) {
            if (buffer.readUInt32LE(offset) === 0x04034b50) {
                const nameLen = buffer.readUInt16LE(offset + 26);
                const extraLen = buffer.readUInt16LE(offset + 28);
                const dataStart = offset + 30 + nameLen + extraLen;
                let next = buffer.indexOf(Buffer.from([0x50, 0x4b, 0x03, 0x04]), dataStart);
                if (next === -1)
                    next = buffer.length;
                const compressed = buffer.subarray(dataStart, next);
                const decompressed = await inflate(compressed);
                return decompressed.toString("utf8");
            }
            offset++;
        }
        throw new Error("No JSON found in ZIP");
    }
    sleep(ms) {
        return new Promise(r => setTimeout(r, ms));
    }
}
//# sourceMappingURL=BlastApiClient.js.map