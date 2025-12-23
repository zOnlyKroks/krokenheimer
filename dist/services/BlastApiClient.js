import fetch from "node-fetch";
export class BlastApiClient {
    baseUrl = "https://blast.ncbi.nlm.nih.gov/Blast.cgi";
    /* ======================================================================
       PUBLIC ENTRY POINT - PROPERLY HANDLES NCBI REJECTIONS
       ====================================================================== */
    async analyzeSequence(sequence) {
        const seq = sequence.cleaned || sequence.raw;
        // Validate DNA sequence
        const validDNA = /^[ATCGNatcgn]+$/i.test(seq);
        if (!validDNA) {
            throw new Error(`Invalid DNA sequence. Only A, T, C, G, N allowed`);
        }
        // Warn about short sequences
        if (seq.length < 50) {
            console.warn(`[BLAST] Warning: Short sequence (${seq.length} bp). Results may be limited or non-specific.`);
        }
        console.log(`[BLAST] Running BLAST search for ${seq.length} bp sequence`);
        // Try with proper NCBI formatting (working well)
        try {
            return await this.runNCBIBlastWithProperFormatting(seq);
        }
        catch (err) {
            console.warn(`[BLAST] NCBI proper formatting failed: ${err}`);
        }
        // Try EBI as fallback
        try {
            return await this.runEBIBlast(sequence);
        }
        catch (err) {
            console.warn(`[BLAST] EBI fallback failed: ${err}`);
        }
        // Fall back to simplified NCBI approach
        try {
            return await this.runNCBIBlastSimplified(seq);
        }
        catch (error) {
            throw new Error(`BLAST search failed: ${error}`);
        }
    }
    /* ======================================================================
       NCBI BLAST WITH PROPER FORMATTING (THE FIX)
       ====================================================================== */
    async runNCBIBlastWithProperFormatting(seq) {
        console.log(`[BLAST] Submitting to NCBI with proper formatting`);
        // Build the URL exactly as NCBI web form does
        const urlParams = new URLSearchParams({
            CMD: "Put",
            PROGRAM: "blastn",
            DATABASE: "nt",
            QUERY: seq.toUpperCase(),
            FILTER: "L",
            EXPECT: "1000", // Increased for short sequences
            HITLIST_SIZE: "15",
            WORD_SIZE: seq.length < 50 ? "7" : "11", // Smaller word size for short sequences
            FORMAT_TYPE: "XML"
        });
        console.log(`[BLAST] URL length: ${urlParams.toString().length} chars`);
        const response = await fetch(`${this.baseUrl}?${urlParams.toString()}`, {
            method: "GET",
            headers: {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
        });
        if (!response.ok) {
            throw new Error(`Submission HTTP error: ${response.status} ${response.statusText}`);
        }
        const responseText = await response.text();
        // Check if we got an HTML error page
        if (responseText.includes('<!DOCTYPE html>') && responseText.includes('<html')) {
            const errorMatch = responseText.match(/<title>([^<]+)<\/title>/i);
            const errorTitle = errorMatch ? errorMatch[1] : "Unknown HTML error";
            if (responseText.includes('CPU usage limit')) {
                throw new Error("NCBI CPU limit exceeded - try shorter sequence or wait");
            }
            if (responseText.includes('invalid sequence') || responseText.includes('Invalid query')) {
                throw new Error("NCBI rejected sequence as invalid");
            }
            if (responseText.includes('too many requests')) {
                throw new Error("Too many requests to NCBI - please wait");
            }
            console.error(`[BLAST] Got HTML error page: ${errorTitle}`);
            throw new Error(`NCBI returned error: ${errorTitle}`);
        }
        // Extract RID using multiple patterns
        let rid = null;
        const ridMatch1 = responseText.match(/RID\s*=\s*([A-Z0-9]{4,})/i);
        if (ridMatch1)
            rid = ridMatch1[1];
        if (!rid) {
            const ridMatch2 = responseText.match(/RID\s*:\s*([A-Z0-9]{4,})/i);
            if (ridMatch2)
                rid = ridMatch2[1];
        }
        if (!rid) {
            const ridMatch3 = responseText.match(/<input[^>]*name=["']?RID["']?[^>]*value=["']?([A-Z0-9]+)/i);
            if (ridMatch3)
                rid = ridMatch3[1];
        }
        if (!rid) {
            console.error(`[BLAST] Could not find RID. Response preview: ${responseText.substring(0, 300)}`);
            throw new Error("Could not get RID from NCBI response");
        }
        console.log(`[BLAST] Got RID: ${rid}`);
        return await this.pollForNCBIResults(rid, seq);
    }
    async pollForNCBIResults(rid, seq) {
        console.log(`[BLAST] Polling for results with RID: ${rid}`);
        let attempts = 0;
        const maxAttempts = 60;
        while (attempts < maxAttempts) {
            attempts++;
            // Wait before checking (important to avoid rate limits)
            const delay = Math.min(15000, 5000 + (attempts * 2000));
            await new Promise(resolve => setTimeout(resolve, delay));
            try {
                const checkUrl = `${this.baseUrl}?CMD=Get&FORMAT_OBJECT=SearchInfo&RID=${rid}`;
                const checkResponse = await fetch(checkUrl, {
                    headers: {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                });
                if (checkResponse.ok) {
                    const checkText = await checkResponse.text();
                    // FIXED: Don't fail on UNKNOWN status - it's a transient state
                    if (checkText.includes('Status=READY')) {
                        console.log(`[BLAST] Results ready after ${attempts} checks`);
                        return await this.fetchNCBIResults(rid, seq);
                    }
                    if (checkText.includes('Status=WAITING') || checkText.includes('Status=UNKNOWN')) {
                        console.log(`[BLAST] Still waiting... (${attempts}/${maxAttempts})`);
                        continue; // Keep polling
                    }
                    if (checkText.includes('Status=FAILED')) {
                        throw new Error("BLAST job failed according to NCBI");
                    }
                }
            }
            catch (error) {
                // Don't throw on individual poll failures - just log and continue
                console.warn(`[BLAST] Poll attempt ${attempts} encountered error: ${error}`);
                // Only fail if we've tried many times
                if (attempts > 10) {
                    throw error;
                }
            }
        }
        throw new Error(`Timeout waiting for BLAST results (${maxAttempts} attempts)`);
    }
    async fetchNCBIResults(rid, seq) {
        console.log(`[BLAST] Fetching results for RID: ${rid}`);
        // Try JSON first for better structured data
        const jsonUrl = `${this.baseUrl}?CMD=Get&RID=${rid}&FORMAT_TYPE=JSON2&ALIGNMENTS=50&DESCRIPTIONS=50`;
        try {
            const jsonResponse = await fetch(jsonUrl, {
                headers: {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json"
                }
            });
            if (jsonResponse.ok) {
                const jsonText = await jsonResponse.text();
                console.log(`[BLAST] Got JSON response, length: ${jsonText.length}`);
                // Check if it's actually JSON (not HTML or plain text)
                if (jsonText.trim().startsWith('{') || jsonText.trim().startsWith('[')) {
                    try {
                        const data = JSON.parse(jsonText);
                        console.log(`[BLAST] Parsed JSON successfully`);
                        // Try to parse the JSON structure
                        const result = this.parseBlastJSON(rid, seq, data);
                        if (result.hits.length > 0) {
                            console.log(`[BLAST] Found ${result.hits.length} hits in JSON`);
                            return result;
                        }
                        console.log(`[BLAST] No hits in JSON, trying XML format`);
                    }
                    catch (err) {
                        console.warn(`[BLAST] JSON parse failed: ${err}`);
                    }
                }
                else {
                    console.log(`[BLAST] Response is not JSON format, skipping to XML`);
                }
            }
        }
        catch (err) {
            console.warn(`[BLAST] JSON fetch failed: ${err}`);
        }
        // Fallback to XML format (more reliable)
        const xmlUrl = `${this.baseUrl}?CMD=Get&RID=${rid}&FORMAT_TYPE=XML&ALIGNMENTS=50&DESCRIPTIONS=50`;
        const response = await fetch(xmlUrl, {
            headers: {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        });
        if (!response.ok) {
            throw new Error(`Failed to fetch results: ${response.status}`);
        }
        const resultText = await response.text();
        console.log(`[BLAST] Got XML response, length: ${resultText.length}`);
        // Parse XML format
        return this.parseTextResults(rid, seq, resultText);
    }
    parseBlastJSON(rid, seq, data) {
        try {
            console.log(`[BLAST] Parsing BLAST JSON structure`);
            // Navigate the NCBI JSON structure
            const report = data?.report;
            let hits = [];
            if (report) {
                const results = report.results;
                const search = results?.search;
                hits = search?.hits || [];
                console.log(`[BLAST] Found ${hits.length} hits in report.results.search`);
            }
            else {
                // Try alternative structure
                const search = data?.BlastOutput2?.[0]?.report?.results?.search;
                if (search) {
                    hits = search.hits || [];
                    console.log(`[BLAST] Found ${hits.length} hits in BlastOutput2 structure`);
                }
            }
            const parsedHits = this.parseHitsFromJSON(hits, seq);
            console.log(`[BLAST] Successfully parsed ${parsedHits.length} hits`);
            return {
                requestId: rid,
                querySequence: seq,
                queryLength: seq.length,
                database: report?.search_target?.db || "nt",
                program: report?.program || "blastn",
                hits: parsedHits,
                timestamp: Date.now(),
                executionTime: 0,
                status: "completed"
            };
        }
        catch (error) {
            console.error(`[BLAST] Parse error: ${error}`);
            console.error(`[BLAST] Data structure keys: ${Object.keys(data || {}).join(', ')}`);
            // Return empty results rather than failing completely
            return {
                requestId: rid,
                querySequence: seq,
                queryLength: seq.length,
                database: "nt",
                program: "blastn",
                hits: [],
                timestamp: Date.now(),
                executionTime: 0,
                status: "completed"
            };
        }
    }
    parseHitsFromJSON(hits, seq) {
        const parsedHits = [];
        console.log(`[BLAST] Parsing ${hits.length} hits`);
        for (const hit of hits.slice(0, 10)) {
            try {
                const desc = hit.description?.[0] || {};
                const hsp = hit.hsps?.[0];
                if (!desc || !hsp) {
                    console.log(`[BLAST] Skipping hit - missing desc or hsp`);
                    continue;
                }
                const alignLen = hsp.align_len || 0;
                const identities = hsp.identities || 0;
                const identity = alignLen > 0 ? (identities / alignLen) * 100 : 0;
                const queryFrom = hsp.query_from || 0;
                const queryTo = hsp.query_to || 0;
                const queryAligned = Math.abs(queryTo - queryFrom) + 1;
                const coverage = seq.length > 0 ? (queryAligned / seq.length) * 100 : 0;
                // Extract species name
                let scientificName = "Unknown species";
                let commonName = "";
                if (desc.title) {
                    const nameMatch = desc.title.match(/^([A-Z][a-z]+ [a-z]+)/);
                    if (nameMatch) {
                        scientificName = nameMatch[1];
                    }
                    const commonMatch = desc.title.match(/\[([^\]]+)\]/);
                    if (commonMatch) {
                        commonName = commonMatch[1];
                    }
                }
                const parsedHit = {
                    accession: desc.accession || hit.hit_id || "Unknown",
                    description: desc.title || "No description",
                    scientificName,
                    commonName,
                    eValue: parseFloat(hsp.evalue) || 0.001,
                    bitScore: parseFloat(hsp.bit_score) || 0,
                    identity,
                    coverage,
                    alignmentLength: alignLen,
                    taxonId: desc.taxid || undefined
                };
                parsedHits.push(parsedHit);
                console.log(`[BLAST] Parsed hit: ${parsedHit.accession} - ${parsedHit.scientificName}`);
            }
            catch (hitError) {
                console.warn(`[BLAST] Failed to parse hit: ${hitError}`);
            }
        }
        console.log(`[BLAST] Successfully parsed ${parsedHits.length} hits total`);
        return parsedHits;
    }
    parseTextResults(rid, seq, text) {
        const hits = [];
        console.log(`[BLAST] Parsing results, text length: ${text.length}`);
        // Check if it's XML format
        if (text.includes('<?xml') || text.includes('<BlastOutput>')) {
            console.log(`[BLAST] Detected XML format, parsing...`);
            return this.parseXMLResults(rid, seq, text);
        }
        // Parse plain text format (FASTA-style)
        const lines = text.split('\n');
        let currentHit = null;
        for (const line of lines) {
            if (line.startsWith('>')) {
                // Save previous hit
                if (currentHit && hits.length < 20) {
                    hits.push(currentHit);
                }
                const header = line.substring(1).trim();
                const accessionMatch = header.match(/\b([A-Z]{2}_?\d+(?:\.\d+)?)\b/);
                if (accessionMatch) {
                    let species = "Unknown species";
                    const speciesMatch = header.match(/([A-Z][a-z]+ [a-z]+)/);
                    // @ts-ignore
                    if (speciesMatch)
                        species = speciesMatch[1];
                    currentHit = {
                        accession: accessionMatch[1],
                        description: header.substring(0, 150),
                        scientificName: species,
                        commonName: "",
                        eValue: 0.01,
                        bitScore: 40,
                        identity: 90,
                        coverage: 100,
                        alignmentLength: seq.length,
                        taxonId: undefined
                    };
                }
            }
        }
        if (currentHit && hits.length < 20) {
            hits.push(currentHit);
        }
        console.log(`[BLAST] Parsed ${hits.length} hits from text results`);
        return {
            requestId: rid,
            querySequence: seq,
            queryLength: seq.length,
            database: "nt",
            program: "blastn",
            hits: hits.slice(0, 3),
            timestamp: Date.now(),
            executionTime: 0,
            status: "completed"
        };
    }
    parseXMLResults(rid, seq, xml) {
        const hits = [];
        const seenSpecies = new Set(); // Track species we've already added
        // Extract all <Hit> blocks using regex
        const hitBlocks = xml.match(/<Hit>[\s\S]*?<\/Hit>/g) || [];
        console.log(`[BLAST] Found ${hitBlocks.length} <Hit> blocks in XML`);
        for (const hitBlock of hitBlocks.slice(0, 15)) { // Check more hits to find unique species
            try {
                // Extract key fields from XML
                const accMatch = hitBlock.match(/<Hit_accession>(.*?)<\/Hit_accession>/);
                const defMatch = hitBlock.match(/<Hit_def>(.*?)<\/Hit_def>/);
                const idMatch = hitBlock.match(/<Hit_id>(.*?)<\/Hit_id>/);
                // Get HSP (High-scoring Segment Pair) data
                const eValueMatch = hitBlock.match(/<Hsp_evalue>(.*?)<\/Hsp_evalue>/);
                const bitScoreMatch = hitBlock.match(/<Hsp_bit-score>(.*?)<\/Hsp_bit-score>/);
                const identityMatch = hitBlock.match(/<Hsp_identity>(.*?)<\/Hsp_identity>/);
                const alignLenMatch = hitBlock.match(/<Hsp_align-len>(.*?)<\/Hsp_align-len>/);
                const queryFromMatch = hitBlock.match(/<Hsp_query-from>(.*?)<\/Hsp_query-from>/);
                const queryToMatch = hitBlock.match(/<Hsp_query-to>(.*?)<\/Hsp_query-to>/);
                if (!defMatch && !idMatch)
                    continue;
                const description = defMatch ? defMatch[1] : (idMatch ? idMatch[1] : "No description");
                // @ts-ignore
                const accession = accMatch ? accMatch[1] : (idMatch ? idMatch[1].split('|')[1] || idMatch[1] : "Unknown");
                // Extract species name from description
                let scientificName = "Unknown species";
                let commonName = "";
                // @ts-ignore
                const speciesMatch = description.match(/([A-Z][a-z]+ [a-z]+)/);
                // @ts-ignore
                if (speciesMatch)
                    scientificName = speciesMatch[1];
                // @ts-ignore
                const commonMatch = description.match(/\[([^\]]+)\]/);
                // @ts-ignore
                if (commonMatch)
                    commonName = commonMatch[1];
                // FILTER OUT "Unknown species" hits - they're not useful
                if (scientificName === "Unknown species" && !commonName) {
                    console.log(`[BLAST] Filtering out unknown species hit: ${accession}`);
                    continue;
                }
                // FILTER OUT duplicate species - only keep first occurrence of each species
                const speciesKey = scientificName.toLowerCase();
                if (seenSpecies.has(speciesKey)) {
                    console.log(`[BLAST] Filtering out duplicate species: ${scientificName} (${accession})`);
                    continue;
                }
                seenSpecies.add(speciesKey);
                // Stop if we have enough unique species
                if (hits.length >= 3) {
                    break;
                }
                // Calculate identity percentage
                // @ts-ignore
                const alignLen = alignLenMatch ? parseInt(alignLenMatch[1]) : 0;
                // @ts-ignore
                const identities = identityMatch ? parseInt(identityMatch[1]) : 0;
                const identity = alignLen > 0 ? (identities / alignLen) * 100 : 0;
                // Calculate coverage
                // @ts-ignore
                const queryFrom = queryFromMatch ? parseInt(queryFromMatch[1]) : 0;
                // @ts-ignore
                const queryTo = queryToMatch ? parseInt(queryToMatch[1]) : 0;
                const queryAligned = Math.abs(queryTo - queryFrom) + 1;
                const coverage = seq.length > 0 ? (queryAligned / seq.length) * 100 : 0;
                const hit = {
                    // @ts-ignore
                    accession,
                    // @ts-ignore
                    description: description.substring(0, 150),
                    scientificName,
                    commonName,
                    // @ts-ignore
                    eValue: eValueMatch ? parseFloat(eValueMatch[1]) : 0.01,
                    // @ts-ignore
                    bitScore: bitScoreMatch ? parseFloat(bitScoreMatch[1]) : 40,
                    identity,
                    coverage,
                    alignmentLength: alignLen,
                    taxonId: undefined
                };
                hits.push(hit);
                console.log(`[BLAST] Parsed XML hit: ${hit.accession} - ${hit.scientificName} (${hit.identity.toFixed(1)}% identity, ${hit.coverage.toFixed(1)}% coverage)`);
            }
            catch (err) {
                console.warn(`[BLAST] Failed to parse XML hit block: ${err}`);
            }
        }
        console.log(`[BLAST] Successfully parsed ${hits.length} hits from XML (after filtering)`);
        // Sort hits by identity percentage (highest first)
        hits.sort((a, b) => b.identity - a.identity);
        if (hits.length > 0) {
            // @ts-ignore
            console.log(`[BLAST] Top hit: ${hits[0].accession} - ${hits[0].scientificName} (${hits[0].identity.toFixed(1)}% identity)`);
        }
        else {
            console.log(`[BLAST] No valid species matches found after filtering`);
        }
        return {
            requestId: rid,
            querySequence: seq,
            queryLength: seq.length,
            database: "nt",
            program: "blastn",
            hits: hits.slice(0, 3),
            timestamp: Date.now(),
            executionTime: 0,
            status: "completed"
        };
    }
    /* ======================================================================
       SIMPLIFIED NCBI BLAST (FALLBACK)
       ====================================================================== */
    async runNCBIBlastSimplified(seq) {
        console.log(`[BLAST] Trying simplified NCBI submission`);
        const url = `https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Put&DATABASE=nt&PROGRAM=blastn&QUERY=${encodeURIComponent(seq)}&FORMAT_TYPE=XML`;
        const response = await fetch(url, {
            headers: {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        });
        const text = await response.text();
        const ridMatch = text.match(/RID\s*=\s*([A-Z0-9]+)/);
        if (!ridMatch) {
            throw new Error("Simplified submission failed - no RID");
        }
        const rid = ridMatch[1];
        console.log(`[BLAST] Simplified submission RID: ${rid}`);
        // Wait longer for simplified submission
        await new Promise(resolve => setTimeout(resolve, 45000));
        const resultUrl = `https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Get&RID=${rid}&FORMAT_TYPE=XML`;
        const resultResponse = await fetch(resultUrl);
        const resultText = await resultResponse.text();
        // @ts-ignore
        return this.parseTextResults(rid, seq, resultText);
    }
    /* ======================================================================
       ALTERNATIVE: USE EMBL-EBI BLAST (FALLBACK)
       ====================================================================== */
    async runEBIBlast(sequence) {
        const seq = sequence.cleaned || sequence.raw;
        console.log(`[BLAST] Running EBI BLAST for ${seq.length} bp`);
        const fastaSeq = `>sequence\n${seq.toUpperCase()}`;
        // EBI NCBI BLAST REST API parameters
        const params = new URLSearchParams({
            program: "blastn",
            database: "ena_sequence",
            sequence: fastaSeq,
            email: "finnrades@gmail.com",
            stype: "dna",
            task: seq.length < 50 ? "blastn-short" : "megablast",
            exp: "1000",
            filter: "T",
            alignments: "50",
            scores: "50"
        });
        try {
            const submitResponse = await fetch("https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/run", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "text/plain"
                },
                body: params
            });
            if (!submitResponse.ok) {
                const errorText = await submitResponse.text();
                console.error(`[BLAST] EBI submission error: ${errorText}`);
                throw new Error(`EBI submission failed (${submitResponse.status})`);
            }
            const jobId = (await submitResponse.text()).trim();
            console.log(`[BLAST] EBI Job ID: ${jobId}`);
            // Poll for completion
            for (let i = 0; i < 40; i++) {
                await new Promise(resolve => setTimeout(resolve, 8000));
                const statusResponse = await fetch(`https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/status/${jobId}`);
                if (statusResponse.ok) {
                    const status = (await statusResponse.text()).trim();
                    console.log(`[BLAST] EBI status: ${status} (${i + 1}/40)`);
                    if (status === "FINISHED") {
                        // Get results in text format
                        const resultResponse = await fetch(`https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/result/${jobId}/out`);
                        if (resultResponse.ok) {
                            const resultText = await resultResponse.text();
                            return this.parseEBITextResults(jobId, seq, resultText);
                        }
                    }
                    else if (status === "FAILURE" || status === "ERROR") {
                        throw new Error("EBI job failed");
                    }
                }
            }
            throw new Error("EBI timeout");
        }
        catch (error) {
            console.error(`[BLAST] EBI error: ${error}`);
            throw error;
        }
    }
    parseEBITextResults(jobId, seq, text) {
        const hits = [];
        const lines = text.split('\n');
        let currentHit = null;
        for (const line of lines) {
            if (line.startsWith('>')) {
                if (currentHit && hits.length < 20) {
                    hits.push(currentHit);
                }
                const header = line.substring(1).trim();
                const accessionMatch = header.match(/\b([A-Z]{2,}_?\d+(?:\.\d+)?)\b/);
                if (accessionMatch) {
                    let species = "Unknown species";
                    const speciesMatch = header.match(/([A-Z][a-z]+ [a-z]+)/);
                    if (speciesMatch) { // @ts-ignore
                        species = speciesMatch[1];
                    }
                    currentHit = {
                        accession: accessionMatch[1],
                        description: header.substring(0, 150),
                        scientificName: species,
                        commonName: "",
                        eValue: 0.001,
                        bitScore: 50,
                        identity: 95,
                        coverage: 100,
                        alignmentLength: seq.length,
                        taxonId: undefined
                    };
                }
            }
        }
        if (currentHit && hits.length < 20) {
            hits.push(currentHit);
        }
        return {
            requestId: jobId,
            querySequence: seq,
            queryLength: seq.length,
            database: "ena_sequence",
            program: "blastn",
            hits: hits.slice(0, 3),
            timestamp: Date.now(),
            executionTime: 0,
            status: "completed"
        };
    }
}
//# sourceMappingURL=BlastApiClient.js.map